# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dump a Yaris time-conditional model's predicted trajectories for the OVRTX
exporter (``export_ovrtx.py``), together with the ground truth, the shell and
solid connectivity on kept nodes and the per-node penetration depths of both.

Runs inside the training environment on one GPU, through the same trainer,
datapipe and checkpoint loading as ``evaluate_yaris_tc.py``::

    torchrun --standalone --nproc_per_node=1 visualization/ovrtx/dump_yaris_predictions.py \\
        --config-name yaris_deformer \\
        yaris.data_root=... yaris.split_manifest=... yaris.outlier_mask=... \\
        yaris.work_dir=... training.ckpt_path=<run>/checkpoints +training.load_epoch=60 \\
        datapipe.contact_surface=true hydra.run.dir=<out>/hydra \\
        +ovrtx_dump.output_dir=<out> +ovrtx_dump.label=teacher \\
        '+ovrtx_dump.cases=[<case name>,...]'

Output, per case, under ``<out>/<case>/``: ``mesh.npz`` (reference geometry,
part ids, kept-node ids, thickness, shell faces, solid hexahedra, target
frames), ``ground_truth/`` and ``<label>/`` with ``positions.npy`` ``[F, N, 3]``
float32 millimetres (frame 0 is the reference state), ``depth_envelope.npy``
and ``depth_crossing.npy`` ``[F, N]`` float16 millimetres (shell-to-shell
penetration against the thickness envelope and the mid-surface crossing, from
``penetration_metrics.PenetrationMetrics``), ``wall_crossing.npy`` ``[F, N]``
(rigid wall at ``ovrtx_dump.wall_position_mm``, default 100) and ``meta.json``
with the checkpoint, the per-case vector RMSE and the frame times. Everything
is on the kept nodes of the training recipe in the order of the datapipe.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict

CRASH_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CRASH_DIR))

from datapipe import EPS  # noqa: E402
from penetration_metrics import PenetrationMetrics  # noqa: E402
from train import resolve_trainer_class  # noqa: E402

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper  # noqa: E402

DUMP_VERSION = 1


def kept_shell_faces(store, keep_ids, total_nodes):
    """Shell cells ``[F, 4]`` renumbered to kept nodes (triangles keep the repeated
    last id, as in the stores), with the shell part id of every kept cell."""
    cells = np.asarray(store["shell/connectivity"][:], dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("shell/connectivity must be [num_shells, 4]")
    remap = np.full(total_nodes, -1, dtype=np.int64)
    remap[keep_ids] = np.arange(len(keep_ids), dtype=np.int64)
    faces = remap[cells]
    keep = (faces >= 0).all(axis=1)
    part_id = (
        np.asarray(store["shell/part_id"][:], dtype=np.int64).reshape(-1)
        if "shell/part_id" in store
        else np.full(len(cells), -1, dtype=np.int64)
    )
    return (
        faces[keep].astype(np.int32),
        part_id[keep].astype(np.int32),
        int((~keep).sum()),
    )


def kept_solids(store, keep_ids, total_nodes):
    """Hexahedral cells ``[S, 8]`` renumbered to kept nodes (cells touching an
    excluded node are dropped); empty when the store has no solids."""
    if "solid/connectivity" not in store:
        return np.empty((0, 8), dtype=np.int32)
    cells = np.asarray(store["solid/connectivity"][:], dtype=np.int64)
    if cells.size == 0:
        return np.empty((0, 8), dtype=np.int32)
    remap = np.full(total_nodes, -1, dtype=np.int64)
    remap[keep_ids] = np.arange(len(keep_ids), dtype=np.int64)
    hexes = remap[cells]
    return hexes[(hexes >= 0).all(axis=1)].astype(np.int32)


def write_mesh(path, reader, store, dataset, sample, dt_ms):
    keep_ids = np.asarray(reader.keep_ids, dtype=np.int64)
    total = int(reader.total_nodes)
    faces, face_part, dropped = kept_shell_faces(store, keep_ids, total)
    part_id = (
        np.asarray(store["nodes/part_id"][:]).reshape(-1)[keep_ids].astype(np.int32)
    )
    is_wall = (
        np.asarray(store["nodes/is_wall"][:]).reshape(-1)[keep_ids].astype(bool)
        if "nodes/is_wall" in store
        else np.zeros(len(keep_ids), dtype=bool)
    )
    part_ids = (
        np.asarray(store["part/ids"][:]).reshape(-1).astype(np.int32)
        if "part/ids" in store
        else np.unique(part_id)
    )
    np.savez_compressed(
        path,
        version=DUMP_VERSION,
        reference=np.asarray(reader.reference, dtype=np.float32),
        keep_ids=keep_ids,
        total_nodes=total,
        part_id=part_id,
        part_ids=part_ids,
        is_wall=is_wall,
        thickness=np.asarray(reader.thickness(store), dtype=np.float32),
        shell_faces=faces,
        shell_part_id=face_part,
        shell_faces_dropped=dropped,
        solid_hexes=kept_solids(store, keep_ids, total),
        target_frames=np.asarray(sample.yaris_target_frames, dtype=np.int64),
        dt_ms=float(dt_ms),
        stride=int(reader.stride),
    )


def depth_fields(metrics, positions_mm, reference_mm, thickness, graph, wall):
    """Envelope, crossing and wall-crossing depth per node (float16, mm)."""
    envelope, crossing, _, _ = metrics.surface_depths(
        positions_mm, reference_mm, thickness, graph
    )
    out = [envelope, crossing]
    if wall is not None:
        out.append(metrics.wall_depths(positions_mm, thickness, wall)[1])
    else:
        out.append(torch.zeros_like(envelope))
    return [v.to(torch.float16).cpu().numpy() for v in out]


def write_trajectory(
    directory, positions_mm, reference_mm, metrics, thickness, graph, wall
):
    """``positions_mm`` [T, N, 3] (targets only); frame 0 written is the reference."""
    directory.mkdir(parents=True, exist_ok=True)
    steps, n = int(positions_mm.shape[0]), int(positions_mm.shape[1])
    all_positions = np.lib.format.open_memmap(
        directory / "positions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(steps + 1, n, 3),
    )
    names = ("depth_envelope", "depth_crossing", "wall_crossing")
    depths = [
        np.lib.format.open_memmap(
            directory / f"{name}.npy", mode="w+", dtype=np.float16, shape=(steps + 1, n)
        )
        for name in names
    ]
    frames = [reference_mm] + [positions_mm[t] for t in range(steps)]
    maxima = {name: 0.0 for name in names}
    for index, frame in enumerate(frames):
        all_positions[index] = frame.cpu().numpy()
        for array, values, name in zip(
            depths,
            depth_fields(metrics, frame, reference_mm, thickness, graph, wall),
            names,
        ):
            array[index] = values
            maxima[name] = max(maxima[name], float(values.max()))
    all_positions.flush()
    for array in depths:
        array.flush()
    return maxima


@hydra.main(
    version_base="1.3",
    config_path="../../conf",
    config_name="yaris_deformer",
)
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    logger0 = RankZeroLoggingWrapper(PythonLogger("ovrtx_dump"), dist)
    if dist.world_size != 1:
        raise RuntimeError("the dump runs on one process; shard cases by hand instead")
    dump_cfg = cfg.get("ovrtx_dump")
    if not dump_cfg or not dump_cfg.get("output_dir") or not dump_cfg.get("label"):
        raise ValueError(
            "pass +ovrtx_dump.output_dir=<dir> and +ovrtx_dump.label=<name>"
        )
    out_root = Path(dump_cfg.output_dir)
    label = str(dump_cfg.label)
    wall = dump_cfg.get("wall_position_mm", 100.0)
    wall = None if wall is None else float(wall)
    with open_dict(cfg):
        # Nothing of the run's own directories is written: logs go to the dump.
        cfg.training.tensorboard_log_dir = str(out_root / f"tensorboard_{label}")
        cfg.training.restore_rng = False
        if "validation_penetration" not in cfg.training:
            cfg.training.validation_penetration = {"enabled": True}
    trainer = resolve_trainer_class(cfg.training.get("trainer_class"))(cfg, logger0)
    if trainer.epoch_init <= 0:
        raise RuntimeError(f"No checkpoint found under {cfg.training.ckpt_path}")
    dataset = trainer.validation_dataset
    reader = dataset.reader
    names = [path.stem for path in dataset.paths]
    cases = [str(c) for c in (dump_cfg.get("cases") or names)]
    missing = sorted(set(cases) - set(names))
    if missing:
        raise ValueError(f"not validation cases: {missing}")
    model = trainer.model.eval()
    device = dist.device
    std = (trainer.data_stats["node"]["pos_std"].float() + EPS).reshape(1, 1, 3)
    mean = trainer.data_stats["node"]["pos_mean"].float().reshape(1, 1, 3)
    metrics = PenetrationMetrics(
        cfg.training.get("validation_penetration") or {}, trainer.rollout_steps, device
    )
    logger0.info(
        f"Dumping {label} (epoch {trainer.epoch_init} of {cfg.training.ckpt_path}) for "
        f"{len(cases)} case(s) into {out_root}"
    )
    for case in cases:
        started = time.monotonic()
        index = names.index(case)
        sample = dataset[index].to(device)
        graph = sample.graph
        if getattr(graph, "contact_faces", None) is None:
            raise RuntimeError("penetration depths need datapipe.contact_surface=true")
        thickness = graph.shell_thickness.to(
            device=device, dtype=torch.float32
        ).reshape(-1)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda", enabled=bool(trainer.amp), dtype=trainer.amp_dtype
            ),
        ):
            prediction = model(sample=sample, data_stats=trainer.data_stats)
            if hasattr(prediction, "trajectory"):
                prediction = prediction.trajectory
        prediction = prediction.float()
        target = sample.node_target.float()
        if prediction.shape != target.shape or target.shape[-1] != 3:
            raise ValueError(
                f"expected a displacement-only trajectory {tuple(target.shape)}, got "
                f"{tuple(prediction.shape)}"
            )
        with torch.no_grad():
            pred_mm = (
                (prediction * std + mean).transpose(0, 1).contiguous()
            )  # [T, N, 3]
            true_mm = (target * std + mean).transpose(0, 1).contiguous()
            reference_mm = (
                sample.node_features["coords"].float() * std[0] + mean[0]
            ).contiguous()
            error = (pred_mm - true_mm).double()
            vector_rmse_mm = float(torch.sqrt(error.square().sum(-1).mean()))
            rmse_per_time = (
                torch.sqrt(error.square().sum(-1).mean(dim=1)).cpu().tolist()
            )
        case_dir = out_root / case
        case_dir.mkdir(parents=True, exist_ok=True)
        store = reader.open(dataset.paths[index])
        dt_ms = reader.dt * 1000.0
        if not (case_dir / "mesh.npz").exists():
            write_mesh(case_dir / "mesh.npz", reader, store, dataset, sample, dt_ms)
        with torch.no_grad():
            if not (case_dir / "ground_truth" / "meta.json").exists():
                maxima = write_trajectory(
                    case_dir / "ground_truth",
                    true_mm,
                    reference_mm,
                    metrics,
                    thickness,
                    graph,
                    wall,
                )
                (case_dir / "ground_truth" / "meta.json").write_text(
                    json.dumps(
                        dict(
                            label="ground_truth",
                            case=case,
                            frames=[0] + [int(f) for f in sample.yaris_target_frames],
                            time_ms=[0.0]
                            + [
                                float(f) * dt_ms / reader.stride
                                for f in sample.yaris_target_frames
                            ],
                            depth_maxima_mm=maxima,
                            penetration_settings=metrics.settings(),
                            wall_position_mm=wall,
                        ),
                        indent=1,
                    )
                )
            maxima = write_trajectory(
                case_dir / label, pred_mm, reference_mm, metrics, thickness, graph, wall
            )
        meta = dict(
            version=DUMP_VERSION,
            label=label,
            case=case,
            config_name=hydra.core.hydra_config.HydraConfig.get().job.config_name,
            checkpoint_dir=str(cfg.training.ckpt_path),
            epoch=int(trainer.epoch_init),
            teacher_weights=list(cfg.model.get("teacher_weights", []) or []),
            kept_nodes=int(target.shape[0]),
            rollout_steps=int(target.shape[1]),
            frames=[0] + [int(f) for f in sample.yaris_target_frames],
            time_ms=[0.0]
            + [float(f) * dt_ms / reader.stride for f in sample.yaris_target_frames],
            vector_RMSE_mm=vector_rmse_mm,
            vector_RMSE_mm_per_time=rmse_per_time,
            depth_maxima_mm=maxima,
            penetration_settings=metrics.settings(),
            wall_position_mm=wall,
            amp=bool(trainer.amp),
            amp_dtype=str(trainer.amp_dtype),
            seconds=round(time.monotonic() - started, 1),
        )
        (case_dir / label / "meta.json").write_text(json.dumps(meta, indent=1))
        logger0.info(
            f"{label} {case}: vector RMSE {vector_rmse_mm:.3f} mm over "
            f"{target.shape[1]} frames; max crossing depth "
            f"{maxima['depth_crossing']:.2f} mm; {meta['seconds']} s"
        )
        del sample, prediction, target, pred_mm, true_mm, error
        torch.cuda.empty_cache()
    logger0.info("Dump complete")


if __name__ == "__main__":
    main()
