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

r"""FlareJEPA inference on the SuperWing test split.

Loads a ``train_flarejepa.py`` checkpoint (EMA weights by default — the
weights model selection was measured on), runs the inference path
(``encode_geometry -> predict_latent -> decode_field``) on the full
``128 x 256`` grid for every test case, denormalises, and writes a
``predictions.npz`` with the same schema as the AeroJEPA ``inference.py``
so ``src.postprocessing.superwing_metrics`` scores it unchanged::

    python inference_flarejepa.py \
        checkpoint=/path/to/best.pt \
        data.path=/path/to/SuperWing_Dataset \
        inference_output_dir=/path/to/out

    python -m src.postprocessing.superwing_metrics \
        --predictions /path/to/out/predictions.npz \
        --output /path/to/out/metrics.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from physicsnemo.experimental.models.flarejepa import FlareJEPA

from src.datapipes import SuperWingDataset, superwing_collate
from train import _ensure_superwing_artifacts

log = logging.getLogger(__name__)

SUPERWING_GRID = (128, 256)


def _build_model_from_payload(payload: dict, device: torch.device, use_ema: bool):
    import inspect

    # Build from the checkpoint's stored config, filtered to the current
    # ctor signature: older checkpoints carry since-removed config keys.
    # A checkpoint whose removed knob CHANGED the architecture still fails
    # loudly below at load_state_dict(strict=True).
    stored = dict(payload["config"]["model"])
    accepted = set(inspect.signature(FlareJEPA.__init__).parameters)
    model_cfg = {k: v for k, v in stored.items() if k in accepted}
    if "decoder" in model_cfg:
        dec_accepted = {
            "cross_layers", "query_chunk_size", "head_mlp_ratio",
            "point_read", "point_neighbor_k", "use_cond",
        }
        model_cfg["decoder"] = {
            k: v for k, v in dict(model_cfg["decoder"]).items()
            if k in dec_accepted
        }
    dropped = sorted(set(stored) - accepted - {"_target_", "_convert_"})
    if dropped:
        log.info("Ignoring removed config keys from checkpoint: %s", dropped)
    model = FlareJEPA(**model_cfg).to(device)
    model.load_state_dict(payload["model"], strict=True)
    if use_ema:
        if not payload.get("ema_shadow"):
            raise ValueError(
                "use_ema=true but the checkpoint has no ema_shadow; "
                "set use_ema=false to score the live weights."
            )
        model.load_state_dict(
            {k: v.to(device) for k, v in payload["ema_shadow"].items()},
            strict=True,
        )
        log.info("Loaded EMA weights (decay=%s)", payload.get("ema_decay"))
    else:
        log.info("Loaded LIVE weights")
    return model


@hydra.main(config_path="conf", config_name="config_flarejepa", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point — FlareJEPA test-split inference."""
    ckpt_path = cfg.get("checkpoint")
    if not ckpt_path:
        raise ValueError("Pass checkpoint=/path/to/best.pt")
    out_dir = Path(str(cfg.get("inference_output_dir", "inference_flarejepa")))
    use_ema = bool(cfg.get("use_ema", True))
    max_cases = cfg.get("max_cases", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    log.info(
        "Checkpoint %s: epoch=%s best_val=%.6f",
        ckpt_path,
        payload.get("epoch"),
        payload.get("best_val_loss", float("nan")),
    )
    model = _build_model_from_payload(payload, device, use_ema).eval()
    use_normals = bool(payload["config"]["model"]["use_normals"])

    split_path, stats_path = _ensure_superwing_artifacts(cfg.data)
    dataset = SuperWingDataset(
        root_dir=str(cfg.data.path),
        split="test",
        split_manifest_path=split_path,
        normalization_stats_path=stats_path,
        surface_points=int(cfg.data.surface_points),
        target_encoder_points=int(cfg.data.target_encoder_points),
        query_points=int(cfg.data.query_points),
        eval_full_grid_query=True,
        deterministic_sampling=True,
        normalize_xyz=bool(cfg.data.normalize_xyz),
        include_normals=use_normals,
        mach_range=cfg.get("mach_range", None),
        mach_range_invert=bool(cfg.get("mach_range_invert", False)),
    )
    n_cases = len(dataset) if max_cases is None else min(int(max_cases), len(dataset))
    log.info("Test cases: %d (of %d)", n_cases, len(dataset))

    t_mean = dataset.target_mean.reshape(3, 1, 1)
    t_std = dataset.target_std.reshape(3, 1, 1)

    pred_fields, target_fields, case_ids, aoas, machs = [], [], [], [], []
    with torch.no_grad():
        for i in range(n_cases):
            sample = superwing_collate([dataset[i]])
            geo_pos = sample["context_pos"].to(device)
            geo_feat = (
                torch.cat(
                    [geo_pos, sample["context_normals"].to(device)], dim=-1
                )
                if use_normals
                else None
            )
            q_pos = sample["query_pos"].to(device)
            cond = sample["gen_params"].to(device)
            train_precision = str(
                payload["config"].get("training", {}).get("precision", "bf16")
            )
            use_autocast = device.type == "cuda" and train_precision in (
                "bf16",
                "bfloat16",
            )
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_autocast
            ):
                pred_norm = model.predict(
                    geo_pos, q_pos, cond, geometry_features=geo_feat
                )
            pred_norm = (
                pred_norm[0].float().cpu().numpy().T.reshape(3, *SUPERWING_GRID)
            )
            tgt_norm = (
                sample["query_target"][0].numpy().T.reshape(3, *SUPERWING_GRID)
            )
            pred_fields.append(pred_norm * t_std + t_mean)
            target_fields.append(tgt_norm * t_std + t_mean)
            case_ids.append(sample["case_id"][0])
            aoas.append(sample["aoa_deg"][0])
            machs.append(sample["mach"][0])
            if (i + 1) % 200 == 0:
                log.info("Processed %d/%d cases", i + 1, n_cases)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "predictions.npz"
    np.savez_compressed(
        npz_path,
        pred_field=np.stack(pred_fields, axis=0),
        target_field=np.stack(target_fields, axis=0),
        case_ids=np.asarray(case_ids),
        aoa_deg=np.asarray(aoas, dtype=np.float32),
        mach=np.asarray(machs, dtype=np.float32),
        target_mean=dataset.target_mean,
        target_std=dataset.target_std,
    )
    log.info("Saved predictions to %s (%d cases)", npz_path, n_cases)


if __name__ == "__main__":
    main()
