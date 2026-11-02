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
"""Full-resolution Yaris datapipe, trainer and recipe contracts."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))
CONF = CRASH_DIR / "conf"

zarr = pytest.importorskip("zarr")

from datapipe import KNN_EDGE_STATS_FILE  # noqa: E402
from local_neighborhoods import LocalNeighborhoodData  # noqa: E402
from rollout import MeshGeoFLAREAutoregressive  # noqa: E402
from yaris_full import (  # noqa: E402
    STATS_FILE,
    YarisFullGraphDataset,
    YarisFullReader,
    compute_frozen_stats,
    load_splits,
)

NODES, FRAMES, STRIDE = 60, 31, 5
CASES = {"train": ["A_1", "A_2", "A_3"], "validation": ["B_1", "B_2"], "test": ["C_1"]}
FRAMES_BY_CASE = {"B_2": 26}  # shorter history: 6 selected frames, 4 targets


DESIGN_MISSING_CASE = "B_2"  # one case without a design record, like 6 of the 99 runs


def _write_dataset(root):
    """Shared reference mesh; per-case thickness and motion; two wall + one flyaway node."""
    rng = np.random.default_rng(0)
    reference = rng.uniform(-50, 50, size=(NODES, 3)).astype(np.float32)
    reference[-2:] = [[500.0, 0, 0], [500.0, 10, 0]]  # rigid wall nodes, static
    is_wall = np.zeros(NODES, dtype=np.uint8)
    is_wall[-2:] = 1
    edges = np.stack((np.arange(NODES - 3), np.arange(1, NODES - 2)), axis=1)
    rows = []
    for split, names in CASES.items():
        for case_index, name in enumerate(names):
            rows.append((split, name))
            velocity = np.array([13.0 + case_index, 0.5, -0.2], dtype=np.float32)
            frames = FRAMES_BY_CASE.get(name, FRAMES)
            pos = np.empty((frames, NODES, 3), dtype=np.float32)
            for t in range(frames):
                drift = velocity * t + 0.01 * np.array([t * t, 0, 0], dtype=np.float32)
                noise = rng.normal(0, 0.05, size=(NODES, 3)) if t else 0.0
                pos[t] = reference + drift + noise  # frame 0 is the shared mesh
                pos[t, -2:] = reference[-2:]
                pos[t, 0] = reference[0] + np.array([0, 0, 300.0 * t])  # flies away
            store = zarr.open_group(str(root / f"{name}.zarr"), mode="w")
            store.create_array("nodes/pos", data=pos, chunks=(1, 17, 3))
            store.create_array(
                "nodes/thickness",
                data=rng.uniform(0.5, 3.0, size=NODES).astype(np.float32),
            )
            store.create_array("nodes/is_wall", data=is_wall)
            store.create_array(
                "nodes/part_id", data=(np.arange(NODES) // 20 + 1).astype(np.int64)
            )
            store.create_array("edges", data=edges.astype(np.int64))
            # One proper hexahedron over chain nodes 1..8 and one collapsed hex
            # (repeated ids); both reference kept nodes only.
            store.create_array(
                "solid/connectivity",
                data=np.array(
                    [[1, 2, 3, 4, 5, 6, 7, 8], [9, 9, 10, 10, 11, 11, 12, 12]],
                    dtype=np.int32,
                ),
            )
            # Shell facets: quads over consecutive chain nodes (random geometry,
            # so most split into triangles), one explicit triangle, one facet on
            # the excluded fly-away node 0 and one degenerate facet.
            shells = [[i, i + 1, i + 2, i + 3] for i in range(1, NODES - 6, 4)]
            shells += [[20, 21, 22, 22], [0, 1, 2, 3], [5, 5, 6, 7]]
            store.create_array(
                "shell/connectivity", data=np.array(shells, dtype=np.int32)
            )
            if name != DESIGN_MISSING_CASE:
                for i in range(1, 9):
                    store.attrs[f"bc_thickness_group_{i}_deviation"] = round(
                        0.05 * i * (1 if case_index % 2 else -1), 3
                    )
    with (root / "splits.csv").open("w") as stream:
        stream.write("split,sample_name\n")
        for split, name in rows:
            stream.write(f"{split},{name}.zarr\n")
    flyaway = np.zeros(NODES, dtype=bool)
    flyaway[0] = True
    np.savez(
        root / "outliers.npz",
        flyaway=flyaway,
        wall=is_wall.astype(bool),
        exclude=flyaway | is_wall.astype(bool),
    )
    return root


@pytest.fixture(scope="module")
def yaris_root(tmp_path_factory):
    return _write_dataset(tmp_path_factory.mktemp("yaris"))


def _compose(root, work_dir, overrides=()):
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose(
            config_name="crash_deformer_knn_yaris_autoregressive",
            overrides=[
                f"yaris.data_root={root}",
                f"yaris.split_manifest={root / 'splits.csv'}",
                f"yaris.outlier_mask={root / 'outliers.npz'}",
                f"yaris.work_dir={work_dir}",
                "training.num_training_samples=3",
                "training.num_validation_samples=2",
                "training.num_time_steps=7",
                "model.local_knn_k=4",
                *overrides,
            ],
        )


def test_recipe_inherits_the_knn_deformer_and_adds_nested_checkpointing(
    yaris_root, tmp_path
):
    cfg = _compose(yaris_root, tmp_path / "work")
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        gm = compose(
            config_name="crash_deformer_knn_autoregressive",
            overrides=[
                "training.raw_data_dir=/x",
                "training.raw_data_dir_validation=/x",
                "inference.raw_data_dir_test=/x",
            ],
        )
    for key in gm.model:
        if key in (
            "dt",
            "local_neighborhood",
            "local_attention_fused",
            "num_time_steps",
            "local_knn_k",  # overridden by the test fixture
        ):
            continue
        assert gm.model[key] == cfg.model[key], key
    assert cfg.model.dt == 0.005 and cfg.model.local_neighborhood == "euclidean"
    assert cfg.model.local_attention_fused is True
    assert cfg.model.activation_checkpointing is True
    assert cfg.model.checkpointing_ratio == 0.5
    assert set(cfg.model.activation_checkpointing_components) == {"blocks", "context"}
    assert cfg.model.checkpoint_rollout is True
    assert cfg.model.num_pre_processor_checkpoint_segments == 1
    assert cfg.model.num_post_processor_checkpoint_segments == 2
    for key in (
        "optimizer",
        "amp",
        "amp_dtype",
        "epochs",
        "start_lr",
        "end_lr",
        "seed",
    ):
        assert gm.training[key] == cfg.training[key], key
    assert cfg.training.deterministic is False
    assert cfg.training.trainer_class == "yaris_full_trainer.YarisFullTrainer"
    assert cfg.training.save_checkpoint_freq == 5 and cfg.training.validation_freq == 20
    assert cfg.reader._target_ == "yaris_full.YarisFullReader"
    assert cfg.datapipe._target_ == "yaris_full.YarisFullGraphDataset"
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["datapipe"]["dt"] == 0.005
    assert resolved["datapipe"]["local_knn_metric"] == "euclidean"
    assert resolved["datapipe"]["local_knn_k"] == 4
    assert resolved["reader"]["stride"] == 5


def test_split_manifest_rules(yaris_root, tmp_path):
    splits = load_splits(yaris_root / "splits.csv")
    assert splits == CASES
    bad = tmp_path / "bad.csv"
    bad.write_text("split,sample_name\ntrain,A_1\nvalidation,A_1\ntest,C\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_splits(bad)
    bad.write_text("run,name\ntrain,A_1\n")
    with pytest.raises(ValueError, match="columns split,sample_name"):
        load_splits(bad)


def test_dataset_drops_outliers_and_builds_shared_neighborhoods(yaris_root, tmp_path):
    cfg = _compose(yaris_root, tmp_path / "work")
    reader = instantiate(cfg.reader)
    dataset = instantiate(
        cfg.datapipe, reader=reader, split="train", stats_mode="compute"
    )
    kept = NODES - 3
    assert dataset.num_nodes == kept and len(dataset) == 3
    assert reader.keep_ids.tolist() == list(range(1, NODES - 2))
    graph = dataset.graph
    assert isinstance(graph, LocalNeighborhoodData)
    assert graph.num_nodes == kept and graph.num_edges == 0
    assert graph.knn_index.shape == (kept, 5) and graph.knn_mask.all()
    assert graph.knn_edge_attr.shape == (kept, 5, 4)
    assert graph.knn_metric == "euclidean"
    torch.testing.assert_close(graph.knn_index[:, 0], torch.arange(kept))
    # Neighbors never point at removed nodes (indices are in kept numbering).
    assert graph.knn_index.max() < kept
    stats_dir = tmp_path / "work" / "stats"
    assert (stats_dir / STATS_FILE).exists() and (
        stats_dir / KNN_EDGE_STATS_FILE
    ).exists()
    saved = json.loads((stats_dir / STATS_FILE).read_text())
    assert saved["identity"]["kept_nodes"] == kept
    assert saved["values"]["frames_per_case"] == [7, 7, 7]
    # Statistics ignore the fly-away and wall nodes entirely.
    direct = compute_frozen_stats(reader, dataset.paths, workers=1)
    assert direct["node"] == saved["values"]["node"]
    assert dataset.node_stats["pos_std"][2] < 100  # with the fly-away node z std ~1e3
    sample = dataset[0]
    assert sample.node_features["coords"].shape == (kept, 3)
    assert sample.node_features["previous_coords"].shape == (kept, 3)
    assert sample.node_features["features"].shape == (kept, 1)
    assert sample.node_target.shape == (kept, 4, 3)
    assert len(sample.yaris_target_frames) == 4
    assert all(torch.isfinite(v).all() for v in sample.node_features.values())
    # Windows are stateless in (seed, epoch, slot) and change with the epoch.
    first = dataset[1].yaris_target_frames
    assert dataset[1].yaris_target_frames == first
    frames = {tuple(dataset[1].yaris_target_frames)}
    for epoch in range(1, 12):
        dataset.set_epoch(epoch)
        frames.add(tuple(dataset[1].yaris_target_frames))
    assert len(frames) > 1
    assert all(f[0] >= 10 and f[-1] <= 30 for f in frames)
    # Validation covers every remaining transition and loads the published stats.
    validation = instantiate(
        cfg.datapipe,
        reader=reader,
        split="validation",
        stats_mode="load",
        sample_type="all_time_steps",
        num_samples=1,
    )
    held_out = validation[0]
    assert held_out.node_target.shape == (kept, 5, 3)
    assert held_out.yaris_target_frames == [10, 15, 20, 25, 30]
    torch.testing.assert_close(validation.graph.knn_edge_attr, graph.knn_edge_attr)
    # A changed outlier mask invalidates the frozen statistics.
    other = tmp_path / "other.npz"
    np.savez(other, exclude=np.zeros(NODES, dtype=bool) | (np.arange(NODES) == 5))
    options = OmegaConf.to_container(cfg.datapipe, resolve=True)
    options = {k: v for k, v in options.items() if not k.startswith("_")}
    with pytest.raises(ValueError, match="identity changed"):
        YarisFullGraphDataset(
            reader=YarisFullReader(yaris_root / "splits.csv", other, stride=STRIDE),
            split="train",
            stats_mode="load",
            **options,
        )


def test_reader_rejects_mismatched_geometry(yaris_root, tmp_path):
    reader = YarisFullReader(yaris_root / "splits.csv", yaris_root / "outliers.npz")
    reader.open(yaris_root / "A_1.zarr")
    moved = tmp_path / "moved.zarr"
    store = zarr.open_group(str(moved), mode="w")
    pos = np.asarray(
        zarr.open_group(str(yaris_root / "A_1.zarr"), mode="r")["nodes/pos"][:]
    )
    pos[0, 3] += 1.0
    store.create_array("nodes/pos", data=pos)
    with pytest.raises(ValueError, match="Reference geometry"):
        reader.open(moved)
    edges = reader.structural_edges(
        zarr.open_group(str(yaris_root / "A_1.zarr"), mode="r")
    )
    assert edges.shape[0] == 2 and edges.min() >= 0 and edges.max() < NODES - 3
    assert edges.shape[1] == NODES - 4  # 57 chain edges minus the one touching node 0


def _small_model(**overrides):
    kwargs = dict(
        num_time_steps=7,
        dt=0.005,
        functional_dim=4,
        out_dim=3,
        geometry_dim=3,
        global_dim=None,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        n_layers=2,
        use_te=False,
        attention_type="GALE_FA",
        input_dim_edges=4,
        mesh_hidden_dim=16,
        num_pre_processor_layers=1,
        num_post_processor_layers=2,
        num_pre_processor_checkpoint_segments=1,
        num_post_processor_checkpoint_segments=2,
        mesh_context_fusion="pre_post",
        mesh_pre_residual_gate_init=0.3,
        mesh_post_residual_gate_init=0.2,
        initial_velocity_mode="previous_coords",
        rollout_steps_from_target=True,
        checkpoint_rollout=True,
        teacher_forcing=False,
        node_input_mode="velocity",
        use_contact=False,
        enable_contact=False,
        enable_node_contact=False,
        enable_cylinder_contact=False,
        local_processor="knn_attention",
        local_knn_k=4,
        local_neighborhood="euclidean",
        local_num_heads=4,
        local_edge_dim=8,
        local_attention_fused=True,
        local_attention_backend="torch",
    )
    kwargs.update(overrides)
    torch.manual_seed(3)
    return MeshGeoFLAREAutoregressive(**kwargs)


def test_nested_block_checkpointing_matches_plain_training(yaris_root, tmp_path):
    """Checkpointed FLARE blocks inside checkpointed rollout steps: same loss and grads."""
    cfg = _compose(yaris_root, tmp_path / "work")
    reader = instantiate(cfg.reader)
    dataset = instantiate(
        cfg.datapipe, reader=reader, split="train", stats_mode="compute"
    )
    sample = dataset[0]
    stats = {"node": {k: v.view(1, 3) for k, v in dataset.node_stats.items()}}
    results = {}
    for name, flag in (("plain", False), ("nested", True)):
        model = _small_model(
            activation_checkpointing=flag,
            checkpointing_ratio=1.0,
            activation_checkpointing_components=["blocks", "context"],
        ).train()
        trajectory = model(sample=sample, data_stats=stats)
        assert trajectory.shape == sample.node_target.shape
        loss = torch.nn.functional.mse_loss(trajectory, sample.node_target)
        loss.backward()
        grads = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        assert grads, name
        results[name] = (loss.detach(), grads)
    torch.testing.assert_close(results["plain"][0], results["nested"][0])
    assert results["plain"][1].keys() == results["nested"][1].keys()
    for key in results["plain"][1]:
        torch.testing.assert_close(
            results["plain"][1][key], results["nested"][1][key], rtol=1e-4, atol=1e-6
        )
    # Evaluation ignores both checkpoint layers and runs the full horizon.
    model.eval()
    validation = instantiate(
        cfg.datapipe,
        reader=reader,
        split="validation",
        stats_mode="load",
        sample_type="all_time_steps",
        num_samples=1,
    )
    with torch.no_grad():
        prediction = model(sample=validation[0], data_stats=stats)
    assert prediction.shape == (NODES - 3, 5, 3) and torch.isfinite(prediction).all()


def test_trainer_hook_resolves_the_yaris_trainer():
    from train import Trainer, resolve_trainer_class
    from yaris_full_trainer import YarisFullTrainer

    assert resolve_trainer_class(None) is Trainer
    assert (
        resolve_trainer_class("yaris_full_trainer.YarisFullTrainer") is YarisFullTrainer
    )
    assert issubclass(YarisFullTrainer, Trainer)
    with pytest.raises(TypeError, match="not a Trainer subclass"):
        resolve_trainer_class("yaris_full_trainer.ExactShardSampler")
    # Under torchrun train.py is the script module, so its Trainer is a different
    # class object from the ``train.Trainer`` the subclass derives from. Loading
    # the file under another module name reproduces that without running main().
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_as_script", CRASH_DIR / "train.py"
    )
    as_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(as_script)
    assert as_script.Trainer is not Trainer
    assert (
        as_script.resolve_trainer_class("yaris_full_trainer.YarisFullTrainer")
        is YarisFullTrainer
    )
    assert as_script.resolve_trainer_class(None) is as_script.Trainer


def test_exact_shard_sampler_covers_every_case_once():
    from yaris_full_trainer import ExactShardSampler

    for replicas in (1, 2, 3, 16):
        shards = [
            list(ExactShardSampler(20, replicas, rank)) for rank in range(replicas)
        ]
        assert sorted(sum(shards, [])) == list(range(20))
        assert (
            max(len(shard) for shard in shards) - min(len(shard) for shard in shards)
            <= 1
        )
    assert list(ExactShardSampler(2, 3, 2)) == []  # idle rank still joins reductions


def test_exact_validation_pools_variable_horizons(yaris_root, tmp_path):
    """Two held-out cases with 5 and 4 target steps: per-step counts, pooled MSE, mm RMSE."""
    from datapipe import simsample_collate
    from torch.utils.data import DataLoader
    from yaris_full_trainer import (
        ExactShardSampler,
        accumulate_validation,
        summarize_cases,
        summarize_validation,
    )

    cfg = _compose(yaris_root, tmp_path / "work")
    reader = instantiate(cfg.reader)
    instantiate(cfg.datapipe, reader=reader, split="train", stats_mode="compute")
    validation = instantiate(
        cfg.datapipe,
        reader=reader,
        split="validation",
        stats_mode="load",
        sample_type="all_time_steps",
        num_samples=2,
    )
    assert [len(validation[i].yaris_target_frames) for i in range(2)] == [5, 4]
    stats = {"node": dict(validation.node_stats)}
    model = _small_model(activation_checkpointing=True).eval()
    kept = NODES - 3
    # Two ranks, exact partition, summed like the distributed all-reduce.
    totals = None
    seen = []
    records = []
    for rank in range(2):
        loader = DataLoader(
            validation,
            batch_size=1,
            sampler=ExactShardSampler(len(validation), 2, rank),
            collate_fn=simsample_collate,
        )
        part, names = accumulate_validation(
            model, loader, stats, 5, torch.device("cpu"), per_case=records
        )
        assert part.shape == (5, 5) and part.dtype == torch.float64
        totals = part if totals is None else totals + part
        seen += names
    assert sorted(seen) == ["B_1", "B_2"]
    torch.testing.assert_close(
        totals[:, 3], torch.tensor([2.0, 2, 2, 2, 1]) * kept, check_dtype=False
    )
    summary = summarize_validation(totals)
    # Reference: pooled over every (case, step, node, channel) entry.
    errors = []
    names = []
    with torch.no_grad():
        for i in range(2):
            sample = validation[i]
            names.append(sample.yaris_case_name)
            errors.append(
                (model(sample=sample, data_stats=stats) - sample.node_target).double()
            )
    pooled = torch.cat([e.reshape(-1) for e in errors])
    torch.testing.assert_close(
        summary["MSE"], torch.square(pooled).mean(), rtol=1e-6, atol=1e-9
    )
    step4 = torch.square(errors[0][:, 4]).mean()  # only the longer case reaches step 4
    torch.testing.assert_close(summary["MSE_w_time"][4], step4, rtol=1e-6, atol=1e-9)
    assert summary["MSE_w_time"].shape == (5,) and summary[
        "MSE_w_time_channel"
    ].shape == (5, 3)
    std = validation.node_stats["pos_std"].double()
    physical = torch.cat([(e * std).reshape(-1, 3) for e in errors])
    torch.testing.assert_close(
        summary["vector_RMSE_mm"],
        torch.sqrt(torch.square(physical).sum(1).mean()),
        rtol=1e-6,
        atol=1e-9,
    )
    # Per-case breakdown: each case pooled over its own steps and nodes, and
    # pooling the cases back together recovers the global number.
    cases = summarize_cases(records)
    assert [c["case"] for c in cases] == ["B_1", "B_2"]
    assert sorted(names) == ["B_1", "B_2"]
    for case in cases:
        e = errors[names.index(case["case"])]
        assert case["steps"] == e.shape[1]
        assert len(case["vector_RMSE_mm_per_time"]) == case["steps"]
        own = torch.sqrt(torch.square((e * std).reshape(-1, 3)).sum(1).mean())
        assert case["vector_RMSE_mm"] == pytest.approx(own.item(), rel=1e-6)
        first = torch.sqrt(torch.square(e[:, 0] * std).sum(1).mean())
        assert case["vector_RMSE_mm_per_time"][0] == pytest.approx(
            first.item(), rel=1e-6
        )
    pooled_sq = sum(c["vector_RMSE_mm"] ** 2 * c["steps"] * kept for c in cases)
    entries = sum(c["steps"] * kept for c in cases)
    assert (pooled_sq / entries) ** 0.5 == pytest.approx(
        summary["vector_RMSE_mm"].item(), rel=1e-6
    )
    # A trailing step nobody reaches is dropped; an interior hole is rejected.
    padded = torch.cat((totals, torch.zeros(2, 5, dtype=torch.float64)))
    assert summarize_validation(padded)["MSE_w_time"].shape == (5,)
    holed = totals.clone()
    holed[2] = 0
    with pytest.raises(ValueError, match="hole"):
        summarize_validation(holed)


def test_one_time_step_sampling_and_time_conditional_model(yaris_root, tmp_path):
    """Every (case, future frame) pair once, fixed time normalization, TC rollout by horizon."""
    from time_conditional import MeshGeoFLARETimeConditional
    from yaris_full_trainer import accumulate_validation, summarize_validation

    cfg = _compose(
        yaris_root,
        tmp_path / "work",
        ["datapipe.sample_type=one_time_step", "datapipe.initial_history_steps=1"],
    )
    reader = instantiate(cfg.reader)
    dataset = instantiate(
        cfg.datapipe, reader=reader, split="train", stats_mode="compute"
    )
    # Three training cases with 7 selected frames each: 6 future frames per case.
    assert dataset.case_future_frames == [6, 6, 6] and len(dataset) == 18
    assert (
        dataset._pair(0) == (0, 0)
        and dataset._pair(6) == (1, 0)
        and dataset._pair(17) == (2, 5)
    )
    kept = NODES - 3
    first, last = dataset[0], dataset[17]
    assert first.node_features["coords"].shape == (kept, 3)
    assert "previous_coords" not in first.node_features
    assert first.node_target.shape == (kept, 3)
    assert first.node_features["time"].item() == 0.0
    assert last.node_features["time"].item() == pytest.approx(
        5 / 6
    )  # k / (num_steps - 1)
    assert first.yaris_target_frames == [5] and last.yaris_target_frames == [30]
    # Validation keeps complete histories from the single observed frame, exactly
    # as the Yaris trainer builds it: B_1 has 6 future frames, B_2 has 5.
    validation = instantiate(
        cfg.datapipe,
        reader=reader,
        split="validation",
        stats_mode="load",
        sample_type="all_time_steps",
        num_samples=2,
    )
    assert validation.initial_history_steps == 1
    assert "previous_coords" not in validation[0].node_features
    assert validation[0].node_target.shape == (kept, 6, 3)
    assert validation[0].yaris_target_frames == [5, 10, 15, 20, 25, 30]
    assert validation[1].node_target.shape == (kept, 5, 3)
    torch.manual_seed(0)
    model = MeshGeoFLARETimeConditional(
        num_time_steps=7,
        functional_dim=3 + 1 + 8,
        out_dim=3,
        geometry_dim=3,
        global_dim=8,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        n_layers=1,
        use_te=False,
        attention_type="GALE_FA",
        time_fourier_bands=3,
        time_dim=8,
        input_dim_edges=4,
        mesh_hidden_dim=16,
        num_pre_processor_layers=1,
        num_post_processor_layers=1,
        mesh_context_fusion="pre_post",
        mesh_post_residual_gate_init=1.0,
        use_contact=False,
        local_processor="knn_attention",
        local_knn_k=4,
        local_neighborhood="euclidean",
        local_num_heads=4,
        local_edge_dim=8,
        local_attention_backend="torch",
        local_value_norm=False,
        local_difference_values=True,
        local_update="concat_mlp",
        local_split_key_value=True,
        activation_checkpointing=True,
        activation_checkpointing_components=["blocks", "context"],
    )
    stats = {"node": dict(dataset.node_stats)}
    model.train()
    output = model(sample=first, data_stats=stats, return_auxiliary=True)
    assert output.prediction.shape == (kept, 3) and output.coarse_prediction.shape == (
        kept,
        3,
    )
    loss = torch.nn.functional.mse_loss(output.prediction, first.node_target)
    loss.backward()
    # Evaluation follows each case's own horizon (5 and 4 steps) under one time scale.
    model.eval()
    with torch.no_grad():
        totals, seen = accumulate_validation(
            model,
            [[validation[0]], [validation[1]]],
            stats,
            rollout_steps=6,
            device=torch.device("cpu"),
        )
    assert sorted(seen) == ["B_1", "B_2"]
    torch.testing.assert_close(
        totals[:, 3], torch.tensor([2.0, 2, 2, 2, 2, 1]) * kept, check_dtype=False
    )
    summary = summarize_validation(totals)
    assert summary["MSE_w_time"].shape == (6,) and torch.isfinite(summary["MSE"])
    # The same physical frame gets the same time code in training and evaluation:
    # training k=2 of a 7-frame case is frame 15 = evaluation query k=2.
    sample_k2 = dataset[2]
    assert sample_k2.yaris_target_frames == [15]
    assert sample_k2.node_features["time"].item() == pytest.approx(2 / 6)


def test_yaris_time_conditional_recipe_contract(yaris_root, tmp_path):
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        cfg = compose(
            config_name="yaris_deformer",
            overrides=[
                f"yaris.data_root={yaris_root}",
                f"yaris.split_manifest={yaris_root / 'splits.csv'}",
                f"yaris.outlier_mask={yaris_root / 'outliers.npz'}",
                f"yaris.work_dir={tmp_path / 'work'}",
            ],
        )
    assert cfg.model._target_ == "time_conditional.MeshGeoFLARETimeConditional"
    assert (
        cfg.datapipe.sample_type == "one_time_step"
        and cfg.datapipe.initial_history_steps == 1
    )
    assert cfg.model.local_neighborhood == "euclidean" and cfg.model.local_knn_k == 32
    assert (
        cfg.model.activation_checkpointing is True
        and cfg.model.checkpointing_ratio == 1.0
    )
    assert (
        cfg.model.local_update == "concat_mlp" and cfg.model.time_embedding == "fourier"
    )
    assert cfg.training.trainer_class == "yaris_full_trainer.YarisFullTrainer"
    assert cfg.training.epochs == 60 and cfg.training.validation_freq == 5
    assert cfg.training.coarse_loss_weight == 0.1 and cfg.training.num_time_steps == 31
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["datapipe"]["local_knn_metric"] == "euclidean"
    assert (
        resolved["model"]["dt"] == 0.005 and resolved["model"]["num_time_steps"] == 31
    )


def _compose_yaris_tc(yaris_root, work_dir, extra=()):
    """The Yaris TC recipe at tiny dimensions on the synthetic store."""
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose(
            config_name="yaris_deformer",
            overrides=[
                f"yaris.data_root={yaris_root}",
                f"yaris.split_manifest={yaris_root / 'splits.csv'}",
                f"yaris.outlier_mask={yaris_root / 'outliers.npz'}",
                f"yaris.work_dir={work_dir}",
                "training.num_training_samples=3",
                "training.num_validation_samples=2",
                "training.num_time_steps=7",
                "training.amp=false",
                "training.num_dataloader_workers=0",
                "model.n_hidden=16",
                "model.n_head=4",
                "model.slice_num=4",
                "model.n_layers=1",
                "model.functional_dim=12",
                "model.global_dim=8",
                "model.time_dim=8",
                "model.time_fourier_bands=3",
                "model.mesh_hidden_dim=16",
                "model.local_num_heads=4",
                "model.local_edge_dim=8",
                "model.local_knn_k=4",
                "model.local_attention_backend=torch",
                *extra,
            ],
        )


def test_yaris_time_conditional_trainer_end_to_end(yaris_root, tmp_path):
    """The exact remote path: YarisFullTrainer builds both datasets from the recipe,
    instantiates the model from cfg, trains one step with the coarse loss and runs
    the exact validation over variable horizons (CPU, tiny dimensions)."""
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    cfg = _compose_yaris_tc(yaris_root, tmp_path / "work")
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(
        PythonLogger("yaris-tc-test"), DistributedManager()
    )
    trainer = YarisFullTrainer(cfg, logger0)
    assert trainer.coarse_loss_weight == 0.1 and trainer.rollout_steps == 6
    assert len(trainer.dataloader.dataset) == 18  # 3 cases x 6 future frames
    sample = next(iter(trainer.dataloader))[0]
    loss = trainer.train(sample, 0)
    assert torch.isfinite(loss) and "coarse" in trainer.last_loss_components
    stats = trainer.validate(0)
    assert torch.isfinite(stats["MSE"]) and stats["MSE_w_time"].shape == (6,)
    record = tmp_path / "work" / "validation" / "epoch_0001.json"
    assert record.exists()
    payload = json.loads(record.read_text())
    assert sorted(payload["cases"]) == ["B_1", "B_2"]
    assert payload["nodes_per_time"][-1] == NODES - 3  # only B_1 reaches step 6
    # Target frames 5, 10, ..., 30 of the 1 ms stores: the labels are their times.
    assert payload["target_time_ms"] == [5.0 * k for k in range(1, 7)]


def test_yaris_tc_trainer_relative_l2_end_to_end(yaris_root, tmp_path):
    """The relative L2 loss trains through the time-conditional path (main and
    coarse terms take the undeformed coordinates as the reference) and the
    exact validation still reports millimetres."""
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    cfg = _compose_yaris_tc(
        yaris_root,
        tmp_path / "work",
        extra=[
            "training.loss=relative_l2",
            "training.loss_space=physical",
            "training.loss_scale_mm=10.0",
        ],
    )
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(
        PythonLogger("yaris-tc-rel-l2-test"), DistributedManager()
    )
    trainer = YarisFullTrainer(cfg, logger0)
    assert trainer.position_loss.kind == "relative_l2"
    loader = iter(trainer.dataloader)
    for _ in range(2):
        loss = trainer.train(next(loader)[0], 0)
        assert torch.isfinite(loss) and loss.item() >= 0.0
    components = trainer.last_loss_components
    assert {"trajectory", "coarse"} <= set(components)
    # A norm ratio, not a squared error in scale units: both terms stay O(1).
    assert 0.0 <= components["trajectory"].item() < 100.0
    stats = trainer.validate(0)
    assert torch.isfinite(stats["MSE"])
    payload = json.loads(
        (tmp_path / "work" / "validation" / "epoch_0001.json").read_text()
    )
    assert payload["vector_RMSE_mm"] > 0 and len(payload["nodes_per_time"]) == 6


def test_solid_edges_and_isolated_fallback(yaris_root):
    from yaris_full import (
        YarisFullReader,
        connect_isolated_nodes,
        solid_element_edges,
    )

    plain = YarisFullReader(yaris_root / "splits.csv", yaris_root / "outliers.npz")
    store = plain.open(yaris_root / "A_1.zarr")
    chain = plain.structural_edges(store)
    solids = YarisFullReader(
        yaris_root / "splits.csv",
        yaris_root / "outliers.npz",
        include_solid_edges=True,
    )
    solids.open(yaris_root / "A_1.zarr")
    with_solids = solids.structural_edges(store)
    # Expected: chain edges plus the distinct hex edges not already in the chain,
    # all renumbered to kept nodes (node 0 is excluded, so kept id = store id - 1).
    hex_edges = solid_element_edges(store, NODES)
    assert hex_edges.shape == (12 + 4, 2)  # proper hex: 12; collapsed hex: 4 distinct
    expected = {tuple(sorted(e)) for e in (chain.t() - 0).tolist()}
    expected |= {tuple(sorted((a - 1, b - 1))) for a, b in hex_edges.tolist()}
    got = {tuple(sorted(e)) for e in with_solids.t().tolist()}
    assert got == expected and with_solids.shape[1] == len(expected)
    assert with_solids.shape[1] > chain.shape[1]
    # Isolated-node fallback links an edge-less node to its nearest neighbors only.
    positions = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0], [11, 0, 0]])
    edges = np.array([[0, 1], [1, 2]])
    linked = connect_isolated_nodes(edges, positions, k=2)
    assert {tuple(e) for e in linked.tolist()} == {
        (0, 1),
        (1, 2),
        (3, 4),
        (2, 3),
        (2, 4),
    }
    assert np.array_equal(
        connect_isolated_nodes(np.array([[0, 1], [1, 2], [3, 4]]), positions, 2),
        np.array([[0, 1], [1, 2], [3, 4]]),
    )
    with pytest.raises(ValueError):
        YarisFullReader(yaris_root / "splits.csv", connect_isolated_k=-1)


def test_yaris_tc_geodesic_with_solids_end_to_end(yaris_root, tmp_path):
    """Geodesic neighborhoods over shell + solid edges with the isolated-node
    fallback train and validate through the time-conditional path."""
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    cfg = _compose_yaris_tc(
        yaris_root,
        tmp_path / "work",
        extra=[
            "reader.include_solid_edges=true",
            "reader.connect_isolated_k=2",
            "model.local_neighborhood=geodesic",
        ],
    )
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(
        PythonLogger("yaris-geo-test"), DistributedManager()
    )
    trainer = YarisFullTrainer(cfg, logger0)
    dataset = trainer.dataloader.dataset
    assert dataset.graph.knn_metric == "geodesic" and dataset.graph.num_edges > 0
    assert (
        dataset.graph.knn_mask[:, 1:].any(dim=1).all()
    )  # every kept node has neighbors
    sample = next(iter(trainer.dataloader))[0]
    assert sample.global_features is None
    loss = trainer.train(sample, 0)
    assert torch.isfinite(loss)
    stats = trainer.validate(0)
    assert torch.isfinite(stats["MSE"])


def _compose_public(name):
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose(
            config_name=name,
            overrides=[
                "yaris.data_root=/fixtures/yaris",
                "yaris.split_manifest=/fixtures/splits.csv",
                "yaris.outlier_mask=/fixtures/outliers.npz",
                "yaris.work_dir=/fixtures/work",
            ],
        )


def test_contact_faces_from_shell_connectivity(yaris_root):
    from yaris_full import YarisFullReader

    from physicsnemo.nn.functional.neighbors.surface_contact import supported_quad_mask

    reader = YarisFullReader(yaris_root / "splits.csv", yaris_root / "outliers.npz")
    store = reader.open(yaris_root / "A_1.zarr")
    faces = reader.contact_faces(store)
    cells = np.asarray(store["shell/connectivity"][:])
    kept_quads = [c for c in cells if 0 not in c and c[2] != c[3] and len(set(c)) == 4]
    assert faces.dtype == torch.long and faces.shape[1] == 4
    assert faces.min() >= 0 and faces.max() < NODES - 3
    tri = faces[:, 2] == faces[:, 3]
    # Every kept quad is either supported or split into two triangles; the explicit
    # triangle survives; the degenerate facet and the one on node 0 are gone.
    reference = torch.from_numpy(reader.reference)
    quads = faces[~tri]
    assert supported_quad_mask(reference[quads]).all() if len(quads) else True
    n_quads_in = len(kept_quads)
    n_supported = (
        int(supported_quad_mask(reference[torch.as_tensor(kept_quads) - 1]).sum())
        if n_quads_in
        else 0
    )
    assert len(quads) == n_supported
    assert int(tri.sum()) == 1 + 2 * (n_quads_in - n_supported)
    assert (
        (faces == torch.tensor([19, 20, 21, 21])).all(dim=1).any()
    )  # ids shift by one


def test_yaris_contact_dataset_graph(yaris_root, tmp_path):
    from surface_topology import SurfaceContactData
    from yaris_full import YarisFullGraphDataset, YarisFullReader

    reader = YarisFullReader(yaris_root / "splits.csv", yaris_root / "outliers.npz")
    dataset = YarisFullGraphDataset(
        reader,
        yaris_root,
        num_samples=3,
        num_steps=7,
        split="train",
        stats_dir=tmp_path / "stats",
        stats_mode="compute",
        sample_type="one_time_step",
        initial_history_steps=1,
        local_knn_k=4,
        local_knn_cache_dir=tmp_path / "cache",
        contact_surface=True,
        rigid_wall_position=60.0,
    )
    graph = dataset.graph
    assert isinstance(graph, SurfaceContactData)
    assert graph.contact_faces.shape[1] == 4 and graph.shell_thickness.shape == (
        NODES - 3,
    )
    assert graph.contact_surface_exclusions.shape[0] == 2
    assert graph.contact_surface_exclusions.shape[1] > len(graph.contact_faces)
    assert graph.rigid_wall.tolist() == [60.0]
    assert any(
        p.name.startswith("contact_one_ring_") for p in (tmp_path / "stats").iterdir()
    )
    sample = dataset[0]
    assert sample.graph.contact_faces is graph.contact_faces


def test_per_step_lr_warmup(yaris_root, tmp_path):
    """training.lr_warmup_epochs: linear per-step rise over the warm-up epochs,
    then the cosine from the base rate to end_lr over the remaining epochs."""
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    cfg = _compose_yaris_tc(
        yaris_root, tmp_path / "work", extra=["training.lr_warmup_epochs=2"]
    )
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(PythonLogger("yaris-warmup"), DistributedManager())
    trainer = YarisFullTrainer(cfg, logger0)
    steps = len(trainer.dataloader)
    base = {id(g): g["initial_lr"] for g in trainer.optimizer.param_groups}
    assert steps == 18 and all(v == cfg.training.start_lr for v in base.values())
    sample = next(iter(trainer.dataloader))[0]
    # Linear per-step rise over the two warmup epochs.
    for k in range(1, 4):
        trainer.train(sample, 0)
        for g in trainer.optimizer.param_groups:
            assert g["lr"] == pytest.approx(base[id(g)] * k / (2 * steps))
    trainer.scheduler.step()  # end of epoch 0: the factor is 1 inside the warmup
    trainer.train(sample, 1)
    for g in trainer.optimizer.param_groups:
        assert g["lr"] == pytest.approx(base[id(g)] * (steps + 1) / (2 * steps))
    # After the warmup the cosine runs from the base rate to end_lr over the
    # remaining epochs, reaching end_lr after the last one.
    trainer.scheduler.step()
    trainer.train(sample, 2)
    for g in trainer.optimizer.param_groups:
        assert g["lr"] == pytest.approx(base[id(g)])
    for _ in range(cfg.training.epochs - 2):
        trainer.scheduler.step()
    for g in trainer.optimizer.param_groups:
        assert g["lr"] == pytest.approx(cfg.training.end_lr, rel=1e-6)
    with pytest.raises(ValueError, match="lr_warmup_epochs"):
        YarisFullTrainer(
            _compose_yaris_tc(
                yaris_root,
                tmp_path / "work",
                extra=[f"training.lr_warmup_epochs={cfg.training.epochs}"],
            ),
            logger0,
        )


def test_yaris_mesh_edges_on_the_graph(yaris_root, tmp_path):
    """datapipe.mesh_edges: structural connectivity in both directions with
    standardized reference features, next to an unchanged (empty) edge_index."""
    from yaris_full import YarisFullGraphDataset, YarisFullReader

    reader = YarisFullReader(yaris_root / "splits.csv", yaris_root / "outliers.npz")
    dataset = YarisFullGraphDataset(
        reader,
        yaris_root,
        num_samples=3,
        num_steps=7,
        split="train",
        stats_dir=tmp_path / "stats",
        stats_mode="compute",
        sample_type="one_time_step",
        initial_history_steps=1,
        local_knn_k=4,
        local_knn_metric="euclidean",
        local_knn_cache_dir=tmp_path / "cache",
        mesh_edges=True,
    )
    graph = dataset.graph
    assert graph.edge_index.shape == (2, 0)
    index = graph.mesh_edge_index
    assert index.shape[0] == 2 and index.shape[1] > 0 and index.shape[1] % 2 == 0
    pairs = {tuple(e) for e in index.t().tolist()}
    assert pairs == {(b, a) for a, b in pairs}
    assert (index[0] != index[1]).all()
    attr = graph.mesh_edge_attr
    assert attr.shape == (index.shape[1], 4)
    assert torch.allclose(attr.mean(dim=0), torch.zeros(4), atol=1e-4)
    assert torch.allclose(attr.std(dim=0, unbiased=False), torch.ones(4), atol=1e-3)
    assert graph.mesh_edge_mean.shape == (4,) and graph.mesh_edge_std.shape == (4,)
    assert (graph.mesh_edge_std > 0).all()
    sample = dataset[0]
    assert sample.graph.mesh_edge_index is graph.mesh_edge_index


def test_frame_pairs_serve_two_consecutive_frames(yaris_root, tmp_path):
    """datapipe.frame_pairs: the target holds frames k and k+1 under the earlier
    frame's label, the last frame pairs backwards, validation ignores the flag."""
    cfg = _compose(
        yaris_root,
        tmp_path / "work",
        [
            "datapipe.sample_type=one_time_step",
            "datapipe.initial_history_steps=1",
            "datapipe.frame_pairs=true",
        ],
    )
    reader = instantiate(cfg.reader)
    dataset = instantiate(
        cfg.datapipe, reader=reader, split="train", stats_mode="compute"
    )
    assert dataset.frame_pairs is True and len(dataset) == 18
    kept = NODES - 3
    first = dataset[0]
    assert first.node_target.shape == (kept, 2, 3)
    assert first.yaris_target_frames == [5, 10]
    assert first.node_features["time"].item() == 0.0
    single = instantiate(
        cfg.datapipe, reader=reader, split="train", stats_mode="load", frame_pairs=False
    )
    torch.testing.assert_close(first.node_target[:, 0], single[0].node_target)
    torch.testing.assert_close(first.node_target[:, 1], single[1].node_target)
    last = dataset[5]  # k = 5 is the case's last future frame: the pair is (4, 5)
    assert last.yaris_target_frames == [25, 30]
    assert last.node_features["time"].item() == pytest.approx(4 / 6)
    torch.testing.assert_close(last.node_target, dataset[4].node_target)
    validation = instantiate(
        cfg.datapipe,
        reader=reader,
        split="validation",
        stats_mode="load",
        sample_type="all_time_steps",
        num_samples=2,
    )
    assert validation.frame_pairs is False
    assert validation[0].node_target.shape == (kept, 6, 3)
