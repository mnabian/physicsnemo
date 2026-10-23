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
"""Full-resolution Yaris datapipe and nested-checkpointing recipe contracts."""

import json
import math
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
    bad.write_text("split,sample_name\ntrain,X_100_GR252303\nvalidation,B\ntest,C\n")
    with pytest.raises(ValueError, match="100_GR252303"):
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
    for rank in range(2):
        loader = DataLoader(
            validation,
            batch_size=1,
            sampler=ExactShardSampler(len(validation), 2, rank),
            collate_fn=simsample_collate,
        )
        part, names = accumulate_validation(
            model, loader, stats, 5, torch.device("cpu")
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
    with torch.no_grad():
        for i in range(2):
            sample = validation[i]
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
            config_name="crash_deformer_knn_yaris_time_conditional",
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
            config_name="crash_deformer_knn_yaris_time_conditional",
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


def test_yaris_tc_trainer_loss_options_end_to_end(yaris_root, tmp_path):
    """Huber in millimetres, EMA frame weighting and the edge term train together,
    and the exact validation still runs."""
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    cfg = _compose_yaris_tc(
        yaris_root,
        tmp_path / "work",
        extra=[
            "+training.loss=huber",
            "+training.loss_space=physical",
            "+training.loss_scale_mm=10.0",
            "+training.huber_delta_mm=3.0",
            "+training.frame_weighting=ema",
            "+training.edge_loss_weight=1.0",
        ],
    )
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(
        PythonLogger("yaris-tc-loss-test"), DistributedManager()
    )
    trainer = YarisFullTrainer(cfg, logger0)
    assert (
        trainer.position_loss.kind == "huber"
        and trainer.position_loss.space == "physical"
    )
    assert trainer.frame_weighter is not None and trainer.edge_loss_weight == 1.0
    loader = iter(trainer.dataloader)
    for _ in range(2):
        loss = trainer.train(next(loader)[0], 0)
        assert torch.isfinite(loss)
    assert {"trajectory", "coarse", "edge"} <= set(trainer.last_loss_components)
    assert trainer.frame_weighter.seen.any()
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


def test_design_variables_become_a_global_token(yaris_root, tmp_path):
    from yaris_full import YarisFullGraphDataset, YarisFullReader

    reader = YarisFullReader(yaris_root / "splits.csv", yaris_root / "outliers.npz")
    reader.open(yaris_root / "A_1.zarr")
    known = reader.design_variables(
        zarr.open_group(str(yaris_root / "A_1.zarr"), mode="r")
    )
    assert known.shape == (9,) and known[-1] == 1.0
    assert np.allclose(
        known[:8], 4.0 * np.array([-0.05 * i for i in range(1, 9)]), atol=1e-6
    )
    missing = reader.design_variables(
        zarr.open_group(str(yaris_root / f"{DESIGN_MISSING_CASE}.zarr"), mode="r")
    )
    assert np.array_equal(missing, np.zeros(9, dtype=np.float32))
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
        design_variables=True,
    )
    sample = dataset[0]
    assert sample.global_features["design"].shape == (9,)
    assert dataset.global_stats["global_mean"].shape == (9,)
    assert dataset.global_features_keys == ["design"]


def test_yaris_tc_geodesic_with_solids_and_design_token_end_to_end(
    yaris_root, tmp_path
):
    """Geodesic neighborhoods over shell + solid edges with the isolated-node
    fallback, and the design token through the model's global path."""
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
            "datapipe.design_variables=true",
            "model.global_dim=17",
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
    assert sample.global_features["design"].shape == (9,)
    loss = trainer.train(sample, 0)
    assert torch.isfinite(loss)
    stats = trainer.validate(0)
    assert torch.isfinite(stats["MSE"])


@pytest.mark.parametrize(
    "name, reader_keys, datapipe_keys, model_keys",
    [
        (
            "crash_deformer_knn_yaris_tc_geo_attn",
            {"include_solid_edges": True, "connect_isolated_k": 8},
            {"local_knn_metric": "geodesic"},
            {"local_neighborhood": "geodesic"},
        ),
        (
            "crash_deformer_knn_yaris_tc_geo_knnmpnn",
            {"include_solid_edges": True, "connect_isolated_k": 8},
            {"local_knn_metric": "geodesic", "local_knn_k": 8},
            {
                "local_neighborhood": "geodesic",
                "local_processor": "knn_mpnn",
                "local_knn_k": 8,
            },
        ),
        (
            "crash_deformer_knn_yaris_tc_euclid_attn_doe",
            {},
            {"design_variables": True},
            {"global_dim": 41},
        ),
        (
            "crash_deformer_knn_yaris_tc_euclid_knnmpnn",
            {},
            {"local_knn_k": 8},
            {"local_processor": "knn_mpnn", "local_knn_k": 8},
        ),
    ],
)
def test_yaris_tc_variant_recipes_change_only_their_keys(
    name, reader_keys, datapipe_keys, model_keys
):
    base = _compose_public("crash_deformer_knn_yaris_tc_loss_physical")
    variant = _compose_public(name)
    for group, changed in (
        ("reader", reader_keys),
        ("datapipe", datapipe_keys),
        ("model", model_keys),
    ):
        b = OmegaConf.to_container(base[group], resolve=True)
        v = OmegaConf.to_container(variant[group], resolve=True)
        diff = {k for k in set(b) | set(v) if b.get(k) != v.get(k)}
        assert diff == set(changed), (group, diff)
        for k, val in changed.items():
            assert v[k] == val, (group, k)
    assert OmegaConf.to_container(
        variant.training, resolve=True
    ) == OmegaConf.to_container(base.training, resolve=True)
    assert variant.experiment_name != base.experiment_name


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


@pytest.mark.parametrize(
    "name",
    ["physical", "geo_attn", "geo_knnmpnn", "euclid_attn_doe", "euclid_knnmpnn"],
)
def test_stride_one_recipes_change_only_the_time_sampling(name):
    parent = (
        "crash_deformer_knn_yaris_tc_loss_physical"
        if name == "physical"
        else f"crash_deformer_knn_yaris_tc_{name}"
    )
    base = _compose_public(parent)
    s1 = _compose_public(f"crash_deformer_knn_yaris_tc_s1_{name}")
    assert s1.yaris.stride == 1 and base.yaris.stride == 5
    assert s1.training.num_time_steps == 152 and s1.training.epochs == 15
    assert s1.training.validation_freq == 3 and s1.training.save_checkpoint_freq == 3
    assert s1.model.dt == 0.001 and s1.datapipe.dt == 0.001 and s1.reader.stride == 1
    for key in base.training:
        if key not in (
            "num_time_steps",
            "epochs",
            "validation_freq",
            "save_checkpoint_freq",
        ):
            assert base.training[key] == s1.training[key], key
    b_model = OmegaConf.to_container(base.model, resolve=True)
    s_model = OmegaConf.to_container(s1.model, resolve=True)
    assert {k for k in b_model if b_model[k] != s_model[k]} == {"dt", "num_time_steps"}
    assert s1.experiment_name != base.experiment_name


def test_yaris_tc_stride_one_end_to_end(yaris_root, tmp_path):
    """Every frame is a pair at stride 1; the exact validation covers all of them."""
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    cfg = _compose_yaris_tc(
        yaris_root,
        tmp_path / "work",
        extra=["yaris.stride=1", "training.num_time_steps=31", "model.dt=0.001"],
    )
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(
        PythonLogger("yaris-s1-test"), DistributedManager()
    )
    trainer = YarisFullTrainer(cfg, logger0)
    assert trainer.rollout_steps == 30
    # 3 training cases x 30 future frames (B_2 is shorter but is a validation case).
    assert len(trainer.dataloader.dataset) == 3 * 30
    sample = next(iter(trainer.dataloader))[0]
    assert torch.isfinite(trainer.train(sample, 0))
    stats = trainer.validate(0)
    assert stats["MSE_w_time"].shape == (30,)
    payload = json.loads(
        (tmp_path / "work" / "validation" / "epoch_0001.json").read_text()
    )
    assert payload["target_time_ms"][:3] == [2.0, 3.0, 4.0]
    assert payload["nodes_per_time"][-1] == NODES - 3  # only B_1 reaches frame 30


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


def test_rigid_wall_contact_encoder():
    from rigid_wall_contact import RigidWallContactEncoder

    encoder = RigidWallContactEncoder(axis=0, sign=1.0, activation_distance=5.0)
    positions = torch.tensor(
        [[90.0, 0, 0], [98.0, 1, 0], [100.0, 0, 1], [101.0, 0, 0], [50.0, 0, 0]]
    )
    thickness = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0])
    gap = encoder.signed_gap(positions, thickness, 100.0)
    torch.testing.assert_close(gap, torch.tensor([9.5, 1.5, -1.0, -1.5, 49.5]))
    graph = encoder(positions, thickness, 100.0)
    assert graph.edge_index.tolist() == [[1, 2, 3], [1, 2, 3]]
    assert graph.obstacle_mask.all() and graph.edge_features.shape == (3, 8)
    assert graph.edge_weights.min() > 0 and graph.edge_weights.max() <= 1
    # Displacement points from the node to the wall surface along +x for a node
    # still in front of it, and back toward the wall for one that passed it.
    assert graph.edge_features[0, 0] > 0 and graph.edge_features[2, 0] < 0
    assert encoder(positions[4:], thickness[4:], 100.0).edge_index.shape[1] == 0
    with pytest.raises(ValueError):
        RigidWallContactEncoder(axis=3)


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
    assert graph.penetration_reference_positions.shape == (NODES - 3, 3)
    assert graph.rigid_wall.tolist() == [60.0]
    assert any(
        p.name.startswith("contact_one_ring_") for p in (tmp_path / "stats").iterdir()
    )
    sample = dataset[0]
    assert sample.graph.contact_faces is graph.contact_faces


def test_yaris_tc_contact_both_end_to_end(yaris_root, tmp_path):
    """Surface + wall contact on the provisional shape, wall clamp on the output."""
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    cfg = _compose_yaris_tc(
        yaris_root,
        tmp_path / "work",
        extra=[
            "datapipe.contact_surface=true",
            "datapipe.rigid_wall_position=60.0",
            "model.use_contact=true",
            "model.contact_source=both",
            "model.clamp_to_wall=true",
            "+model.contact_search_implementation=torch",
        ],
    )
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(
        PythonLogger("yaris-contact-test"), DistributedManager()
    )
    trainer = YarisFullTrainer(cfg, logger0)
    model = trainer.model
    assert model.node_contact_builder is not None and model.wall_contact is not None
    sample = next(iter(trainer.dataloader))[0]
    loss = trainer.train(sample, 0)
    assert torch.isfinite(loss)
    stats = trainer.validate(0)
    assert torch.isfinite(stats["MSE"])
    # The clamp holds on a direct query: no predicted x beyond the wall.
    model.eval()
    with torch.no_grad():
        out = model._at_time(sample, trainer.data_stats)
    std = trainer.data_stats["node"]["pos_std"].reshape(-1)[0]
    mean = trainer.data_stats["node"]["pos_mean"].reshape(-1)[0]
    assert (out.prediction[:, 0] * std + mean <= 60.0 + 1e-3).all()


@pytest.mark.parametrize(
    "name, datapipe_keys, source, clamp",
    [
        ("surface", {"contact_surface": True}, "surface", False),
        ("wall", {"rigid_wall_position": 100.0}, "wall", True),
        ("both", {"contact_surface": True, "rigid_wall_position": 100.0}, "both", True),
    ],
)
def test_yaris_contact_recipes_change_only_contact_keys(
    name, datapipe_keys, source, clamp
):
    base = _compose_public("crash_deformer_knn_yaris_tc_loss_physical")
    variant = _compose_public(f"crash_deformer_knn_yaris_tc_contact_{name}")
    b = OmegaConf.to_container(base.datapipe, resolve=True)
    v = OmegaConf.to_container(variant.datapipe, resolve=True)
    assert {k for k in set(b) | set(v) if b.get(k) != v.get(k)} == set(datapipe_keys)
    for k, val in datapipe_keys.items():
        assert v[k] == val
    bm = OmegaConf.to_container(base.model, resolve=True)
    vm = OmegaConf.to_container(variant.model, resolve=True)
    changed = {k for k in set(bm) | set(vm) if bm.get(k) != vm.get(k)}
    assert all(
        k.startswith(("contact_", "use_contact", "node_contact", "wall_", "clamp_"))
        for k in changed
    ), changed
    assert (
        vm["use_contact"] is True
        and vm["contact_source"] == source
        and vm["clamp_to_wall"] is clamp
    )
    assert OmegaConf.to_container(
        variant.training, resolve=True
    ) == OmegaConf.to_container(base.training, resolve=True)
    assert variant.experiment_name != base.experiment_name


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "surface_v2",
            {
                "contact_side_from_reference": True,
                "contact_reference_clearance": 10.0,
            },
        ),
        (
            "surface_v2_band15",
            {
                "contact_side_from_reference": True,
                "contact_reference_clearance": 20.0,
                "contact_activation_distance": 15.0,
                "node_contact_radius": 20.0,
            },
        ),
        (
            "surface_v2_band10",
            {
                "contact_side_from_reference": True,
                "contact_reference_clearance": 15.0,
                "contact_activation_distance": 10.0,
            },
        ),
    ],
)
def test_yaris_surface_v2_recipes_only_add_builder_options(name, expected):
    surface = _compose_public("crash_deformer_knn_yaris_tc_contact_surface")
    variant = _compose_public(f"crash_deformer_knn_yaris_tc_contact_{name}")
    for group in ("datapipe", "reader"):
        assert OmegaConf.to_container(
            getattr(variant, group), resolve=True
        ) == OmegaConf.to_container(getattr(surface, group), resolve=True), group
    training = OmegaConf.to_container(variant.training, resolve=True)
    assert training.pop("validation_penetration") == {
        "enabled": True,
        "tolerance_mm": 0.5,
        "activation_distance_mm": 5.0,
        "reference_clearance_mm": 10.0,
        "feature_scale_mm": 10.0,
    }
    assert training == OmegaConf.to_container(surface.training, resolve=True)
    sm = OmegaConf.to_container(surface.model, resolve=True)
    vm = OmegaConf.to_container(variant.model, resolve=True)
    assert sm["contact_side_from_reference"] is False
    assert sm["contact_reference_clearance"] is None
    changed = {k for k in set(sm) | set(vm) if sm.get(k) != vm.get(k)}
    assert changed == set(expected), changed
    for key, value in expected.items():
        assert vm[key] == value
    assert variant.experiment_name != surface.experiment_name


def test_yaris_penalty_recipe_adds_only_the_penetration_loss():
    v2 = _compose_public("crash_deformer_knn_yaris_tc_contact_surface_v2")
    penalty = _compose_public("crash_deformer_knn_yaris_tc_contact_surface_v2_penalty")
    for group in ("datapipe", "model", "reader"):
        assert OmegaConf.to_container(
            getattr(penalty, group), resolve=True
        ) == OmegaConf.to_container(getattr(v2, group), resolve=True), group
    base = OmegaConf.to_container(v2.training, resolve=True)
    with_penalty = OmegaConf.to_container(penalty.training, resolve=True)
    assert "penetration_loss" not in base and "validation_penetration" in base
    assert with_penalty.pop("penetration_loss") == {
        "tolerance_mm": 0.5,
        "length_scale_mm": 10.0,
        "max_weight": 1.0,
        "warmup_epochs": 3,
        "ramp_epochs": 7,
    }
    assert with_penalty == base
    assert penalty.experiment_name != v2.experiment_name


def test_yaris_tc_penalty_end_to_end(yaris_root, tmp_path):
    """The trainer applies the crossing-aware penalty to the single TC frame."""
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    cfg = _compose_yaris_tc(
        yaris_root,
        tmp_path / "work",
        extra=[
            "datapipe.contact_surface=true",
            "model.use_contact=true",
            "model.contact_source=surface",
            "model.contact_side_from_reference=true",
            "model.contact_reference_clearance=10.0",
            "+model.contact_search_implementation=torch",
            "+training.penetration_loss.tolerance_mm=0.5",
            "+training.penetration_loss.length_scale_mm=10.0",
            "+training.penetration_loss.max_weight=1.0",
            "+training.penetration_loss.warmup_epochs=0",
            "+training.penetration_loss.ramp_epochs=0",
            "+training.validation_penetration.enabled=true",
            "+training.validation_penetration.implementation=torch",
            "+training.validation_penetration.tolerance_mm=0.0",
            "+training.validation_node_errors=true",
        ],
    )
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(
        PythonLogger("yaris-penalty-test"), DistributedManager()
    )
    trainer = YarisFullTrainer(cfg, logger0)
    assert trainer.penetration_cfg is not None
    sample = next(iter(trainer.dataloader))[0]
    loss = trainer.train(sample, 0)
    assert torch.isfinite(loss)
    components = trainer.last_loss_components
    assert components["penetration_weight"] == 1.0
    assert "penetration" in components and torch.isfinite(components["penetration"])
    assert "penetration_crossed_pairs" in components
    assert torch.isfinite(trainer.validate(0)["MSE"])
    # The validation probe scored prediction and target on every step.
    record = json.loads((trainer.validation_dir / "epoch_0001.json").read_text())
    probe = record["penetration"]
    assert probe["complete"] is True and probe["side_from_reference"] is True
    horizon = len(record["vector_RMSE_mm_per_time"])
    for name in ("prediction", "target"):
        assert len(probe[name]["crossed_pairs_per_time"]) == horizon
        assert 0.0 <= probe[name]["penetrating_node_fraction"] <= 1.0
        assert math.isfinite(probe[name]["mean_depth_mm"])
    # Per-node RMSE over every validated frame, consistent with the pooled RMSE.
    import numpy as np

    node_rmse = np.load(trainer.validation_dir / "node_rmse_epoch_0001.npy")
    assert node_rmse.shape == (trainer.validation_dataset.num_nodes,)
    pooled = math.sqrt(float(np.mean(node_rmse.astype(np.float64) ** 2)))
    assert abs(pooled - record["vector_RMSE_mm"]) < 1e-3 * max(1.0, pooled)
    meta = np.load(trainer.validation_dir / "node_meta.npz")
    assert meta["reference"].shape == (len(node_rmse), 3)
    assert meta["part_id"].shape == (len(node_rmse),)
    # Regional translational breakdown: the mean-error part never exceeds the MSE.
    regions = record["regions"]
    per_case = regions.pop("per_case_mean_error_mm")
    assert (
        len(per_case) == len(record["cases"]) and per_case[0]["case"] in record["cases"]
    )
    assert len(per_case[0]["mean_error_mm"]) == 3
    assert len(regions) == 3 and sum(r["nodes"] for r in regions.values()) == len(
        node_rmse
    )
    for region in regions.values():
        if region["nodes"]:
            assert 0.0 <= region["translational_share"] <= 1.0 + 1e-9
            assert len(region["rmse_mm_per_time"]) == horizon


def test_yaris_iter2_recipe_changes_only_the_pass_count():
    v2 = _compose_public("crash_deformer_knn_yaris_tc_contact_surface_v2")
    iter2 = _compose_public("crash_deformer_knn_yaris_tc_contact_surface_v2_iter2")
    for group in ("datapipe", "training", "reader"):
        assert OmegaConf.to_container(
            getattr(iter2, group), resolve=True
        ) == OmegaConf.to_container(getattr(v2, group), resolve=True), group
    vm = OmegaConf.to_container(v2.model, resolve=True)
    im = OmegaConf.to_container(iter2.model, resolve=True)
    assert vm["contact_iterations"] == 1 and im["contact_iterations"] == 2
    assert {k for k in set(vm) | set(im) if vm.get(k) != im.get(k)} == {
        "contact_iterations"
    }
    assert iter2.experiment_name != v2.experiment_name


@pytest.mark.parametrize(
    "name, base, extra_datapipe",
    [
        (
            "crash_deformer_knn_yaris_tc_contact_surface_v2_ft",
            "crash_deformer_knn_yaris_tc_contact_surface_v2",
            {},
        ),
        (
            "crash_deformer_knn_yaris_tc_loss_physical_ft",
            "crash_deformer_knn_yaris_tc_loss_physical",
            {"contact_surface": True},
        ),
    ],
)
def test_fine_tune_recipes_change_only_the_schedule(name, base, extra_datapipe):
    base_cfg = _compose_public(base)
    ft = _compose_public(name)
    assert OmegaConf.to_container(ft.model, resolve=True) == OmegaConf.to_container(
        base_cfg.model, resolve=True
    )
    bd = OmegaConf.to_container(base_cfg.datapipe, resolve=True)
    fd = OmegaConf.to_container(ft.datapipe, resolve=True)
    assert {k for k in set(bd) | set(fd) if bd.get(k) != fd.get(k)} == set(
        extra_datapipe
    )
    bt = OmegaConf.to_container(base_cfg.training, resolve=True)
    ftt = OmegaConf.to_container(ft.training, resolve=True)
    assert ftt.pop("warm_start_model") is None
    assert (ftt.pop("epochs"), ftt.pop("start_lr"), ftt.pop("end_lr")) == (
        10,
        2.0e-5,
        1.0e-6,
    )
    assert (ftt.pop("validation_freq"), ftt.pop("save_checkpoint_freq")) == (2, 2)
    for key in (
        "epochs",
        "start_lr",
        "end_lr",
        "validation_freq",
        "save_checkpoint_freq",
    ):
        bt.pop(key)
    probe = ftt.pop("validation_penetration")
    assert probe["enabled"] is True and probe["reference_clearance_mm"] == 10.0
    bt.pop("validation_penetration", None)
    assert ftt == bt
    assert ft.experiment_name != base_cfg.experiment_name


def test_global_motion_recipes_change_only_the_head_and_the_schedule():
    control = _compose_public("crash_deformer_knn_yaris_tc_loss_physical")
    gm = _compose_public("crash_deformer_knn_yaris_tc_global_motion")
    for group in ("datapipe", "training", "reader"):
        assert OmegaConf.to_container(
            getattr(gm, group), resolve=True
        ) == OmegaConf.to_container(getattr(control, group), resolve=True), group
    cm = OmegaConf.to_container(control.model, resolve=True)
    gmm = OmegaConf.to_container(gm.model, resolve=True)
    assert cm["global_motion"] is False and gmm["global_motion"] is True
    assert {k for k in set(cm) | set(gmm) if cm.get(k) != gmm.get(k)} == {
        "global_motion"
    }
    ft = _compose_public("crash_deformer_knn_yaris_tc_global_motion_ft")
    assert OmegaConf.to_container(ft.model, resolve=True) == gmm
    ftt = OmegaConf.to_container(ft.training, resolve=True)
    assert ftt.pop("warm_start_model") is None
    assert (ftt.pop("epochs"), ftt.pop("start_lr"), ftt.pop("end_lr")) == (
        10,
        2.0e-5,
        1.0e-6,
    )
    assert (ftt.pop("validation_freq"), ftt.pop("save_checkpoint_freq")) == (2, 2)
    base = OmegaConf.to_container(control.training, resolve=True)
    for key in (
        "epochs",
        "start_lr",
        "end_lr",
        "validation_freq",
        "save_checkpoint_freq",
    ):
        base.pop(key)
    assert ftt == base


def test_thickness_deviation_feature_and_recipe(yaris_root, tmp_path):
    """Fleet-mean thickness from the training cases; second feature column."""
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    control = _compose_public("crash_deformer_knn_yaris_tc_loss_physical")
    recipe = _compose_public("crash_deformer_knn_yaris_tc_thickness_dev")
    assert recipe.datapipe.thickness_deviation is True
    assert control.datapipe.thickness_deviation is False
    assert recipe.model.functional_dim == control.model.functional_dim + 1
    assert OmegaConf.to_container(
        recipe.training, resolve=True
    ) == OmegaConf.to_container(control.training, resolve=True)
    tiny = _compose_yaris_tc(yaris_root, tmp_path / "work")
    cfg = _compose_yaris_tc(
        yaris_root,
        tmp_path / "work",
        extra=[
            "datapipe.thickness_deviation=true",
            f"model.functional_dim={int(tiny.model.functional_dim) + 1}",
        ],
    )
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(
        PythonLogger("yaris-thickness-dev"), DistributedManager()
    )
    trainer = YarisFullTrainer(cfg, logger0)
    dataset = trainer.dataloader.dataset
    assert dataset.thickness_fleet_mean.shape == (dataset.num_nodes,)
    cache = list((tmp_path / "work").rglob("thickness_fleet_*.npz"))
    assert len(cache) == 1
    sample = next(iter(trainer.dataloader))[0]
    features = sample.node_features["features"]
    assert features.shape == (dataset.num_nodes, 2) and torch.isfinite(features).all()
    # The deviation column is zero where the node's thickness equals the fleet mean.
    import numpy as np

    store = dataset.reader.open(dataset.paths[0])
    thickness = torch.from_numpy(dataset.reader.thickness(store))
    same = thickness == dataset.thickness_fleet_mean
    assert torch.all(features[same, 1] == 0)
    loss = trainer.train(sample, 0)
    assert torch.isfinite(loss)
    assert torch.isfinite(trainer.validate(0)["MSE"])
    assert np.load(cache[0])["dev_std"] >= 0


def test_ema_weights_track_training_and_are_used_for_validation(yaris_root, tmp_path):
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    recipe = _compose_public("crash_deformer_knn_yaris_tc_ema")
    assert recipe.training.ema_decay == 0.999
    cfg = _compose_yaris_tc(
        yaris_root, tmp_path / "work", extra=["+training.ema_decay=0.5"]
    )
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(PythonLogger("yaris-ema"), DistributedManager())
    trainer = YarisFullTrainer(cfg, logger0)
    assert trainer.ema_model is not None
    model = trainer._unwrapped_model()
    before = [p.detach().clone() for p in model.parameters()]
    for p, q in zip(model.parameters(), trainer.ema_model.parameters()):
        assert torch.equal(p, q)
    loader = iter(trainer.dataloader)
    for step in range(2):
        trainer.train(next(loader)[0], 0)
    assert trainer.ema_updates == 2
    after = [p.detach().clone() for p in model.parameters()]
    moved = [i for i, (a, b) in enumerate(zip(before, after)) if not torch.equal(a, b)]
    assert moved
    # EMA lies between the start and the trained weights, strictly inside.
    i = moved[0]
    ema = list(trainer.ema_model.parameters())[i]
    assert not torch.equal(ema, after[i]) and not torch.equal(ema, before[i])
    # Validation swaps the EMA weights in and restores the trained ones after.
    with trainer.ema_weights():
        assert torch.equal(list(model.parameters())[i], ema)
    assert torch.equal(list(model.parameters())[i], after[i])
    assert torch.isfinite(trainer.validate(0)["MSE"])
    trainer.save_ema(1)
    path = trainer._ema_path(cfg.training.ckpt_path, 1)
    assert path.exists()
    # Resuming at that epoch restores the EMA copy from the file.
    from utils import load_warm_start

    fresh = _compose_yaris_tc(
        yaris_root, tmp_path / "work", extra=["+training.ema_decay=0.5"]
    )
    probe = YarisFullTrainer(fresh, logger0)
    load_warm_start(probe.ema_model, path, allowed_missing_prefixes=())
    assert torch.equal(list(probe.ema_model.parameters())[i], ema)
    with pytest.raises(ValueError, match="ema_decay"):
        YarisFullTrainer(
            _compose_yaris_tc(
                yaris_root, tmp_path / "work", extra=["+training.ema_decay=1.0"]
            ),
            logger0,
        )


def test_design_mixup_blends_two_training_cases(yaris_root, tmp_path):
    recipe = _compose_public("crash_deformer_knn_yaris_tc_mixup")
    assert recipe.datapipe.mixup_alpha == 1.0 and recipe.datapipe.mixup_prob == 0.5
    cfg = _compose_yaris_tc(
        yaris_root,
        tmp_path / "work",
        extra=["datapipe.mixup_alpha=1.0", "datapipe.mixup_prob=1.0"],
    )
    from hydra.utils import instantiate

    reader = instantiate(cfg.reader)
    dataset = instantiate(
        cfg.datapipe,
        name="crash_train",
        reader=reader,
        split="train",
        stats_mode="compute",
        sample_type="one_time_step",
        data_dir=cfg.training.raw_data_dir,
        num_samples=cfg.training.num_training_samples,
    )
    dataset.set_epoch(0)
    sample = dataset[0]
    assert sample.yaris_mixup is not None
    partner, lam = sample.yaris_mixup
    assert 0.0 <= lam <= 1.0 and "+" in sample.yaris_case_name
    case, k = dataset._pair(0)
    assert partner != case
    # Rebuild the blend from the two stores.
    plain = instantiate(
        cfg.datapipe,
        name="crash_train_plain",
        reader=reader,
        split="train",
        stats_mode="load",
        sample_type="one_time_step",
        data_dir=cfg.training.raw_data_dir,
        num_samples=cfg.training.num_training_samples,
        mixup_alpha=0.0,
    )
    own = plain[0]
    other_index = plain.case_offsets[partner] + k
    other = plain[other_index]
    torch.testing.assert_close(
        sample.node_target, lam * own.node_target + (1 - lam) * other.node_target
    )
    torch.testing.assert_close(
        sample.node_features["features"],
        lam * own.node_features["features"]
        + (1 - lam) * other.node_features["features"],
    )
    torch.testing.assert_close(
        sample.node_features["coords"], own.node_features["coords"]
    )
    # Same draw again (stateless in seed, epoch, index); a new epoch may differ.
    assert dataset[0].yaris_mixup == sample.yaris_mixup
    from hydra.errors import InstantiationException

    with pytest.raises((ValueError, InstantiationException), match="mixup"):
        instantiate(
            cfg.datapipe,
            name="bad",
            reader=reader,
            split="train",
            stats_mode="load",
            sample_type="one_time_step",
            data_dir=cfg.training.raw_data_dir,
            num_samples=cfg.training.num_training_samples,
            mixup_alpha=-1.0,
        )


def test_mean_displacement_baseline_is_prepared_served_and_added(yaris_root, tmp_path):
    from hydra.utils import instantiate
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    assert _compose_public(
        "crash_deformer_knn_yaris_tc_baseline"
    ).datapipe.mean_displacement_baseline
    cfg = _compose_yaris_tc(
        yaris_root,
        tmp_path / "work",
        extra=["datapipe.mean_displacement_baseline=true"],
    )
    reader = instantiate(cfg.reader)
    # Loading before preparation is an explicit error.
    with pytest.raises(Exception, match="mean_displacement"):
        instantiate(
            cfg.datapipe,
            name="early",
            reader=reader,
            split="train",
            stats_mode="load",
            sample_type="one_time_step",
            data_dir=cfg.training.raw_data_dir,
            num_samples=cfg.training.num_training_samples,
        )
    train = instantiate(
        cfg.datapipe,
        name="train",
        reader=reader,
        split="train",
        stats_mode="compute",
        sample_type="one_time_step",
        data_dir=cfg.training.raw_data_dir,
        num_samples=cfg.training.num_training_samples,
    )
    steps = train.num_steps - 1
    assert train.mean_displacement.shape == (steps, train.num_nodes, 3)
    assert len(list((tmp_path / "work").rglob("mean_displacement_*.npy"))) == 1
    # k = 0 baseline equals the mean over the training cases of (frame 1 - frame 0) / std.
    expected = []
    for path in reader.paths(
        cfg.training.raw_data_dir, "train", len(reader.splits["train"])
    ):
        store = reader.open(path)
        frames = reader.frames(store)
        coords = reader.coordinates(store, frames[:2]).astype(np.float64)
        expected.append(coords[1] - coords[0])
    expected = np.mean(expected, axis=0) / (train.node_stats["pos_std"].numpy() + 1e-8)
    torch.testing.assert_close(
        train.mean_displacement[0], torch.as_tensor(expected, dtype=torch.float32)
    )
    sample = train[0]
    torch.testing.assert_close(
        sample.node_features["baseline"], train.mean_displacement[0]
    )
    # The trainer path: zero-initialised heads predict coords + baseline exactly,
    # one validation frame per baseline frame.
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(
        PythonLogger("yaris-baseline"), DistributedManager()
    )
    trainer = YarisFullTrainer(cfg, logger0)
    held = next(iter(trainer.val_dataloader))[0]
    assert held.node_features["baseline_all"].shape == (
        train.num_nodes,
        held.node_target.shape[1],
        3,
    )
    model = trainer.model
    model.train()
    batch = next(iter(trainer.dataloader))[0]
    with torch.no_grad():
        out = model(sample=batch, data_stats=trainer.data_stats, return_auxiliary=True)
    torch.testing.assert_close(
        out.coarse_prediction,
        batch.node_features["coords"] + batch.node_features["baseline"],
    )
    model.eval()
    with torch.no_grad():
        rollout = model(sample=held, data_stats=trainer.data_stats)
    torch.testing.assert_close(
        rollout,
        held.node_features["coords"][:, None, :] + held.node_features["baseline_all"],
    )
    assert "baseline" not in held.node_features
    model.train()
    assert torch.isfinite(trainer.train(batch, 0))
    assert torch.isfinite(trainer.validate(0)["MSE"])


def test_concrete_dropout_recipe_trains_with_the_regulariser(yaris_root, tmp_path):
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    recipe = _compose_public("crash_deformer_knn_yaris_tc_concrete_dropout")
    assert recipe.model.concrete_dropout is True and recipe.model.dropout == 0.1
    assert recipe.training.concrete_dropout_lambda == 1.0e-4
    cfg = _compose_yaris_tc(
        yaris_root,
        tmp_path / "work",
        extra=[
            "model.concrete_dropout=true",
            "model.dropout=0.1",
            "+training.concrete_dropout_lambda=1e-3",
        ],
    )
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(
        PythonLogger("yaris-concrete"), DistributedManager()
    )
    trainer = YarisFullTrainer(cfg, logger0)
    rates = trainer.concrete_dropout_rates()
    assert rates and all(abs(r - 0.1) < 1e-5 for r in rates.values())
    sample = next(iter(trainer.dataloader))[0]
    loss = trainer.train(sample, 0)
    assert torch.isfinite(loss)
    assert "concrete_dropout" in trainer.last_loss_components
    assert torch.isfinite(trainer.validate(0)["MSE"])


def test_delta_reference_pairs_designs_and_averages_validation_references(
    yaris_root, tmp_path
):
    from hydra.utils import instantiate
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    recipe = _compose_public("crash_deformer_knn_yaris_tc_delta")
    assert recipe.datapipe.delta_reference is True and recipe.model.functional_dim == 37
    tiny = _compose_yaris_tc(yaris_root, tmp_path / "work")
    cfg = _compose_yaris_tc(
        yaris_root,
        tmp_path / "work",
        extra=[
            "datapipe.delta_reference=true",
            "datapipe.delta_val_references=2",
            f"model.functional_dim={int(tiny.model.functional_dim) + 1}",
        ],
    )
    reader = instantiate(cfg.reader)
    train = instantiate(
        cfg.datapipe,
        name="train",
        reader=reader,
        split="train",
        stats_mode="compute",
        sample_type="one_time_step",
        data_dir=cfg.training.raw_data_dir,
        num_samples=cfg.training.num_training_samples,
    )
    assert len(list((tmp_path / "work").rglob("delta_neighbours_*.json"))) == 1
    sample = train[0]
    case, k = train._pair(0)
    own = train.paths[case].stem
    reference = sample.yaris_delta_reference
    assert reference in train.delta_neighbours["train_cases"] and reference != own
    assert sample.node_features["features"].shape == (train.num_nodes, 2)
    # The baseline is the reference design's displacement at the same frame.
    store = reader.open(train.delta_neighbours["train_paths"][reference])
    frames = reader.frames(store)
    pos = torch.from_numpy(reader.coordinates(store, [frames[0], frames[1 + k]]))
    expected = (pos[1] - pos[0]) / (train.node_stats["pos_std"] + 1e-8)
    torch.testing.assert_close(sample.node_features["baseline"], expected)
    assert train[0].yaris_delta_reference == reference  # stateless draw
    # Validation: nearest training designs, extra references carried on the sample.
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(PythonLogger("yaris-delta"), DistributedManager())
    trainer = YarisFullTrainer(cfg, logger0)
    held = next(iter(trainer.val_dataloader))[0]
    assert len(held.yaris_delta_references) == 2 and len(held.delta_references) == 1
    assert held.node_features["features"].shape == (train.num_nodes, 2)
    steps = held.node_target.shape[1]
    assert held.node_features["baseline_all"].shape == (train.num_nodes, steps, 3)
    assert (
        held.yaris_delta_references
        == train.delta_neighbours["nearest"][held.yaris_case_name][:2]
    )
    # Zero-initialised heads: the prediction is the mean reference trajectory,
    # i.e. a nearest-design baseline; the validation path averages the references.
    from yaris_full_trainer import accumulate_validation

    totals, seen = accumulate_validation(
        trainer.model.eval(),
        trainer.val_dataloader,
        trainer.data_stats,
        trainer.rollout_steps,
        trainer.dist.device,
    )
    coords = held.node_features["coords"][:, None, :]
    mean_reference = (
        coords + (held.node_features["baseline_all"] + held.delta_references[0][1]) / 2
    )
    expected_sq = torch.square((mean_reference - held.node_target).double()).sum()
    assert seen == [held.yaris_case_name] or len(seen) >= 1
    if seen == [held.yaris_case_name]:
        torch.testing.assert_close(totals[:steps, :3].sum(), expected_sq)
    trainer.model.train()
    assert torch.isfinite(trainer.train(next(iter(trainer.dataloader))[0], 0))
    assert torch.isfinite(trainer.validate(0)["MSE"])
    with pytest.raises(Exception, match="delta_reference"):
        instantiate(
            cfg.datapipe,
            name="bad",
            reader=reader,
            split="train",
            stats_mode="load",
            sample_type="one_time_step",
            data_dir=cfg.training.raw_data_dir,
            num_samples=cfg.training.num_training_samples,
            mixup_alpha=1.0,
        )


def test_local_dropout_recipe_and_node_subset_loss(yaris_root, tmp_path):
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    recipe = _compose_public("crash_deformer_knn_yaris_tc_local_dropout")
    assert (
        recipe.model.local_concrete_dropout is True
        and recipe.model.local_dropout == 0.1
    )
    assert recipe.training.loss_node_fraction == 0.5
    cfg = _compose_yaris_tc(
        yaris_root, tmp_path / "work", extra=["+training.loss_node_fraction=0.5"]
    )
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(PythonLogger("yaris-subset"), DistributedManager())
    trainer = YarisFullTrainer(cfg, logger0)
    sample = next(iter(trainer.dataloader))[0]
    prediction = sample.node_target + 0.1 * torch.randn_like(sample.node_target)
    trainer.model.train()
    torch.manual_seed(0)
    first = trainer._supervised_position_loss(prediction, sample.node_target, sample)
    torch.manual_seed(1)
    second = trainer._supervised_position_loss(prediction, sample.node_target, sample)
    assert first != second  # different random node subsets
    trainer.model.eval()
    full = trainer._supervised_position_loss(prediction, sample.node_target, sample)
    torch.testing.assert_close(
        full, trainer.position_loss(prediction, sample.node_target)
    )
    trainer.model.train()
    assert torch.isfinite(trainer.train(sample, 0))
    with pytest.raises(ValueError, match="loss_node_fraction"):
        YarisFullTrainer(
            _compose_yaris_tc(
                yaris_root, tmp_path / "work", extra=["+training.loss_node_fraction=0"]
            ),
            logger0,
        )


def test_part_tokens_pool_features_per_part_and_train(yaris_root, tmp_path):
    from hydra.utils import instantiate
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    recipe = _compose_public("crash_deformer_knn_yaris_tc_part_tokens")
    assert recipe.datapipe.part_tokens and recipe.model.part_tokens
    assert recipe.model.global_dim == 64 and recipe.model.global_motion is True
    tiny = _compose_yaris_tc(yaris_root, tmp_path / "work")
    cfg = _compose_yaris_tc(
        yaris_root,
        tmp_path / "work",
        extra=[
            "datapipe.part_tokens=true",
            "model.part_tokens=true",
            "model.part_token_dim=5",
            "model.design_dim=8",
            f"model.global_dim={int(tiny.model.global_dim) + 8}",
            "model.global_motion=true",
        ],
    )
    reader = instantiate(cfg.reader)
    train = instantiate(
        cfg.datapipe,
        name="train",
        reader=reader,
        split="train",
        stats_mode="compute",
        sample_type="one_time_step",
        data_dir=cfg.training.raw_data_dir,
        num_samples=cfg.training.num_training_samples,
    )
    ids = reader.part_ids(reader.open(train.paths[0]))
    parts = np.unique(ids)
    sample = train[0]
    tokens = sample.global_features["parts"]
    assert tokens.shape == (len(parts), 5)
    features = sample.node_features["features"]
    for i, part in enumerate(parts):
        mask = torch.from_numpy(ids == part)
        torch.testing.assert_close(tokens[i, 4], features[mask, 0].mean())
        ref = (torch.from_numpy(reader.reference) - train.node_stats["pos_mean"]) / (
            train.node_stats["pos_std"] + 1e-8
        )
        torch.testing.assert_close(tokens[i, :3], ref[mask].mean(0))
    assert torch.all(tokens[:, 3] <= 1.0) and torch.all(tokens[:, 3] > 0)
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(PythonLogger("yaris-parts"), DistributedManager())
    trainer = YarisFullTrainer(cfg, logger0)
    batch = next(iter(trainer.dataloader))[0]
    assert "parts" in batch.global_features
    assert torch.isfinite(trainer.train(batch, 0))
    assert torch.isfinite(trainer.validate(0)["MSE"])
