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
"""Recipe contracts for the experimental kNN neighborhood-attention DeFormer."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))
CONF = CRASH_DIR / "conf"

from datapipe import SimSample  # noqa: E402
from local_neighborhoods import (  # noqa: E402
    LocalNeighborhoodCache,
    LocalNeighborhoodData,
    knn_edge_features,
    with_self_slot,
)
from rollout import MeshGeoFLAREAutoregressive  # noqa: E402

LOCAL_KEYS = {
    "local_processor",
    "local_neighborhood",
    "local_knn_k",
    "local_num_heads",
    "local_mlp_ratio",
    "local_edge_dim",
    "local_activation",
    "local_attention_fused",
    "local_attention_backend",
}
DATAPIPE_KEYS = {"local_knn_k", "local_knn_metric", "local_knn_cache_dir"}


def _compose(name, overrides=()):
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose(
            config_name=name,
            overrides=[
                "training.raw_data_dir=/fixtures/train",
                "training.raw_data_dir_validation=/fixtures/validation",
                "inference.raw_data_dir_test=/fixtures/test",
                *overrides,
            ],
        )


@pytest.mark.parametrize("metric", ["geodesic", "euclidean"])
def test_knn_recipe_changes_only_the_local_operator(metric):
    """Both metric variants inherit the full no-contact DeFormer recipe."""
    baseline = _compose("crash_deformer_autoregressive")
    knn = _compose(
        "crash_deformer_knn_autoregressive", [f"model.local_neighborhood={metric}"]
    )
    assert baseline.training == knn.training
    assert baseline.reader == knn.reader
    assert baseline.inference == knn.inference
    for key in baseline.model:
        assert baseline.model[key] == knn.model[key], key
    assert set(knn.model) - set(baseline.model) == LOCAL_KEYS
    assert knn.model.local_processor == "knn_attention"
    assert knn.model.local_neighborhood == metric
    assert knn.model.local_knn_k == 32
    # The fused kernel is opt-in; the gather reference stays the documented default.
    assert knn.model.local_attention_fused is False
    assert knn.model.local_attention_backend == "auto"
    for key in baseline.datapipe:
        assert baseline.datapipe[key] == knn.datapipe[key], key
    assert set(knn.datapipe) - set(baseline.datapipe) == DATAPIPE_KEYS
    resolved = OmegaConf.to_container(knn, resolve=True)
    assert resolved["datapipe"]["local_knn_k"] == 32
    assert resolved["datapipe"]["local_knn_metric"] == metric
    assert resolved["datapipe"]["local_knn_cache_dir"].endswith("local_knn_cache")
    assert not knn.model.use_contact and not knn.model.enable_contact


def _write_stacked_quads(data_dir, runs=2):
    """Two unconnected unit quads, 0.05 apart in z: one case repeated twice."""
    import pyvista as pv

    points = np.array(
        [[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        + [[0.0, 0, 0.05], [1, 0, 0.05], [1, 1, 0.05], [0, 1, 0.05]]
    )
    mesh = pv.PolyData(points, np.array([4, 0, 1, 2, 3, 4, 4, 5, 6, 7]))
    mesh.point_data["thickness"] = np.ones(len(points))
    for t in range(3):
        mesh.point_data[f"displacement_t0.{t * 5:03d}"] = np.ones((len(points), 3)) * t
    for i in range(runs):
        mesh.save(data_dir / f"Run{i}.vtp")


@pytest.mark.parametrize("metric", ["geodesic", "euclidean"])
def test_datapipe_attaches_static_neighborhoods(tmp_path, metric):
    pytest.importorskip("pyvista")
    from datapipe import KNN_EDGE_STATS_FILE, CrashGraphDataset
    from vtp_reader import Reader

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_stacked_quads(data_dir)
    k = 4
    options = dict(
        data_dir=str(data_dir),
        num_samples=2,
        num_steps=3,
        initial_history_steps=2,
        static_features=["thickness"],
        stats_dir=str(tmp_path / "stats"),
        local_knn_k=k,
        local_knn_metric=metric,
        local_knn_cache_dir=str(tmp_path / "stats" / "local_knn_cache"),
    )
    reader = Reader(cache_dir=str(tmp_path / "cache"))
    dataset = CrashGraphDataset(reader=reader, **options)
    graph = dataset.graphs[0]
    assert isinstance(graph, LocalNeighborhoodData)
    assert graph.knn_index.shape == (8, k + 1) and graph.knn_index.dtype == torch.long
    assert graph.knn_mask.shape == (8, k + 1) and graph.knn_mask.dtype == torch.bool
    assert graph.knn_edge_attr.shape == (8, k + 1, 4)
    assert graph.knn_metric == metric
    torch.testing.assert_close(graph.knn_index[:, 0], torch.arange(8))
    assert graph.knn_mask[:, 0].all()
    neighbors = graph.knn_index[0, 1:][graph.knn_mask[0, 1:]]
    if metric == "geodesic":
        # Node 0 reaches only its own quad (3 nodes); the fourth slot is padding.
        assert sorted(neighbors.tolist()) == [1, 2, 3]
        assert graph.knn_mask[0].tolist() == [True, True, True, True, False]
    else:
        assert 4 in neighbors.tolist() and graph.knn_mask.all()
    # Same mesh in both runs: one computation, then an in-memory hit.
    assert dataset.local_knn_cache.counts["computed"] == 1
    assert dataset.local_knn_cache.counts["memory_hits"] == 1
    assert (tmp_path / "stats" / KNN_EDGE_STATS_FILE).exists()
    valid = graph.knn_mask.clone()
    valid[:, 0] = False
    standardized = graph.knn_edge_attr[valid]
    assert torch.isfinite(standardized).all()
    assert standardized.std(0).max() < 10
    # Padded slots repeat the node itself, so batching offsets stay valid.
    batched = Batch.from_data_list(dataset.graphs)
    torch.testing.assert_close(batched.knn_index[8:, 0], torch.arange(8, 16))
    assert batched.knn_index[8:].min() >= 8
    assert dataset[0].to(torch.device("meta")).graph.knn_index.device.type == "meta"
    assert dataset.graphs[0].knn_index.device.type == "cpu"
    # A second construction (validation-style load) reads the disk cache and stats.
    second = CrashGraphDataset(
        reader=reader, split="validation", stats_mode="load", **options
    )
    assert second.local_knn_cache.counts["disk_hits"] == 1
    torch.testing.assert_close(second.graphs[0].knn_index, graph.knn_index)
    torch.testing.assert_close(second.graphs[0].knn_edge_attr, graph.knn_edge_attr)


def test_cache_rejects_tampered_entries_and_metric_changes(tmp_path):
    positions = torch.tensor([[0.0, 0, 0], [1, 0, 0], [2.5, 0, 0], [4.5, 0, 0]])
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    cache = LocalNeighborhoodCache(tmp_path)
    first = cache.get(positions, edges, 2, "geodesic")
    assert first.index[0].tolist() == [1, 2]
    other = cache.get(positions, edges, 2, "euclidean")
    assert cache.counts["computed"] == 2
    torch.testing.assert_close(other.index, first.index)  # a line, no ties
    entry = next(tmp_path.glob("*.pt"))
    payload = torch.load(entry, weights_only=True)
    payload["index"][0, 0] = 3
    torch.save(payload, entry)
    with pytest.raises(RuntimeError, match="Invalid kNN cache entry"):
        LocalNeighborhoodCache(tmp_path).get(
            positions, edges, 2, payload["metadata"]["metric"]
        )
    with pytest.raises(ValueError, match="metric"):
        cache.get(positions, edges, 2, "hops")


def _knn_sample(k=3, metric="geodesic"):
    torch.manual_seed(11)
    n = 7
    coords = (torch.randn(n, 3) * 0.2).requires_grad_()
    previous = (coords.detach() - 0.01 * torch.randn(n, 3)).requires_grad_()
    source = torch.arange(n - 1)
    edges = torch.stack(
        (torch.cat((source, source + 1)), torch.cat((source + 1, source)))
    )
    graph = Data(edge_index=edges, edge_attr=torch.randn(2 * (n - 1), 4), num_nodes=n)
    graph = LocalNeighborhoodData(**graph.to_dict())
    neighborhood = LocalNeighborhoodCache().get(coords.detach(), edges, k, metric)
    graph.knn_index, graph.knn_mask = with_self_slot(neighborhood)
    graph.knn_edge_attr = knn_edge_features(coords.detach(), graph.knn_index)
    graph.knn_metric = metric
    sample = SimSample(
        {"coords": coords, "previous_coords": previous, "features": torch.randn(n, 1)},
        torch.randn(n, 4, 3),
        graph=graph,
    )
    zeros, ones = torch.zeros(1, 3), torch.ones(1, 3)
    stats = {
        "node": dict(
            pos_mean=ones * 4,
            pos_std=torch.tensor([[2.0, 3.0, 4.0]]),
            norm_vel_mean=zeros,
            norm_vel_std=ones,
            norm_acc_mean=zeros,
            norm_acc_std=ones,
        )
    }
    return sample, stats


def test_knn_autoregressive_rollout_trains_through_bptt():
    """Closed-loop BPTT with checkpointed kNN processors reaches every parameter."""
    sample, stats = _knn_sample()
    model = MeshGeoFLAREAutoregressive(
        num_time_steps=5,
        dt=0.1,
        functional_dim=4,
        out_dim=3,
        geometry_dim=3,
        global_dim=None,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        n_layers=1,
        use_te=False,
        attention_type="GALE_FA",
        input_dim_edges=4,
        mesh_hidden_dim=16,
        num_pre_processor_layers=1,
        num_post_processor_layers=2,
        num_pre_processor_checkpoint_segments=1,
        num_post_processor_checkpoint_segments=2,
        mesh_context_fusion="pre_post",
        mesh_pre_residual_gate_init=0.2,
        mesh_post_residual_gate_init=0.1,
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
        local_knn_k=3,
        local_neighborhood="geodesic",
        local_num_heads=4,
        local_edge_dim=8,
    ).train()
    trajectory = model(sample=sample, data_stats=stats)
    assert trajectory.shape == (7, 4, 3) and torch.isfinite(trajectory).all()
    loss = torch.nn.functional.mse_loss(trajectory, sample.node_target)
    loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith(("pre_processor", "post_processor")):
            assert (
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
            ), name
    assert sample.node_features["coords"].grad is not None
    wrong, _ = _knn_sample(metric="euclidean")
    with pytest.raises(ValueError, match="metric"):
        model(sample=wrong, data_stats=stats)
