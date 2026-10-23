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
"""MeshGraphNet messages on static kNN slots, and the raw-values attention arm."""

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))
CONF = CRASH_DIR / "conf"

from rollout import MeshGeoFLAREAutoregressive  # noqa: E402
from test_knn_v2_recipe import _tiny_model  # noqa: E402
from test_local_knn_recipe import _knn_sample  # noqa: E402

from physicsnemo.experimental.models.meshtransolver.local_attention import (  # noqa: E402
    LocalNeighborhood,
)
from physicsnemo.experimental.models.meshtransolver.meshtransolver import (  # noqa: E402
    MeshGeoFLARE,
)


def _compose(name):
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose(
            config_name=name,
            overrides=[
                "training.raw_data_dir=/fixtures/train",
                "training.raw_data_dir_validation=/fixtures/validation",
                "inference.raw_data_dir_test=/fixtures/test",
            ],
        )


def test_knn_slot_graph_has_one_directed_edge_per_valid_slot():
    index = torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 2]])
    mask = torch.tensor([[True, True, True], [True, True, False], [True, True, False]])
    features = torch.arange(27, dtype=torch.float32).view(3, 3, 3)
    graph, slot_features = MeshGeoFLARE._knn_slot_graph(
        LocalNeighborhood(index, mask, features)
    )
    assert graph.num_nodes == 3 and graph.num_edges == 7
    source, destination = graph.edge_index
    # destination is the node owning the slot; the self slot is a self loop
    assert destination.tolist() == [0, 0, 0, 1, 1, 2, 2]
    assert source.tolist() == [0, 1, 2, 1, 0, 2, 1]
    torch.testing.assert_close(slot_features, features[mask])


def test_knn_mpnn_recipe_is_the_knn_recipe_with_mpnn_messages():
    base = _compose("crash_deformer_knn_autoregressive")
    cfg = _compose("crash_deformer_knn_mpnn_autoregressive")
    assert cfg.model.local_processor == "knn_mpnn"
    assert cfg.model.local_knn_k == 8 and cfg.model.local_neighborhood == "geodesic"
    for key in base.model:
        if key in ("local_processor", "local_knn_k"):
            continue
        assert base.model[key] == cfg.model[key], key
    assert set(cfg.model) == set(base.model)
    assert base.training == cfg.training and base.reader == cfg.reader
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["datapipe"]["local_knn_k"] == 8
    assert resolved["datapipe"]["local_knn_metric"] == "geodesic"
    # The MPNN widths and processor depths are the DeFormer ones.
    assert cfg.model.mesh_hidden_dim == 256 and cfg.model.num_post_processor_layers == 2


def test_rawvalues_recipe_keeps_the_residual_update():
    v2 = _compose("crash_deformer_knn_v2_autoregressive")
    raw = _compose("crash_deformer_knn_rawvalues_autoregressive")
    assert (
        raw.model.local_value_norm is False and raw.model.local_split_key_value is True
    )
    assert raw.model.local_bias_fp32 is True and raw.model.local_attention_fused is True
    assert (
        "local_difference_values" not in raw.model and "local_update" not in raw.model
    )
    assert raw.training.muon_min_dim == 9
    for key in raw.model:
        if key in v2.model:
            assert raw.model[key] == v2.model[key], key


def test_knn_mpnn_model_trains_through_bptt_with_the_mpnn_processors():
    sample, stats = _knn_sample()
    model = _tiny_model(local_processor="knn_mpnn", local_knn_k=3).train()
    assert model.edge_encoder is not None
    assert model.pre_processor.__class__.__name__ == "MeshGraphNetProcessor"
    assert not hasattr(model.pre_processor, "blocks")
    trajectory = model(sample=sample, data_stats=stats)
    assert trajectory.shape == (7, 4, 3) and torch.isfinite(trajectory).all()
    torch.nn.functional.mse_loss(trajectory, sample.node_target).backward()
    grads = {
        name
        for name, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    }
    assert any(name.startswith("edge_encoder") for name in grads)
    assert any(name.startswith("pre_processor") for name in grads)
    assert any(name.startswith("post_processor") for name in grads)
    # The structural edge attributes are ignored: perturbing them changes nothing.
    model.eval()
    with torch.no_grad():
        before = model(sample=sample, data_stats=stats)
        sample.graph.edge_attr = sample.graph.edge_attr + 10.0
        after = model(sample=sample, data_stats=stats)
    torch.testing.assert_close(before, after)
    assert isinstance(model, MeshGeoFLAREAutoregressive)
