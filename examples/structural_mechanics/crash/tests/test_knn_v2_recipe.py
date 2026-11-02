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
"""Recipe contract for the second kNN-attention design and its optimizer routing."""

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))
CONF = CRASH_DIR / "conf"

from rollout import MeshGeoFLAREAutoregressive  # noqa: E402
from test_local_knn_recipe import _knn_sample  # noqa: E402
from utils import build_muon_optimizer  # noqa: E402

V2_KEYS = {
    "local_value_norm": False,
    "local_difference_values": True,
    "local_update": "concat_mlp",
    "local_bias_fp32": True,
    "local_split_key_value": True,
    "local_attention_fused": True,
}


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


def _tiny_model(**overrides):
    kwargs = dict(
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
        local_attention_backend="torch",
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return MeshGeoFLAREAutoregressive(**kwargs)


def test_v2_model_trains_through_bptt_and_routes_small_matrices_to_adamw():
    sample, stats = _knn_sample()
    model = _tiny_model(**V2_KEYS).train()
    trajectory = model(sample=sample, data_stats=stats)
    assert trajectory.shape == (7, 4, 3) and torch.isfinite(trajectory).all()
    torch.nn.functional.mse_loss(trajectory, sample.node_target).backward()
    block = model.pre_processor.blocks[0]
    assert block.update == "concat_mlp" and block.update_mlp[0].weight.grad is not None
    assert (
        block.value.weight.grad is not None and block.edge_bias.weight.grad is not None
    )
    cfg = OmegaConf.create(
        {
            "training": {
                "start_lr": 1e-3,
                "optimizer_weight_decay": 0.0,
                "muon_min_dim": 9,
            }
        }
    )
    optimizer = build_muon_optimizer(model, cfg)
    muon = [p for p in optimizer.optimizers[0].param_groups[0]["params"]]
    adamw = [p for p in optimizer.optimizers[1].param_groups[0]["params"]]
    assert optimizer.optimizers[0].__class__.__name__ == "Muon"
    assert any(p is block.edge_bias.weight for p in adamw)  # [4, 8] -> AdamW
    assert any(p is model.pre_processor.edge_encoder[0].weight for p in adamw)  # [8, 4]
    assert any(p is block.value.weight for p in muon)  # [16, 16] -> Muon
    legacy = OmegaConf.create(
        {"training": {"start_lr": 1e-3, "optimizer_weight_decay": 0.0}}
    )
    legacy_muon = (
        build_muon_optimizer(model, legacy).optimizers[0].param_groups[0]["params"]
    )
    assert any(p is block.edge_bias.weight for p in legacy_muon)
