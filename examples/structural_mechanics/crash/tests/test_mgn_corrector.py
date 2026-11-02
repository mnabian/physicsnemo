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
"""MeshGraphNet corrector (mgn_corrector.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from local_neighborhoods import mesh_edge_features  # noqa: E402
from mgn_corrector import MeshGraphNetCorrector  # noqa: E402
from test_time_conditional_recipe import (  # noqa: E402
    FIXTURE_OVERRIDES,
    _contact_sample,
)

CONF = HERE.parent / "conf"
TIME_DIM = 8

# Wrapper-only keys of the corrector recipe's model block, never passed to the student.
WRAPPER_KEYS = (
    "_target_",
    "_convert_",
    "teacher",
    "teacher_weights",
    "final_clamp_to_wall",
    "wall_axis",
    "wall_sign",
    "final_projection_iterations",
    "final_projection_damping",
    "final_projection_side_symmetric",
    "final_projection_crossed_only",
    "final_projection_tolerance_mm",
    "final_projection_band_mm",
    "final_projection_clearance_mm",
)


def _mgn_kwargs(**overrides):
    kwargs = dict(
        num_time_steps=5,
        functional_dim=3 + 1 + TIME_DIM + 3,
        hidden_dim=16,
        processor_size=2,
        time_fourier_bands=3,
        time_dim=TIME_DIM,
        use_contact=True,
        contact_search_implementation="torch",
        contact_reference_clearance=None,  # the two sheets are assembled 0.5 apart
    )
    kwargs.update(overrides)
    return kwargs


def _mesh_sample(gap, time=0.5, scale=0.05):
    """The two-sheet contact sample with mesh edges, and a teacher displacement."""
    sample, stats = _contact_sample(gap=gap, time=time)
    graph = sample.graph
    coords = sample.node_features["coords"]
    graph.mesh_edge_index = graph.edge_index.clone()
    attr = mesh_edge_features(coords, graph.mesh_edge_index)
    graph.mesh_edge_mean = attr.mean(dim=0)
    graph.mesh_edge_std = attr.std(dim=0) + 1e-3
    graph.mesh_edge_attr = (attr - graph.mesh_edge_mean) / graph.mesh_edge_std
    torch.manual_seed(21)
    displacement = scale * torch.randn_like(coords)
    return sample, stats, displacement


def _open(model, std=0.1):
    torch.manual_seed(7)
    nn.init.normal_(model.decoder[-1].weight, std=std)


def _compose_corrector():
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose(
            config_name="yaris_deformer_corrector",
            overrides=[
                *FIXTURE_OVERRIDES,
                "model.teacher_weights=/fixtures/teacher.mdlus",
            ],
        )


def test_corrector_starts_at_the_teacher_field():
    torch.manual_seed(1)
    model = MeshGraphNetCorrector(**_mgn_kwargs()).train()
    sample, stats, displacement = _mesh_sample(gap=0.5)
    coords = sample.node_features["coords"]
    with torch.no_grad():
        out = model(
            sample, stats, teacher_displacement=displacement, return_auxiliary=True
        )
    torch.testing.assert_close(out.prediction, coords + displacement)
    torch.testing.assert_close(out.coarse_prediction, coords + displacement)
    assert model(sample, stats, teacher_displacement=displacement).shape == (8, 3)
    assert model.rollout_steps == 4 and model.dt is None
    sample.node_features["time"] = torch.tensor(1.5)
    with pytest.raises(ValueError, match="future-frame range"):
        model(sample, stats, teacher_displacement=displacement)


def test_corrector_builds_contact_edges_on_the_predicted_shape_and_trains():
    torch.manual_seed(1)
    model = MeshGraphNetCorrector(**_mgn_kwargs()).train()
    _open(model)
    seen = []
    hook = model.processor.register_forward_hook(
        lambda module, args, output: seen.append(args[2].edge_index.shape[1])
    )
    close, stats, displacement = _mesh_sample(gap=0.5)
    far, _, _ = _mesh_sample(gap=50.0)
    mesh_edges = close.graph.mesh_edge_index.shape[1]
    try:
        out = model(
            close, stats, teacher_displacement=displacement, return_auxiliary=True
        )
        out.prediction.square().sum().backward()
        assert seen[-1] > mesh_edges  # contact edges joined the mesh edges
        grads = {n: p.grad for n, p in model.named_parameters()}
        for name in (
            "contact_edge_encoder.model.0.weight",
            "mesh_edge_encoder.model.0.weight",
            "node_encoder.model.0.weight",
            "decoder.1.weight",
        ):
            assert grads[name] is not None and grads[name].abs().sum() > 0, name
        model.zero_grad()
        # Far sheets: no contact edges, but every contact parameter stays in the
        # graph through the zero placeholder (same layout on every rank).
        out = model(far, stats, teacher_displacement=displacement)
        out.square().sum().backward()
        assert seen[-1] == mesh_edges
        grad = model.contact_edge_encoder.model[0].weight.grad
        assert grad is not None and grad.abs().sum() == 0
    finally:
        hook.remove()
    # Bidirectional edges double the contact edge count.
    one_way = MeshGraphNetCorrector(**_mgn_kwargs(contact_bidirectional=False)).train()
    counts = []
    hooks = [
        m.processor.register_forward_hook(
            lambda module, args, output: counts.append(args[2].edge_index.shape[1])
        )
        for m in (model, one_way)
    ]
    try:
        with torch.no_grad():
            model(close, stats, teacher_displacement=displacement)
            one_way(close, stats, teacher_displacement=displacement)
    finally:
        for hook in hooks:
            hook.remove()
    assert counts[0] - mesh_edges == 2 * (counts[1] - mesh_edges) > 0


def test_corrector_reads_the_predicted_geometry_on_the_mesh_edges():
    torch.manual_seed(1)
    model = MeshGraphNetCorrector(**_mgn_kwargs(use_contact=False)).train()
    assert model.contact_search is None and model.contact_edge_encoder is None
    _open(model)
    sample, stats, displacement = _mesh_sample(gap=50.0)
    coords = sample.node_features["coords"]
    with torch.no_grad():
        rigid = model(sample, stats, teacher_displacement=displacement) - coords
        # A different, non-rigid field changes the mesh-edge geometry and the correction.
        bent_field = displacement * torch.linspace(0.0, 2.0, coords.shape[0]).unsqueeze(
            1
        )
        bent = model(sample, stats, teacher_displacement=bent_field) - coords
    assert not torch.allclose(rigid - displacement, bent - bent_field)
    del sample.graph.mesh_edge_index
    with pytest.raises(ValueError, match="mesh_edge_index"):
        model(sample, stats, teacher_displacement=displacement)


def test_corrector_rejects_bad_options_and_unknown_keys():
    with pytest.raises(TypeError, match="unexpected"):
        MeshGraphNetCorrector(**_mgn_kwargs(n_hidden=16))
    # The datapipe's keys of the model config are accepted; the retired oracle
    # option is accepted at its released value only.
    MeshGraphNetCorrector(**_mgn_kwargs(local_knn_k=32, local_neighborhood="euclidean"))
    MeshGraphNetCorrector(**_mgn_kwargs(contact_oracle=False))
    with pytest.raises(ValueError, match="research option"):
        MeshGraphNetCorrector(**_mgn_kwargs(contact_oracle=True))
    checkpointed = MeshGraphNetCorrector(
        **_mgn_kwargs(processor_size=4, activation_checkpointing=True)
    )
    assert checkpointed.processor.num_processor_checkpoint_segments == 4
    with pytest.raises(ValueError, match="divisible"):
        MeshGraphNetCorrector(
            **_mgn_kwargs(processor_size=3, num_processor_checkpoint_segments=2)
        )
    with pytest.raises(ValueError, match="aggregation"):
        MeshGraphNetCorrector(**_mgn_kwargs(aggregation="max"))
    with pytest.raises(ValueError, match="processor_size"):
        MeshGraphNetCorrector(**_mgn_kwargs(processor_size=0))
    with pytest.raises(ValueError, match="side_from_reference"):
        MeshGraphNetCorrector(
            **_mgn_kwargs(
                contact_side_from_reference=False, contact_side_symmetric=True
            )
        )
    sample, stats, displacement = _mesh_sample(gap=50.0)
    with pytest.raises(ValueError, match="functional_dim"):
        MeshGraphNetCorrector(
            **_mgn_kwargs(functional_dim=12, use_contact=False)
        ).train()(sample, stats, teacher_displacement=displacement)


def test_window_corrector_reads_the_teacher_frames_before_and_after():
    torch.manual_seed(1)
    model = MeshGraphNetCorrector(
        **_mgn_kwargs(
            teacher_window=True,
            functional_dim=3 + 1 + TIME_DIM + 3 + 6,
            use_contact=False,
        )
    ).train()
    assert model.teacher_window is True
    sample, stats, displacement = _mesh_sample(gap=50.0)
    coords = sample.node_features["coords"]
    with pytest.raises(ValueError, match="teacher_window"):
        model(sample, stats, teacher_displacement=displacement)
    window = torch.stack((0.5 * displacement, 1.5 * displacement), dim=1)
    with torch.no_grad():  # zero-initialised decoder: still the teacher's field
        torch.testing.assert_close(
            model(
                sample, stats, teacher_displacement=displacement, teacher_window=window
            ),
            coords + displacement,
        )
    _open(model)
    with torch.no_grad():
        moving = model(
            sample, stats, teacher_displacement=displacement, teacher_window=window
        )
        still = model(
            sample,
            stats,
            teacher_displacement=displacement,
            teacher_window=torch.stack((displacement, displacement), dim=1),
        )
    assert not torch.allclose(moving, still)  # the window reaches the correction
    plain = MeshGraphNetCorrector(**_mgn_kwargs(use_contact=False))
    assert (
        plain.teacher_window is False and plain.functional_dim == 3 + 1 + TIME_DIM + 3
    )


def test_discovery_corrections_reach_the_corrector_search():
    model = MeshGraphNetCorrector(
        **_mgn_kwargs(
            contact_reference_clearance=10.0,
            contact_side_symmetric=True,
            contact_crossing_radius=12.0,
            contact_clearance_approach_only=True,
        )
    )
    search = model.contact_search
    assert search.side_symmetric is True and search.crossing_radius == 12.0
    assert search.clearance_approach_only is True and search.reference_clearance == 10.0
    default = MeshGraphNetCorrector(**_mgn_kwargs()).contact_search
    assert default.side_symmetric is False and default.crossing_radius is None
    assert default.clearance_approach_only is False and default.side_from_reference


def test_corrector_recipe_composes_into_the_corrector():
    cfg = _compose_corrector()
    assert cfg.model.functional_dim == 39 and cfg.model.processor_size == 8
    assert cfg.model.hidden_dim == 128 and cfg.model.use_contact is True
    assert cfg.model.contact_activation_distance == 5.0
    assert cfg.model.contact_reference_clearance == 10.0
    assert cfg.model.contact_side_from_reference is True
    assert cfg.model.teacher_window is False
    datapipe = OmegaConf.to_container(cfg.datapipe, resolve=True)
    assert datapipe["local_knn_k"] == 32 and datapipe["local_knn_metric"] == "euclidean"
    assert datapipe["dt"] == 0.005 and datapipe["rigid_wall_position"] == 100.0
    assert cfg.training.epochs == 30 and cfg.training.lr_warmup_epochs == 1
    assert cfg.training.coarse_loss_weight == 0.0
    assert cfg.training.validation_penetration.enabled is True
    student = OmegaConf.to_container(cfg.model, resolve=True)
    for key in WRAPPER_KEYS:
        student.pop(key)
    corrector = MeshGraphNetCorrector(
        **{**student, "contact_search_implementation": "torch"}
    )
    assert corrector.rollout_steps == 30
    assert corrector.processor.num_processor_checkpoint_segments == 8
    assert sum(p.numel() for p in corrector.parameters()) == 1_250_659
