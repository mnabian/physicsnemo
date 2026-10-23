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
"""The area-quadrature switch is default-off and reaches the surface builder."""

import sys
from pathlib import Path

import pytest

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))
from rollout import MeshGeoFLAREAutoregressive  # noqa: E402


def _surface_model(**overrides):
    kwargs = dict(
        num_time_steps=4,
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
        num_post_processor_layers=1,
        mesh_context_fusion="pre_post",
        initial_velocity_mode="previous_coords",
        rollout_steps_from_target=True,
        checkpoint_rollout=True,
        teacher_forcing=False,
        node_input_mode="velocity",
        use_contact=True,
        enable_contact=True,
        enable_node_contact=True,
        enable_cylinder_contact=False,
        contact_graph_backend="surface",
        contact_search_implementation="torch",
        node_contact_radius=10.0,
        contact_include_velocity=True,
        contact_velocity_scale=1000.0,
        contact_smooth_cutoff=True,
        contact_activation_distance=5.0,
        contact_normal_epsilon=0.1,
        base_shell_thickness=0.0,
        contact_surface_predictive=True,
        contact_surface_material_fan=True,
    )
    kwargs.update(overrides)
    return MeshGeoFLAREAutoregressive(**kwargs)


def test_area_quadrature_switch_is_default_off_and_plumbed():
    default = _surface_model()
    assert default.node_contact_builder.area_weighting is False
    assert default.node_contact_builder.reference_area is None
    enabled = _surface_model(
        contact_surface_area_weighting=True, contact_surface_reference_area=78.54
    )
    assert enabled.node_contact_builder.area_weighting is True
    assert enabled.node_contact_builder.reference_area == pytest.approx(78.54)
    with pytest.raises(ValueError, match="surface backend"):
        _surface_model(
            contact_graph_backend="legacy",
            contact_surface_predictive=False,
            contact_surface_material_fan=False,
            contact_surface_area_weighting=True,
        )
