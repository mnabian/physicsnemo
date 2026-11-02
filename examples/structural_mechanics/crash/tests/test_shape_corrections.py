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
"""Evaluation-time geometric corrections (shape_corrections.py)."""

import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))

from datapipe import SimSample  # noqa: E402
from shape_corrections import (  # noqa: E402
    PenetrationProjection,
    RigidWallClamp,
    correct_shape,
    physical_geometry,
    wall_position,
)
from surface_topology import (  # noqa: E402
    SurfaceContactData,
    surface_one_ring_exclusions,
)

MEAN = torch.tensor([[1.0, 2.0, 3.0]])
STD = torch.tensor([[2.0, 2.0, 2.0]])


def _scene():
    """One triangular shell (thickness 1) in z = 0; node 3 overlaps the envelope
    from above, node 4 has crossed the shell. Positions in physical millimetres,
    the datapipe's normalization is (x - MEAN) / STD."""
    reference = torch.tensor(
        [[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [2, 2, 3.0], [3, 2, 3.0]]
    )
    current = reference.clone()
    current[3, 2] = 0.6
    current[4, 2] = -0.5
    faces = torch.tensor([[0, 1, 2, 2]])
    graph = SurfaceContactData(
        **Data(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=5).to_dict()
    )
    graph.contact_faces = faces
    graph.contact_surface_exclusions = surface_one_ring_exclusions(faces, 5)
    graph.shell_thickness = torch.ones(5)
    graph.rigid_wall = torch.tensor([2.5])
    sample = SimSample({"coords": (reference - MEAN) / STD}, None, graph=graph)
    stats = {"node": {"pos_mean": MEAN, "pos_std": STD}}
    return sample, stats, (current - MEAN) / STD, current


def test_projection_and_clamp_in_normalized_coordinates():
    sample, stats, prediction, current = _scene()
    projection = PenetrationProjection(
        iterations=1, implementation="torch", clearance_mm=None
    )
    clamp = RigidWallClamp(enabled=True)
    assert projection.enabled and projection.search().side_from_reference
    projected = correct_shape(prediction, sample, stats, projection=projection)
    physical = projected * STD + MEAN
    torch.testing.assert_close(physical[3], torch.tensor([2.0, 2.0, 1.0]))
    torch.testing.assert_close(physical[4], torch.tensor([3.0, 2.0, 1.0]))
    torch.testing.assert_close(physical[:3], current[:3])
    clamped = correct_shape(prediction, sample, stats, clamp=clamp)
    physical = clamped * STD + MEAN
    torch.testing.assert_close(physical[:, 0], torch.tensor([0.0, 2.5, 0.0, 2.0, 2.5]))
    torch.testing.assert_close(physical[:, 1:], current[:, 1:])
    both = correct_shape(prediction, sample, stats, projection=projection, clamp=clamp)
    physical = both * STD + MEAN
    torch.testing.assert_close(physical[4], torch.tensor([2.5, 2.0, 1.0]))
    # Nothing enabled: the prediction itself comes back.
    assert correct_shape(prediction, sample, stats) is prediction
    assert (
        correct_shape(
            prediction,
            sample,
            stats,
            projection=PenetrationProjection(),
            clamp=RigidWallClamp(),
        )
        is prediction
    )


def test_corrections_report_missing_graph_data_and_bad_settings():
    sample, stats, prediction, _ = _scene()
    positions, reference, thickness = physical_geometry(prediction, sample, stats)
    torch.testing.assert_close(reference[4], torch.tensor([3.0, 2.0, 3.0]))
    torch.testing.assert_close(thickness, torch.ones(5))
    assert wall_position(sample) == 2.5
    del sample.graph.rigid_wall
    with pytest.raises(ValueError, match="rigid_wall"):
        correct_shape(prediction, sample, stats, clamp=RigidWallClamp(enabled=True))
    del sample.graph.contact_faces
    with pytest.raises(ValueError, match="contact_faces"):
        correct_shape(
            prediction,
            sample,
            stats,
            projection=PenetrationProjection(iterations=1, implementation="torch"),
        )
    del sample.graph.shell_thickness
    with pytest.raises(ValueError, match="shell_thickness"):
        physical_geometry(prediction, sample, stats)
    for bad in (
        dict(iterations=-1),
        dict(iterations=1.5),
        dict(damping=0.0),
        dict(damping=1.5),
        dict(tolerance_mm=-1.0),
        dict(band_mm=0.0),
        dict(clearance_mm=-1.0),
        dict(implementation="numpy"),
    ):
        with pytest.raises(ValueError):
            PenetrationProjection(**bad)
    with pytest.raises(ValueError, match="wall_axis"):
        RigidWallClamp(axis=5)
    with pytest.raises(ValueError, match="wall_sign"):
        RigidWallClamp(sign=0.5)
