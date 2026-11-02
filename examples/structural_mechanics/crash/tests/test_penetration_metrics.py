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
"""Target-relative penetration metrics on a hand-checked plate scene (CPU)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from penetration_metrics import (  # noqa: E402
    FAMILIES,
    PenetrationMetrics,
    histogram_percentiles,
    region_names,
)
from surface_topology import surface_one_ring_exclusions  # noqa: E402

from physicsnemo.nn.functional.shell_contact import node_maximum  # noqa: E402

TOL = 0.5
A, B, C, D = 9, 10, 11, 12  # query nodes after the 9 plate nodes
E = 13  # optional slider node (plate_scene(slider=True))


def plate_scene(
    wall=None, pos_std=(1.0, 1.0, 1.0), pos_mean=(0.0, 0.0, 0.0), slider=False
):
    """A 2x2 quad plate (4 mm squares, z = 0, thickness 1 mm) and four query nodes.

    Reference: queries 15 mm above the plate (outside the 10 mm reference
    clearance). Prediction vs target per query:
      A crosses facet 0 by 0.6 mm (envelope 1.6, crossing 0.6); target clean (z = 2)
      B inside the envelope by 0.7 mm in BOTH frames (no excess)
      C clean in the prediction; target crossed by 1.5 mm (missed)
      D far from the plate; beyond a wall at x = 4.5 in the prediction only
    ``slider`` adds E: beside the plate in the reference (x = -12, 2 mm above
    the plane, 12.2 mm from facet 0) and at A's predicted position (0.6 mm
    under facet 0's interior) in the prediction; target clean. The builder's
    one-sided test calls that a crossing, the symmetric test does not.
    """
    grid = [(x, y, 0.0) for y in (0.0, 2.0, 4.0) for x in (0.0, 2.0, 4.0)]
    node = lambda col, row: 3 * row + col  # noqa: E731
    faces = torch.tensor(
        [
            [node(c, r), node(c + 1, r), node(c + 1, r + 1), node(c, r + 1)]
            for r in range(2)
            for c in range(2)
        ],
        dtype=torch.long,
    )
    reference = torch.tensor(
        grid
        + [(1.0, 1.0, 15.0), (3.0, 1.0, 15.0), (1.0, 3.0, 15.0), (50.0, 1.0, 15.0)]
        + ([(-12.0, 1.0, 2.0)] if slider else [])
    )
    prediction = reference.clone()
    prediction[A] = torch.tensor([1.0, 1.0, -0.6])
    if slider:
        prediction[E] = torch.tensor([1.0, 1.0, -0.6])
    prediction[B] = torch.tensor([3.0, 1.0, 0.3])
    prediction[C] = torch.tensor([1.0, 3.0, 4.0])
    target = reference.clone()
    target[A] = torch.tensor([1.0, 1.0, 2.0])
    target[B] = torch.tensor([3.0, 1.0, 0.3])
    target[C] = torch.tensor([1.0, 3.0, -1.5])
    target[D] = torch.tensor([3.0, 1.0, 15.0])
    n = reference.shape[0]
    graph = SimpleNamespace(
        contact_faces=faces,
        contact_surface_exclusions=surface_one_ring_exclusions(faces, n),
        shell_thickness=torch.ones(n),
    )
    if wall is not None:
        graph.rigid_wall = torch.tensor([float(wall)])
    std = torch.tensor(pos_std)
    mean = torch.tensor(pos_mean)
    sample = SimpleNamespace(
        graph=graph, node_features={"coords": (reference - mean) / std}
    )
    stats = {"node": {"pos_std": std, "pos_mean": mean}}
    normalize = lambda x: (x - mean) / std  # noqa: E731
    return sample, stats, normalize(prediction), normalize(target)


def metrics(cfg=None, steps=1, region_bounds=(2.0,)):
    base = dict(implementation="torch", tolerance_mm=TOL)
    base.update(cfg or {})
    return PenetrationMetrics(
        base, steps, torch.device("cpu"), region_bounds=region_bounds
    )


def run(m, steps=1, **scene_kwargs):
    sample, stats, pred, true = plate_scene(**scene_kwargs)
    m.accumulate(
        pred[:, None].expand(-1, steps, -1),
        true[:, None].expand(-1, steps, -1),
        sample,
        stats,
    )
    m.reduce(1)
    return m.summary(torch.ones(steps))


def test_helpers():
    assert region_names([-2000.0, -1000.0]) == [
        "x_-inf_to_-2000",
        "x_-2000_to_-1000",
        "x_-1000_to_inf",
        "all",
    ]
    out = node_maximum(4, torch.tensor([0, 0, 2]), torch.tensor([1.0, 3.0, 2.0]))
    torch.testing.assert_close(out, torch.tensor([3.0, 0.0, 2.0, 0.0]))
    hist = torch.zeros(11)
    hist[3] = 4  # four values in [0.3, 0.4)
    hist[10] = 1  # one overflow
    assert histogram_percentiles(hist, 0.1, (50, 99)) == {"p50": 0.4, "p99": 1.1}
    assert histogram_percentiles(torch.zeros(11), 0.1, (50,)) == {"p50": 0.0}


def test_surface_families_against_hand_values():
    summary = run(metrics())
    assert summary["complete"] is True and summary["wall"] is None
    env = summary["surface"]["envelope"]["per_region"]["all"]
    assert env["node_frames"] == 13
    assert env["penetrating_fraction"] == {"prediction": 2 / 13, "target": 2 / 13}
    assert env["spurious_count"] == 1 and env["missed_count"] == 1
    assert env["missed_fraction"] == pytest.approx(0.5)
    assert env["spurious_depth_mean_mm"] == pytest.approx(1.6, abs=1e-5)
    assert env["spurious_depth_max_mm"] == pytest.approx(1.6, abs=1e-5)
    assert env["excess_mean_mm"] == pytest.approx(1.6 / 13, abs=1e-6)
    assert env["excess_max_mm"] == pytest.approx(1.6, abs=1e-5)
    assert env["max_depth_mm"]["prediction"] == pytest.approx(1.6, abs=1e-5)
    assert env["max_depth_mm"]["target"] == pytest.approx(2.5, abs=1e-5)
    assert env["spurious_above_mm"]["1"]["count"] == 1
    assert env["spurious_above_mm"]["2"]["count"] == 0
    assert env["spurious_above_mm"]["5"]["fraction"] == 0
    assert env["spurious_depth_percentiles_mm"]["p50"] == pytest.approx(1.65, abs=0.051)
    assert env["spurious_depth_percentiles_mm"]["p99.9"] == pytest.approx(
        1.65, abs=0.051
    )
    crossing = summary["surface"]["crossing"]["per_region"]["all"]
    assert crossing["penetrating_fraction"] == {"prediction": 1 / 13, "target": 1 / 13}
    assert crossing["spurious_count"] == 1 and crossing["missed_count"] == 1
    assert crossing["spurious_depth_mean_mm"] == pytest.approx(0.6, abs=1e-5)
    assert crossing["excess_mean_mm"] == pytest.approx(0.6 / 13, abs=1e-6)
    assert crossing["max_depth_mm"]["target"] == pytest.approx(1.5, abs=1e-5)
    assert crossing["spurious_above_mm"]["1"]["count"] == 0
    # Legacy blocks: one crossed pair each, depths with the tolerance subtracted.
    assert summary["prediction"]["crossed_pairs"] == 1
    assert summary["target"]["crossed_pairs"] == 1
    assert summary["prediction"]["penetrating_node_fraction"] == pytest.approx(2 / 13)
    assert summary["prediction"]["mean_depth_mm"] == pytest.approx(1.3 / 13, abs=1e-6)
    assert summary["target"]["mean_depth_mm"] == pytest.approx(2.2 / 13, abs=1e-6)
    assert summary["pairs_per_time"]["crossed_pairs_prediction"] == [1.0]
    assert summary["pairs_per_time"]["active_pairs_target"][0] >= 3
    # Per-time arrays of the whole mesh.
    per_time = summary["surface"]["envelope"]["per_time"]
    assert per_time["spurious_fraction"] == [pytest.approx(1 / 13)]
    assert per_time["missed_fraction"] == [pytest.approx(0.5)]
    assert per_time["spurious_above_1_fraction"] == [pytest.approx(1 / 13)]
    json.dumps(summary)


def test_regions_split_by_reference_x():
    summary = run(metrics())
    env = summary["surface"]["envelope"]["per_region"]
    assert list(env) == ["x_-inf_to_2", "x_2_to_inf", "all"]
    assert env["x_-inf_to_2"]["node_frames"] == 5  # three plate nodes at x=0, A, C
    assert env["x_2_to_inf"]["node_frames"] == 8
    assert (
        env["x_-inf_to_2"]["spurious_count"] == 1
        and env["x_2_to_inf"]["spurious_count"] == 0
    )
    assert env["x_-inf_to_2"]["missed_count"] == 1
    assert env["x_2_to_inf"]["penetrating_fraction"] == {
        "prediction": 1 / 8,
        "target": 1 / 8,
    }
    assert env["x_2_to_inf"]["excess_mean_mm"] == 0.0
    assert env["x_-inf_to_2"]["max_depth_mm"]["target"] == pytest.approx(2.5, abs=1e-5)


def test_wall_from_config_and_from_graph():
    for m, kwargs in (
        (metrics({"wall_position_mm": 4.5}), {}),
        (metrics(), {"wall": 4.5}),
    ):
        summary = run(m, **kwargs)
        assert summary["wall"] is not None
        env = summary["wall"]["envelope"]["per_region"]["all"]
        cross = summary["wall"]["crossing"]["per_region"]["all"]
        assert env["penetrating_fraction"] == {"prediction": 1 / 13, "target": 0.0}
        assert env["spurious_count"] == 1 and env["missed_count"] == 0
        assert env["spurious_depth_mean_mm"] == pytest.approx(46.0, abs=1e-4)
        assert cross["spurious_depth_mean_mm"] == pytest.approx(45.5, abs=1e-4)
        assert cross["excess_max_mm"] == pytest.approx(45.5, abs=1e-4)
        assert env["spurious_above_mm"]["5"]["count"] == 1
        assert cross["spurious_depth_percentiles_mm"]["p90"] == pytest.approx(
            45.55, abs=0.051
        )
        regions = summary["wall"]["crossing"]["per_region"]
        assert regions["x_2_to_inf"]["spurious_count"] == 1
        assert regions["x_-inf_to_2"]["spurious_count"] == 0
    # The config's wall position wins over the graph's.
    summary = run(metrics({"wall_position_mm": 100.0}), wall=4.5)
    assert summary["wall"]["crossing"]["per_region"]["all"]["spurious_count"] == 0


def test_denormalization_and_multiple_steps():
    reference = run(metrics())
    scaled = run(metrics(), pos_std=(2.0, 0.5, 4.0), pos_mean=(10.0, -3.0, 1.0))
    for family in ("envelope", "crossing"):
        a = reference["surface"][family]["per_region"]["all"]
        b = scaled["surface"][family]["per_region"]["all"]
        for key in ("spurious_count", "missed_count"):
            assert a[key] == b[key]
        assert a["excess_mean_mm"] == pytest.approx(b["excess_mean_mm"], abs=1e-5)
    three = run(metrics(steps=3), steps=3)
    env = three["surface"]["envelope"]
    assert env["per_region"]["all"]["node_frames"] == 39
    assert env["per_region"]["all"]["spurious_count"] == 3
    assert len(env["per_time"]["spurious_fraction"]) == 3
    assert env["per_time"]["excess_mean_mm"] == [pytest.approx(1.6 / 13, abs=1e-6)] * 3
    assert three["prediction"]["crossed_pairs"] == 3
    # Legacy maxima subtract the tolerance, as the previous probe did.
    assert three["prediction"]["max_depth_mm"] == pytest.approx(1.1, abs=1e-5)
    # A shorter horizon than the configured steps only reports the seen steps.
    assert (
        len(
            metrics(steps=5).summary(torch.ones(2))["prediction"][
                "crossed_pairs_per_time"
            ]
        )
        == 2
    )


def test_identical_frames_have_no_spurious_or_excess():
    m = metrics({"wall_position_mm": 4.5})
    sample, stats, pred, _ = plate_scene()
    m.accumulate(pred[:, None], pred[:, None], sample, stats)
    summary = m.summary(torch.ones(1))
    for block in (summary["surface"], summary["wall"]):
        for family in ("envelope", "crossing"):
            s = block[family]["per_region"]["all"]
            assert s["spurious_count"] == 0 and s["missed_count"] == 0
            assert s["excess_mean_mm"] == 0.0 and s["excess_max_mm"] == 0.0
            assert s["spurious_depth_percentiles_mm"]["p99"] == 0.0
    assert summary["prediction"] == summary["target"]


def test_symmetric_sidedness_drops_slid_under_pairs():
    """E slid under facet 0's edge: no crossing, 0.4 mm inside the envelope."""
    sample, stats, pred, _ = plate_scene(slider=True)
    graph, thickness = sample.graph, sample.graph.shell_thickness
    reference = sample.node_features["coords"]  # std 1, mean 0: already in mm
    env, cross, pairs, _ = metrics().surface_depths(pred, reference, thickness, graph)
    # Not through the shell, 0.6 mm from the mid-surface inside the 1 mm envelope
    # (below the 0.5 mm tolerance, so not penetrating either).
    assert cross[E] == 0.0 and env[E] == pytest.approx(0.4, abs=1e-5)
    assert pairs == 1.0
    summary = run(metrics(), slider=True)
    crossing = summary["surface"]["crossing"]["per_region"]["all"]
    envelope = summary["surface"]["envelope"]["per_region"]["all"]
    assert crossing["node_frames"] == 14 and crossing["spurious_count"] == 1
    assert envelope["spurious_count"] == 1
    assert crossing["spurious_depth_max_mm"] == pytest.approx(0.6, abs=1e-5)
    assert (summary["sidedness"], summary["metric_version"]) == ("symmetric", 2)
    assert summary["pairs_per_time"]["crossed_pairs_prediction"] == [1.0]
    assert summary["prediction"]["crossed_pairs"] == 1
    assert summary["target"]["crossed_pairs"] == 1  # C's genuine crossing is kept


def test_missing_contact_faces_disables_without_failing():
    m = metrics()
    sample, stats, pred, true = plate_scene()
    sample.graph.contact_faces = None
    m.accumulate(pred[:, None], true[:, None], sample, stats)
    summary = m.summary(torch.ones(1))
    assert summary["complete"] is False
    assert summary["prediction"]["crossed_pairs"] == 0
    assert summary["surface"]["envelope"]["per_region"]["all"]["spurious_count"] == 0
    assert "(INCOMPLETE)" in m.log_line(summary)


def test_log_line_and_settings():
    m = metrics({"wall_position_mm": 4.5, "depth_thresholds_mm": [0.5, 3]})
    summary = run(m)
    line = m.log_line(summary)
    assert "surface crossing: spurious" in line and "wall envelope" in line
    assert summary["depth_thresholds_mm"] == [0.5, 3.0]
    assert summary["region_bounds_mm"] == [2.0]
    assert list(
        summary["surface"]["envelope"]["per_region"]["all"]["spurious_above_mm"]
    ) == ["0.5", "3"]
    assert len(FAMILIES) == 4
    with pytest.raises(ValueError, match="tolerance"):
        metrics({"tolerance_mm": -1})
    with pytest.raises(ValueError, match="wall_axis"):
        metrics({"wall_axis": 3})
