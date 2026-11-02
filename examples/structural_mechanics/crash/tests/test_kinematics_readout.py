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
"""Per-part rigid fit, temporal smoothness and kinematic readouts (kinematics_readout.py)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))

from kinematics_readout import (  # noqa: E402
    G_MM_S2,
    KinematicsReadout,
    fit_part_rigid_motion,
    part_means,
    rigid_displacement,
    rotation_matrix,
)


class _Sample:
    def __init__(self, name):
        self.yaris_case_name = name


def _scene(seed=0):
    torch.manual_seed(seed)
    sizes = torch.tensor([30, 40, 50])
    part_index = torch.repeat_interleave(torch.arange(3), sizes)
    reference = torch.randn(int(sizes.sum()), 3, dtype=torch.float64) * 100.0
    reference[:30, 0] -= 3000.0  # rear part
    reference[30:70, 0] -= 1500.0  # middle part
    reference[70:, 0] += 0.0  # front part
    return part_index, reference


def _truth(reference, part_index, steps, dt, deceleration_g=20.0):
    """Every part translates along x with constant deceleration: x(t) = v t - a t^2 / 2."""
    centroid = part_means(reference, part_index, 3)
    frames = []
    for k in range(steps):
        t = (k + 1) * dt
        translation = torch.zeros(3, 3, dtype=torch.float64)
        translation[:, 0] = (
            -13000.0 * t + 0.5 * deceleration_g * G_MM_S2 * t * t
        )  # 13 m/s then constant braking
        rotation = rotation_matrix(torch.zeros(3, 3, dtype=torch.float64))
        frames.append(
            reference
            + rigid_displacement(reference, part_index, rotation, translation, centroid)
        )
    return torch.stack(frames, 1)  # [N, T, 3]


def test_readout_recovers_pulse_and_flags_temporal_noise(tmp_path):
    part_index, reference = _scene()
    dt, steps = 0.005, 8
    truth = _truth(reference, part_index, steps, dt)
    stats = {
        "node": {
            "pos_mean": torch.zeros(3, dtype=torch.float64),
            "pos_std": torch.ones(3, dtype=torch.float64),
        }
    }
    # Exact prediction: no error anywhere, true pulse close to 20 g on the x axis.
    exact = KinematicsReadout(
        part_index, np.array([1, 2, 3]), reference, dt, [-2000.0, -1000.0], steps, "cpu"
    )
    exact.accumulate(truth, truth, _Sample("A"), stats)
    s = exact.summary()
    assert s["regions"] == [
        "x_-inf_to_-2000",
        "x_-2000_to_-1000",
        "x_-1000_to_inf",
        "all",
    ]
    assert s["smoothness"]["all"]["acceleration_error_g"] < 1e-6
    assert s["parts"]["all"]["part_translation_error_mm"] < 1e-6
    assert s["parts"]["all"]["pulse_error_g"] < 1e-6
    assert s["vehicle_centroid_error_mm"] < 1e-6
    # The deceleration pulse read from the fitted part translations is the 20 g applied
    # from the first entry on (the reference frame precedes frame 0); the last frame has
    # no successor, so its entry is null.
    per_time = s["parts"]["all"]["true_pulse_g_per_time"]
    assert all(abs(v - 20.0) < 0.05 for v in per_time[: steps - 1])
    assert per_time[steps - 1] is None
    assert abs(s["parts"]["all"]["true_pulse_g"] - 20.0) < 0.05
    # Constant velocity is no pulse: the launch speed must not leak into entry 0.
    constant = _truth(reference, part_index, steps, dt, deceleration_g=0.0)
    coasting = KinematicsReadout(
        part_index, np.array([1, 2, 3]), reference, dt, [-2000.0, -1000.0], steps, "cpu"
    )
    coasting.accumulate(constant, constant, _Sample("B"), stats)
    cs = coasting.summary()
    assert cs["parts"]["all"]["true_pulse_g"] < 1e-6
    assert all(
        v < 1e-6 for v in cs["parts"]["all"]["true_pulse_g_per_time"][: steps - 1]
    )
    assert cs["smoothness"]["all"]["true_acceleration_g"] < 1e-6
    assert (
        abs(
            s["smoothness"]["x_-inf_to_-2000"]["true_acceleration_g_per_time"][3] - 20.0
        )
        < 0.05
    )
    # A prediction with a smooth bias and one with frame-to-frame jitter of the same
    # position error: the pooled position error cannot separate them, the readout can.
    smooth = truth + torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)
    jitter = truth.clone()
    jitter[:, 1::2, 0] += 3.0 * math.sqrt(2)
    smooth_out = KinematicsReadout(
        part_index, np.array([1, 2, 3]), reference, dt, [-2000.0, -1000.0], steps, "cpu"
    )
    smooth_out.accumulate(smooth, truth, _Sample("A"), stats)
    jitter_out = KinematicsReadout(
        part_index, np.array([1, 2, 3]), reference, dt, [-2000.0, -1000.0], steps, "cpu"
    )
    jitter_out.accumulate(jitter, truth, _Sample("A"), stats)
    ss, js = smooth_out.summary(), jitter_out.summary()
    assert abs(ss["parts"]["all"]["part_translation_error_mm"] - 3.0) < 1e-6
    assert (
        abs(js["parts"]["all"]["part_translation_error_mm"] - 3.0) < 1e-6
    )  # same RMS position error
    # A constant offset has no acceleration after the first step; the first step
    # sees the jump from the shared (exact) reference frame, which is a real
    # start-up transient of such a prediction.
    smooth_acc = ss["smoothness"]["all"]["acceleration_error_g_per_time"]
    assert all(v < 1e-6 for v in smooth_acc[1 : steps - 1])
    assert smooth_acc[0] > 10.0 and smooth_acc[-1] is None  # no step beyond the horizon
    # Alternating 4.2 mm every 5 ms is a second difference of 8.5 mm per frame pair,
    # about 35 g: tens of g of spurious acceleration from a 3 mm RMS position error.
    assert 20.0 < js["smoothness"]["all"]["acceleration_error_g"] < 60.0
    assert (
        js["smoothness"]["all"]["jitter_ratio"]
        > 1.5
        > ss["smoothness"]["all"]["jitter_ratio"]
    )
    assert (
        20.0 < js["parts"]["all"]["pulse_error_g"]
        and ss["parts"]["all"]["pulse_error_g"] < 20.0
    )
    assert (
        js["smoothness"]["all"]["acceleration_error_g"]
        > 5 * ss["smoothness"]["all"]["acceleration_error_g"]
    )
    # Per-region bookkeeping: node counts per step add up to the mesh.
    assert exact.acc[0, 3, 3] == float(reference.shape[0])
    assert exact.acc[0, 0, 3] == 30.0 and exact.acc[0, 2, 3] == 50.0
    # Saved trajectories: padded past a shorter case, named and ordered.
    short = truth[:, :5]
    exact.accumulate(short, short, _Sample("B"), stats)
    exact.save(tmp_path / "kin.npz")
    saved = np.load(tmp_path / "kin.npz")
    assert saved["cases"].tolist() == ["A", "B"] and saved[
        "predicted_translation"
    ].shape == (2, steps, 3, 3)
    assert (
        np.isnan(saved["predicted_translation"][1, 5:]).all()
        and np.isfinite(saved["true_translation"][0]).all()
    )
    assert exact.log_line(exact.summary()).startswith("accel err")


def test_rotation_matrix_is_proper_and_finite_at_zero():
    omega = torch.tensor(
        [[0.0, 0.0, 0.0], [0.3, -0.2, 0.1], [0.0, 0.0, math.pi / 2]],
        dtype=torch.float64,
    )
    rotation = rotation_matrix(omega)
    eye = torch.eye(3, dtype=torch.float64)
    torch.testing.assert_close(rotation[0], eye)
    torch.testing.assert_close(
        rotation @ rotation.transpose(-1, -2), eye.expand(3, 3, 3)
    )
    torch.testing.assert_close(
        torch.linalg.det(rotation), torch.ones(3, dtype=torch.float64)
    )
    torch.testing.assert_close(
        rotation[2] @ torch.tensor([1.0, 0, 0], dtype=torch.float64),
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
    )
    tiny = torch.full((1, 3), 1e-6, dtype=torch.float64, requires_grad=True)
    rotation_matrix(tiny).sum().backward()
    assert torch.isfinite(tiny.grad).all()


def test_fit_recovers_per_part_rigid_motion_including_tiny_parts():
    torch.manual_seed(3)
    sizes = torch.tensor([40, 25, 1, 2])  # a one-node and a two-node part
    part_index = torch.repeat_interleave(torch.arange(4), sizes)
    reference = torch.randn(int(sizes.sum()), 3, dtype=torch.float64) * 100.0
    omega = torch.randn(4, 3, dtype=torch.float64) * 0.4
    translation = torch.randn(4, 3, dtype=torch.float64) * 50.0
    rotation = rotation_matrix(omega)
    centroid = part_means(reference, part_index, 4)
    target = reference + rigid_displacement(
        reference, part_index, rotation, translation, centroid
    )
    fitted_rotation, fitted_translation, fitted_centroid = fit_part_rigid_motion(
        reference, target, part_index, 4
    )
    torch.testing.assert_close(fitted_centroid, centroid)
    torch.testing.assert_close(fitted_translation, translation, atol=1e-8, rtol=0)
    torch.testing.assert_close(fitted_rotation[:2], rotation[:2], atol=1e-8, rtol=0)
    torch.testing.assert_close(
        torch.linalg.det(fitted_rotation), torch.ones(4, dtype=torch.float64)
    )
    # Every part's nodes are mapped exactly, the tiny parts included.
    refit = reference + rigid_displacement(
        reference, part_index, fitted_rotation, fitted_translation, fitted_centroid
    )
    torch.testing.assert_close(refit, target, atol=1e-7, rtol=0)
    with pytest.raises(ValueError, match="part_index"):
        fit_part_rigid_motion(reference, target, part_index[:-1], 4)
