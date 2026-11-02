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
"""Position-loss options (physical scale, relative L2, increment loss)."""

import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))
CONF = CRASH_DIR / "conf"

from losses import PositionLoss, increment_loss  # noqa: E402


def test_default_position_loss_is_the_plain_mse():
    torch.manual_seed(0)
    pred, target = torch.randn(50, 7, 3), torch.randn(50, 7, 3)
    loss = PositionLoss()
    torch.testing.assert_close(
        loss(pred, target), torch.nn.functional.mse_loss(pred, target)
    )
    assert loss.per_frame(pred, target).shape == (7,)
    assert loss.per_frame(pred[:, 0], target[:, 0]).shape == (1,)


def test_physical_space_rescales_each_axis_to_millimetres():
    torch.manual_seed(1)
    pred, target = torch.randn(40, 3), torch.randn(40, 3)
    std = torch.tensor([1000.0, 500.0, 250.0])
    physical = PositionLoss("mse", "physical", scale_mm=10.0, pos_std=std)
    expected = torch.nn.functional.mse_loss(
        (pred - target) * std / 10.0, torch.zeros(40, 3)
    )
    torch.testing.assert_close(physical(pred, target), expected)
    # With equal stds the physical loss is the normalized one up to the scale.
    same = PositionLoss("mse", "physical", scale_mm=4.0, pos_std=torch.full((3,), 4.0))
    torch.testing.assert_close(same(pred, target), PositionLoss()(pred, target))
    with pytest.raises(ValueError, match="physical"):
        PositionLoss("relative_l2", "normalized")
    with pytest.raises(ValueError, match="position std"):
        PositionLoss("mse", "physical")
    with pytest.raises(ValueError, match="training.loss must"):
        PositionLoss("cauchy")


def test_relative_l2_is_the_norm_ratio_of_error_to_displacement():
    """Per frame, ||error|| / ||target - reference|| over all nodes and axes in
    millimetres; scale-free; a trajectory gives one value per frame; without a
    reference the targets count as displacements; normalized space is refused."""
    torch.manual_seed(3)
    std = torch.tensor([2.0, 4.0, 8.0])
    loss = PositionLoss("relative_l2", "physical", scale_mm=10.0, pos_std=std)
    reference = torch.randn(30, 3)
    motion = torch.randn(30, 3)
    error = 0.1 * torch.randn(30, 3)
    target = reference + motion
    prediction = target + error
    expected = ((error * std).norm() / (motion * std).norm()).item()
    assert loss(prediction, target, reference=reference).item() == pytest.approx(
        expected, rel=1e-5
    )
    assert loss.per_frame(prediction, target, reference=reference).shape == (1,)
    # Scale-free: the common scale cancels, the per-axis std does not.
    other = PositionLoss("relative_l2", "physical", scale_mm=1.0, pos_std=std)
    assert other(prediction, target, reference=reference).item() == pytest.approx(
        expected, rel=1e-5
    )
    # Exact prediction: zero. Doubling the error doubles the loss (a norm, not a square).
    assert loss(target, target, reference=reference).item() == pytest.approx(0.0)
    assert loss(
        target + 2 * error, target, reference=reference
    ).item() == pytest.approx(2 * expected, rel=1e-5)
    # Trajectories: one ratio per frame, each over its own displacement.
    traj_target = torch.stack([reference + motion, reference + 3 * motion], dim=1)
    traj_pred = traj_target + torch.stack([error, error], dim=1)
    per_frame = loss.per_frame(traj_pred, traj_target, reference=reference)
    assert per_frame.shape == (2,)
    assert per_frame[0].item() == pytest.approx(expected, rel=1e-5)
    assert per_frame[1].item() == pytest.approx(expected / 3, rel=1e-5)
    torch.testing.assert_close(
        loss(traj_pred, traj_target, reference=reference), per_frame.mean()
    )
    # No reference: the targets are displacements already.
    assert loss(motion + error, motion).item() == pytest.approx(expected, rel=1e-5)
    with pytest.raises(ValueError, match="physical"):
        PositionLoss("relative_l2", "normalized")
    with pytest.raises(ValueError, match="elementwise"):
        loss.elementwise(prediction, target)


def _compose(name):
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


def test_increment_loss_scores_the_change_between_the_two_frames():
    """The increment term is the position loss of the predicted minus the true
    increment, in the main loss's millimetre scale; a shared offset of both
    frames costs nothing and a single frame or a trajectory is refused."""
    torch.manual_seed(0)
    std = torch.tensor([2.0, 4.0, 8.0])
    loss = PositionLoss(kind="mse", space="physical", scale_mm=10.0, pos_std=std)
    target = torch.randn(6, 2, 3)
    offset = torch.randn(6, 1, 3)
    assert increment_loss(loss, target + offset, target).item() == pytest.approx(0.0)
    prediction = target.clone()
    prediction[:, 1] += torch.tensor([1.0, 0.0, 0.0])  # 2 mm more motion along x
    expected = ((1.0 * std[0] / 10.0) ** 2) / 3  # one channel of three
    assert increment_loss(loss, prediction, target).item() == pytest.approx(
        expected.item()
    )
    for bad in (torch.zeros(6, 3), torch.zeros(6, 3, 3)):
        with pytest.raises(ValueError, match="frame pairs"):
            increment_loss(loss, bad, bad)


def test_final_recipes_use_the_physical_mse_loss():
    network = _compose("yaris_deformer")
    baseline = _compose("yaris_geotransolver")
    corrector = _compose("yaris_deformer_corrector")
    for cfg in (network, baseline, corrector):
        assert cfg.training.loss == "mse" and cfg.training.loss_space == "physical"
        assert cfg.training.loss_scale_mm == 10.0
    assert corrector.training.get("increment_loss_weight", 0.0) == 0.0
    with pytest.raises(ValueError, match="training.loss"):
        PositionLoss(kind="huber", space="physical", pos_std=torch.ones(3))
    with pytest.raises(ValueError, match="loss_space=physical"):
        PositionLoss(kind="relative_l2", space="normalized")
