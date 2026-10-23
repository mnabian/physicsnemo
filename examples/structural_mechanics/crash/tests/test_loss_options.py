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
"""Position-loss options (physical scale, Huber, L1, frame weighting, edge term)."""

import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))
CONF = CRASH_DIR / "conf"

from datapipe import SimSample  # noqa: E402
from losses import (  # noqa: E402
    FrameWeighter,
    PositionLoss,
    edge_consistency_loss,
    frame_indices,
)

LOSS_KEYS = {
    "loss",
    "loss_space",
    "loss_scale_mm",
    "huber_delta_mm",
    "frame_weighting",
    "frame_weighting_ema",
    "edge_loss_weight",
}


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
        PositionLoss("huber", "normalized")
    with pytest.raises(ValueError, match="position std"):
        PositionLoss("mse", "physical")
    with pytest.raises(ValueError, match="training.loss must"):
        PositionLoss("cauchy")


def test_huber_and_l1_in_millimetres():
    std = torch.ones(3) * 10.0  # 1 normalized unit = 10 mm, scale 10 mm -> identity
    small = torch.full((5, 3), 0.1)  # 1 mm errors
    large = torch.full((5, 3), 2.0)  # 20 mm errors
    zero = torch.zeros(5, 3)
    huber = PositionLoss(
        "huber", "physical", scale_mm=10.0, huber_delta_mm=3.0, pos_std=std
    )
    l1 = PositionLoss("l1", "physical", scale_mm=10.0, pos_std=std)
    mse = PositionLoss("mse", "physical", scale_mm=10.0, pos_std=std)
    # Below delta Huber is 0.5 * squared error; above it is linear in the error.
    torch.testing.assert_close(huber(small, zero), 0.5 * mse(small, zero))
    delta = 0.3
    torch.testing.assert_close(
        huber(large, zero), torch.tensor(delta * (2.0 - 0.5 * delta))
    )
    torch.testing.assert_close(l1(large, zero), torch.tensor(2.0))
    assert huber(large, zero) < mse(large, zero)


def test_frame_weighter_equalises_frames_and_averages_to_one():
    weighter = FrameWeighter(num_frames=4, momentum=0.5)
    frames = torch.arange(4)
    losses = torch.tensor([1.0, 2.0, 4.0, 8.0])
    first = weighter.apply(frames, losses)
    torch.testing.assert_close(first, losses.mean())  # unweighted until frames are seen
    second = weighter.apply(frames, losses)
    w = weighter.weights(frames)
    assert torch.allclose(w.float(), losses.mean() / losses, rtol=1e-5)
    torch.testing.assert_close(second.double(), (w * losses.double()).mean())
    # A frame never seen keeps weight one; state round-trips.
    partial = FrameWeighter(num_frames=3)
    partial.apply(torch.tensor([1]), torch.tensor([5.0]))
    assert partial.weights(torch.tensor([0, 1])).tolist() == [1.0, 1.0]
    clone = FrameWeighter(num_frames=3)
    clone.load_state_dict(partial.state_dict())
    assert torch.equal(clone.ema, partial.ema) and torch.equal(clone.seen, partial.seen)
    with pytest.raises(ValueError):
        FrameWeighter(0)


def test_frame_indices_for_trajectories_and_single_frames():
    assert frame_indices(None, torch.zeros(3, 6, 3), 6).tolist() == list(range(6))
    sample = SimSample(
        {"coords": torch.zeros(3, 3), "time": torch.tensor(0.5)}, torch.zeros(3, 3)
    )
    assert frame_indices(sample, sample.node_target, 6).tolist() == [3]
    sample.node_features["time"] = torch.tensor(5 / 6)
    assert frame_indices(sample, sample.node_target, 6).tolist() == [5]
    with pytest.raises(ValueError, match="time"):
        frame_indices(
            SimSample({"coords": torch.zeros(3, 3)}, torch.zeros(3, 3)),
            torch.zeros(3, 3),
            6,
        )


def test_edge_consistency_loss_ignores_rigid_translation_and_masked_slots():
    torch.manual_seed(2)
    n, k = 12, 4
    target = torch.randn(n, 3)
    index = torch.randint(0, n, (n, k))
    mask = torch.ones(n, k, dtype=torch.bool)
    std = torch.tensor([10.0, 10.0, 10.0])
    # Rigid translation of the whole prediction: no relative error.
    shifted = target + torch.tensor([0.3, -0.2, 0.1])
    assert edge_consistency_loss(
        shifted, target, index, mask, std
    ).item() == pytest.approx(0.0)
    # A single displaced node is caught through its slots and its neighbours' slots.
    distorted = target.clone()
    distorted[0] += 1.0
    loss = edge_consistency_loss(distorted, target, index, mask, std)
    assert loss > 0
    # Masking every slot that touches node 0 removes the signal.
    touched = (index == 0) | (torch.arange(n).unsqueeze(1) == 0)
    assert edge_consistency_loss(
        distorted, target, index, ~touched, std
    ).item() == pytest.approx(0.0)
    # Trajectory inputs average over the time axis with the same slot count.
    traj_pred = distorted.unsqueeze(1).repeat(1, 5, 1)
    traj_true = target.unsqueeze(1).repeat(1, 5, 1)
    torch.testing.assert_close(
        edge_consistency_loss(traj_pred, traj_true, index, mask, std), loss
    )
    with pytest.raises(ValueError, match="knn_index"):
        edge_consistency_loss(target, target, index, mask[:, :2], std)


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


@pytest.mark.parametrize(
    "suffix, expected",
    [
        ("physical", {"loss": "mse", "loss_space": "physical", "loss_scale_mm": 10.0}),
        (
            "huber",
            {
                "loss": "huber",
                "loss_space": "physical",
                "loss_scale_mm": 10.0,
                "huber_delta_mm": 3.0,
            },
        ),
        ("l1", {"loss": "l1", "loss_space": "physical", "loss_scale_mm": 10.0}),
        (
            "frameweight",
            {
                "loss": "mse",
                "loss_space": "physical",
                "loss_scale_mm": 10.0,
                "frame_weighting": "ema",
                "frame_weighting_ema": 0.99,
            },
        ),
        (
            "edge",
            {
                "loss": "mse",
                "loss_space": "physical",
                "loss_scale_mm": 10.0,
                "edge_loss_weight": 1.0,
            },
        ),
    ],
)
def test_loss_sweep_recipes_change_only_the_loss(suffix, expected):
    base = _compose("crash_deformer_knn_yaris_time_conditional")
    variant = _compose(f"crash_deformer_knn_yaris_tc_loss_{suffix}")
    added = set(variant.training) - set(base.training)
    assert added == set(expected) and added <= LOSS_KEYS
    for key, value in expected.items():
        assert variant.training[key] == value, key
    for key in base.training:
        assert base.training[key] == variant.training[key], key
    for group in ("model", "datapipe", "reader", "yaris"):
        assert OmegaConf.to_container(
            variant[group], resolve=True
        ) == OmegaConf.to_container(base[group], resolve=True), group
    assert variant.experiment_name != base.experiment_name
