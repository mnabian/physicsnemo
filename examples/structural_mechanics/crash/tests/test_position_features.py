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

"""Default-off XYZ features, closed-loop inputs, and checkpoint compatibility."""

import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))
from rollout import EPS, MeshGeoFLAREAutoregressive  # noqa: E402
from test_deformer_contact import make_model, make_sample  # noqa: E402


@pytest.mark.parametrize(
    "config_name",
    ["crash_deformer_autoregressive", "crash_deformer_contact_autoregressive"],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_position_feature_override_is_default_off_and_expands_encoder(
    config_name, enabled
):
    """One override changes the input width without changing the comparison recipe."""
    with initialize_config_dir(config_dir=str(CRASH_DIR / "conf"), version_base="1.3"):
        baseline = compose(config_name=config_name)
        config = compose(
            config_name=config_name,
            overrides=[f"model.include_position_features={str(enabled).lower()}"],
        )
    assert baseline.model.include_position_features is False
    assert config.model.functional_dim == 4
    assert config.model.node_input_mode == "velocity"
    expected = OmegaConf.to_container(baseline, resolve=False)
    actual = OmegaConf.to_container(config, resolve=False)
    expected["model"].pop("include_position_features")
    actual["model"].pop("include_position_features")
    assert actual == expected
    model = instantiate(
        config.model,
        n_hidden=16,
        mesh_hidden_dim=16,
        n_head=4,
        n_layers=1,
        slice_num=4,
        include_local_features=False,
    )
    assert model.preprocess[0].layers[0].in_features == (7 if enabled else 4)
    assert model.node_input_mode == ("position_velocity" if enabled else "velocity")
    assert model.include_position_features is enabled


def test_default_and_explicit_disabled_models_are_identical():
    """The default does not change tensors, initialization, or predictions."""
    torch.manual_seed(102)
    default = make_model().eval()
    torch.manual_seed(102)
    disabled = make_model(include_position_features=False).eval()
    assert default.state_dict().keys() == disabled.state_dict().keys()
    for key, value in default.state_dict().items():
        torch.testing.assert_close(value, disabled.state_dict()[key], atol=0, rtol=0)
    sample, stats = make_sample()
    with torch.no_grad():
        torch.testing.assert_close(
            default(sample, stats), disabled(sample, stats), atol=0, rtol=0
        )


@pytest.mark.parametrize("use_alias", [False, True])
def test_position_switch_matches_legacy_explicit_seven_channel_model(use_alias):
    """New opt-in and old explicit XYZ configuration have identical semantics."""
    width = dict(functional_dim=None, input_dim_nodes=4) if use_alias else {}
    torch.manual_seed(103)
    actual = make_model(include_position_features=True, **width).eval()
    torch.manual_seed(103)
    legacy = make_model(functional_dim=7, node_input_mode="position_velocity").eval()
    assert actual.state_dict().keys() == legacy.state_dict().keys()
    for key, value in actual.state_dict().items():
        torch.testing.assert_close(value, legacy.state_dict()[key], atol=0, rtol=0)
    sample, stats = make_sample()
    with torch.no_grad():
        torch.testing.assert_close(
            actual(sample, stats), legacy(sample, stats), atol=0, rtol=0
        )


def test_xyz_features_are_normalized_live_predictions_not_future_targets():
    """XYZ follows the predicted trajectory; velocity/thickness order is preserved."""
    model = make_model(include_position_features=True).eval()
    sample, stats = make_sample()
    stats["node"]["norm_vel_mean"].fill_(0.25)
    stats["node"]["norm_vel_std"].fill_(2.0)
    seen = []

    def capture(module, args):
        seen.append(args[0].detach().clone())

    handle = model.preprocess[0].register_forward_pre_hook(capture)
    with torch.no_grad():
        output = model(sample, stats)
    handle.remove()
    assert len(seen) == output.shape[1] == 3
    previous = sample.node_features["previous_coords"]
    current = sample.node_features["coords"]
    for step, features in enumerate(seen):
        assert features.shape == (6, 7)
        torch.testing.assert_close(features[:, :3], current)
        velocity = (current - previous) / model.dt
        expected_velocity = (velocity - stats["node"]["norm_vel_mean"]) / (
            stats["node"]["norm_vel_std"] + EPS
        )
        torch.testing.assert_close(features[:, 3:6], expected_velocity)
        torch.testing.assert_close(features[:, 6:], sample.node_features["features"])
        previous, current = current, output[:, step]
    sample.node_target.fill_(12345)
    with torch.no_grad():
        torch.testing.assert_close(output, model(sample, stats), atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["disabled", "enabled", "legacy", "legacy_position"])
def test_position_feature_checkpoint_round_trip(tmp_path, mode):
    """Restore the base width plus opt-in exactly once, including older metadata."""
    kwargs = (
        dict(functional_dim=7, node_input_mode="position_velocity")
        if mode == "legacy_position"
        else {}
    )
    model = make_model(include_position_features=mode == "enabled", **kwargs).eval()
    if mode.startswith("legacy"):
        model._args["__args__"].pop("include_position_features")
    assert model._args["__args__"]["functional_dim"] == (
        7 if mode == "legacy_position" else 4
    )
    checkpoint = tmp_path / "deformer.mdlus"
    model.save(str(checkpoint))
    restored = MeshGeoFLAREAutoregressive.from_checkpoint(str(checkpoint)).eval()
    assert restored.include_position_features is (mode == "enabled")
    assert restored.preprocess[0].layers[0].in_features == (
        7 if mode in ("enabled", "legacy_position") else 4
    )
    sample, stats = make_sample()
    with torch.no_grad():
        torch.testing.assert_close(
            model(sample, stats), restored(sample, stats), atol=0, rtol=0
        )


@pytest.mark.parametrize(
    "node_input_mode", ["position_velocity", "position_velocity_globals"]
)
def test_position_switch_rejects_double_opt_in(node_input_mode):
    """Existing explicit XYZ modes cannot accidentally receive three extra channels."""
    with pytest.raises(ValueError, match="selects position_velocity automatically"):
        make_model(include_position_features=True, node_input_mode=node_input_mode)


@pytest.mark.parametrize("width", [None, 2, True, 4.5])
def test_position_switch_requires_base_velocity_feature_width(width):
    """The width must count three velocity channels plus optional static features."""
    with pytest.raises(ValueError, match="count velocity plus static features"):
        make_model(include_position_features=True, functional_dim=width)


@pytest.mark.parametrize("invalid", [None, "false", 1])
def test_position_switch_requires_boolean(invalid):
    """Reject ambiguous truthy values rather than silently changing the model."""
    with pytest.raises(ValueError, match="must be a bool"):
        make_model(include_position_features=invalid)
