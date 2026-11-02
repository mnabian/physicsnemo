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
"""DeFormer wrapper: a frozen teacher followed by the corrector (residual_model.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from mgn_corrector import MeshGraphNetCorrector  # noqa: E402
from residual_model import ResidualTimeConditional  # noqa: E402
from test_mgn_corrector import (  # noqa: E402
    WRAPPER_KEYS,
    _compose_corrector,
    _mesh_sample,
    _mgn_kwargs,
    _open,
)
from test_time_conditional_recipe import _compose, _excite, _tc_model  # noqa: E402
from train import trainable_parameters  # noqa: E402
from utils import build_muon_optimizer  # noqa: E402

import physicsnemo  # noqa: E402


def _teacher_file(tmp_path, name, seed):
    teacher = _tc_model()
    torch.manual_seed(seed)
    _excite(teacher)
    path = tmp_path / f"{name}.pt"
    torch.save(teacher.state_dict(), path, _use_new_zipfile_serialization=False)
    return teacher.eval(), str(path)


def _residual(teacher_weights, **overrides):
    torch.manual_seed(2)
    return ResidualTimeConditional(
        teacher=_tc_model(), teacher_weights=teacher_weights, **_mgn_kwargs(**overrides)
    )


def _window_kwargs(**overrides):
    return dict(teacher_window=True, functional_dim=3 + 1 + 8 + 3 + 6, **overrides)


def test_residual_model_starts_at_the_teacher_and_trains_the_student_only(tmp_path):
    teacher, path = _teacher_file(tmp_path, "teacher", seed=3)
    model = _residual(path)
    assert isinstance(model.student, MeshGraphNetCorrector)
    assert all(not p.requires_grad for p in model.teacher.parameters())
    model.train()
    assert model.student.training and not model.teacher.training
    sample, stats, _ = _mesh_sample(gap=0.5)
    with torch.no_grad():
        reference = teacher.predict_frame(sample, stats).prediction
        out = model(sample=sample, data_stats=stats, return_auxiliary=True)
    # Zero student decoder: prediction and provisional shape are the teacher's.
    torch.testing.assert_close(out.prediction, reference)
    torch.testing.assert_close(out.coarse_prediction, reference)
    # Evaluation: the teacher's trajectory, frame by frame.
    model.eval()
    with torch.no_grad():
        trajectory = model(sample=sample, data_stats=stats)
        torch.testing.assert_close(trajectory, teacher(sample=sample, data_stats=stats))
    assert trajectory.shape == (8, 4, 3)
    # Live student decoder moves the prediction; gradients reach the student only.
    model.train()
    _open(model.student)
    out = model(sample=sample, data_stats=stats, return_auxiliary=True)
    assert not torch.allclose(out.prediction, reference)
    out.prediction.square().sum().backward()
    assert all(p.grad is None for p in model.teacher.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.student.parameters()
    )
    assert model.rollout_steps == 4


def test_residual_model_final_clamp_and_projection_post_process_evaluation_only(
    tmp_path,
):
    """With the final flags on, the evaluation trajectory is projected and clamped
    frame by frame (no node past the wall plane); the training output is untouched
    and invalid settings are refused."""
    teacher, path = _teacher_file(tmp_path, "teacher_final", seed=5)
    plain = _residual(path)
    clamped = _residual(path, final_clamp_to_wall=True)
    final = _residual(path, final_clamp_to_wall=True, final_projection_iterations=2)
    sample, stats, _ = _mesh_sample(gap=0.5)
    plain.eval()
    with torch.no_grad():
        base = plain(sample=sample, data_stats=stats)
    # Wall plane at the median predicted x (physical == normalized in this fixture;
    # wall_axis 0, sign +1), so about half the node-frames lie past it.
    wall = float(base[..., 0].median())
    sample.graph.rigid_wall = torch.tensor([wall])
    clamped.eval()
    final.eval()
    with torch.no_grad():
        clamp_only = clamped(sample=sample, data_stats=stats)
        post = final(sample=sample, data_stats=stats)
    assert clamp_only.shape == post.shape == base.shape == (8, 4, 3)
    assert bool((base[..., 0] > wall + 1e-6).any())
    assert bool((clamp_only[..., 0] <= wall + 1e-6).all())
    assert bool((post[..., 0] <= wall + 1e-6).all())
    # The clamp alone touches only the wall axis, and only nodes past the plane.
    torch.testing.assert_close(clamp_only[..., 1:], base[..., 1:])
    torch.testing.assert_close(clamp_only[..., 0], base[..., 0].clamp(max=wall))
    assert bool(torch.isfinite(post).all())
    # The teacher's prediction served to the student is the uncorrected one:
    # training outputs of both models agree.
    plain.train()
    final.train()
    with torch.no_grad():
        torch.testing.assert_close(
            final(sample=sample, data_stats=stats),
            plain(sample=sample, data_stats=stats),
        )
    assert final.projection.enabled and final.projection.implementation == "torch"
    for bad in (
        dict(final_projection_iterations=-1),
        dict(final_projection_damping=0.0),
        dict(final_projection_tolerance_mm=-0.1),
        dict(wall_sign=0.0),
    ):
        with pytest.raises(ValueError):
            _residual(path, **bad)


def test_residual_model_round_trips_through_an_mdlus_file(tmp_path):
    teacher, path = _teacher_file(tmp_path, "teacher", seed=3)
    model = _residual(path)
    _open(model.student)
    file = tmp_path / "residual.mdlus"
    model.save(str(file))
    loaded = physicsnemo.Module.from_checkpoint(str(file))
    assert isinstance(loaded, ResidualTimeConditional)
    sample, stats, _ = _mesh_sample(gap=0.5)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(sample=sample, data_stats=stats),
            model(sample=sample, data_stats=stats),
        )


def test_research_checkpoints_and_options_load_into_the_release_class(tmp_path):
    """State dicts written with the teacher in a ``teachers`` list migrate, and
    the retired constructor options are accepted at their released values."""
    teacher, path = _teacher_file(tmp_path, "teacher", seed=3)
    model = _residual(path)
    _open(model.student)
    legacy = {
        ("teachers.0." + k[len("teacher.") :] if k.startswith("teacher.") else k): v
        for k, v in model.state_dict().items()
    }
    assert any(k.startswith("teachers.0.") for k in legacy)
    fresh = _residual(
        path,
        teacher_sampling="final",
        student_class="mgn_corrector.MeshGraphNetCorrector",
    )
    fresh.load_state_dict(legacy, strict=True)
    sample, stats, _ = _mesh_sample(gap=0.5)
    model.eval()
    fresh.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            fresh(sample=sample, data_stats=stats),
            model(sample=sample, data_stats=stats),
        )
    # The keys of the research recipes' attention-model chain are dropped.
    _residual(
        path,
        local_processor="knn_attention",
        local_attention_fused=True,
        checkpointing_ratio=1.0,
        activation_checkpointing_components=["blocks", "context"],
        num_pre_processor_checkpoint_segments=1,
        num_post_processor_checkpoint_segments=1,
        local_knn_k=32,
        local_neighborhood="euclidean",
    )
    assert _residual([path]).teacher_weights == path
    with pytest.raises(ValueError, match="research option"):
        _residual(path, teacher_sampling="uniform")
    with pytest.raises(ValueError, match="research option"):
        _residual(path, student_class="time_conditional.MeshGeoFLARETimeConditional")
    with pytest.raises(ValueError, match="one checkpoint"):
        _residual([path, path])
    with pytest.raises(ValueError, match="teacher_weights"):
        _residual("")


def test_optimizers_skip_the_frozen_teacher(tmp_path):
    _, path = _teacher_file(tmp_path, "teacher", seed=3)
    model = _residual(path)
    frozen = {id(p) for p in model.teacher.parameters()}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    covered = {id(p) for p in trainable_parameters(model)}
    assert covered == trainable and not (covered & frozen)
    if hasattr(torch.optim, "Muon"):
        cfg = OmegaConf.create({"training": {"start_lr": 1e-4, "muon_min_dim": 9}})
        optimizer = build_muon_optimizer(model, cfg)
        covered = {
            id(p)
            for part in getattr(optimizer, "optimizers", [optimizer])
            for g in part.param_groups
            for p in g["params"]
        }
        assert covered == trainable and not (covered & frozen)


def test_residual_wrapper_serves_the_window_with_reference_and_clamped_ends(tmp_path):
    teacher, path = _teacher_file(tmp_path, "teacher", seed=3)
    model = _residual(path, **_window_kwargs())
    assert model.teacher_window is True and model.student.teacher_window is True
    served = {}

    def grab(module, args, kwargs):
        served.update({k: v.clone() for k, v in kwargs.items() if torch.is_tensor(v)})

    hook = model.student.register_forward_pre_hook(grab, with_kwargs=True)
    try:
        for k in (0, 2, 3):  # first, middle and last frame of the 4-frame horizon
            sample, stats, _ = _mesh_sample(gap=0.5, time=k / 4)
            coords = sample.node_features["coords"]
            model.train()
            served.clear()
            with torch.no_grad():
                out = model(sample=sample, data_stats=stats)
                current = teacher.predict_frame(sample, stats).prediction
            torch.testing.assert_close(out, current)  # zero-initialised decoder
            window = served["teacher_window"]
            assert window.shape == (8, 2, 3)
            torch.testing.assert_close(served["teacher_displacement"], current - coords)

            def at(label):
                sample.node_features["time"] = torch.tensor(label)
                return teacher.predict_frame(sample, stats).prediction - coords

            with torch.no_grad():
                before = torch.zeros_like(coords) if k == 0 else at((k - 1) / 4)
                after = current - coords if k == 3 else at((k + 1) / 4)
            torch.testing.assert_close(window[:, 0], before)
            torch.testing.assert_close(window[:, 1], after)
        # Off-grid time labels are refused.
        sample, stats, _ = _mesh_sample(gap=0.5, time=0.3)
        with pytest.raises(ValueError, match="frame grid"):
            model(sample=sample, data_stats=stats)
        # Evaluation over the whole horizon: the last "next" repeats the last frame.
        sample, stats, _ = _mesh_sample(gap=0.5)
        coords = sample.node_features["coords"]
        model.eval()
        windows = []
        hook.remove()
        hook = model.student.register_forward_pre_hook(
            lambda module, args, kwargs: windows.append(
                kwargs["teacher_window"].clone()
            ),
            with_kwargs=True,
        )
        with torch.no_grad():
            trajectory = model(sample=sample, data_stats=stats)
            reference = teacher(sample=sample, data_stats=stats)
        torch.testing.assert_close(trajectory, reference)
        window_all = torch.stack(windows, dim=1)
        assert window_all.shape == (8, 4, 2, 3)
        field = reference - coords.unsqueeze(1)
        torch.testing.assert_close(window_all[:, 0, 0], torch.zeros_like(coords))
        torch.testing.assert_close(window_all[:, 1:, 0], field[:, :-1])
        torch.testing.assert_close(window_all[:, :-1, 1], field[:, 1:])
        torch.testing.assert_close(window_all[:, -1, 1], field[:, -1])
        # A shorter case (3 of the 4 frames) gets the teacher's fourth frame as its last "next".
        short = type(sample)(
            sample.node_features, torch.zeros(8, 3, 3), graph=sample.graph
        )
        windows.clear()
        with torch.no_grad():
            assert model(sample=short, data_stats=stats).shape == (8, 3, 3)
        torch.testing.assert_close(windows[-1][:, 1], field[:, 3])
    finally:
        hook.remove()
    # The window survives the checkpoint round trip.
    file = tmp_path / "window.mdlus"
    model.save(str(file))
    loaded = physicsnemo.Module.from_checkpoint(str(file))
    assert loaded.teacher_window is True and loaded.student.teacher_window is True
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(sample=sample, data_stats=stats),
            model(sample=sample, data_stats=stats),
        )


def test_residual_wrapper_trains_on_frame_pairs(tmp_path):
    """A two-frame training target (datapipe.frame_pairs) is served frame by frame:
    the student runs at k and k+1 with each frame's own window, the teacher frames
    the pair shares are computed once, and the predictions come back stacked."""
    teacher, path = _teacher_file(tmp_path, "teacher", seed=3)
    model = _residual(path, **_window_kwargs()).train()
    queried = []
    original = model.teacher.predict_frame

    def counting(sample, stats):
        queried.append(round(float(sample.node_features["time"]) * 4))
        return original(sample, stats)

    model.teacher.predict_frame = counting
    served = []
    hook = model.student.register_forward_pre_hook(
        lambda module, args, kwargs: served.append(
            (
                round(float(args[0].node_features["time"]) * 4),
                kwargs["teacher_displacement"].clone(),
                kwargs["teacher_window"].clone(),
            )
        ),
        with_kwargs=True,
    )
    try:
        # Middle pair (1, 2) of the 4-frame horizon: frames 0..3 once each.
        sample, stats, _ = _mesh_sample(gap=0.5, time=1 / 4)
        sample.node_target = torch.zeros(8, 2, 3)
        coords = sample.node_features["coords"]
        with torch.no_grad():
            out = model(sample=sample, data_stats=stats)
        assert out.shape == (8, 2, 3)
        assert sorted(queried) == [0, 1, 2, 3]
        assert sample.node_features["time"].item() == pytest.approx(0.25)
        assert [label for label, _, _ in served] == [1, 2]

        def at(label):
            sample.node_features["time"] = torch.tensor(label)
            return teacher.predict_frame(sample, stats).prediction

        with torch.no_grad():
            frames = {j: at(j / 4) for j in range(4)}
        sample.node_features["time"] = torch.tensor(1 / 4)
        torch.testing.assert_close(out[:, 0], frames[1])  # zero-initialised decoder
        torch.testing.assert_close(out[:, 1], frames[2])
        torch.testing.assert_close(served[0][1], frames[1] - coords)
        torch.testing.assert_close(served[0][2][:, 0], frames[0] - coords)
        torch.testing.assert_close(served[0][2][:, 1], frames[2] - coords)
        torch.testing.assert_close(served[1][1], frames[2] - coords)
        torch.testing.assert_close(served[1][2][:, 0], frames[1] - coords)
        torch.testing.assert_close(served[1][2][:, 1], frames[3] - coords)
        # Auxiliary output stacks the coarse prediction as well, and trains the
        # student only.
        _open(model.student)
        out = model(sample=sample, data_stats=stats, return_auxiliary=True)
        assert out.prediction.shape == (8, 2, 3) and out.coarse_prediction.shape == (
            8,
            2,
            3,
        )
        (out.prediction[:, 1] - out.prediction[:, 0]).square().sum().backward()
        assert all(p.grad is None for p in model.teacher.parameters())
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.student.parameters()
        )
        # Last pair (2, 3): the frame after the last trained label repeats it.
        queried.clear()
        served.clear()
        sample.node_features["time"] = torch.tensor(2 / 4)
        with torch.no_grad():
            model(sample=sample, data_stats=stats)
        assert sorted(queried) == [1, 2, 3]
        torch.testing.assert_close(served[1][2][:, 1], frames[3] - coords)
        # A pair cannot start at the last frame, and only pairs are accepted.
        sample.node_features["time"] = torch.tensor(3 / 4)
        with pytest.raises(ValueError, match="earlier frame"):
            model(sample=sample, data_stats=stats)
        sample.node_features["time"] = torch.tensor(1 / 4)
        sample.node_target = torch.zeros(8, 3, 3)
        with pytest.raises(ValueError, match="frame pair"):
            model(sample=sample, data_stats=stats)
        # A single-frame target still takes the single-frame path.
        sample.node_target = torch.zeros(8, 3)
        queried.clear()
        with torch.no_grad():
            assert model(sample=sample, data_stats=stats).shape == (8, 3)
        assert sorted(queried) == [0, 1, 2]
    finally:
        hook.remove()


def test_corrector_recipe_wraps_the_network_as_teacher():
    corrector = _compose_corrector()
    network = _compose("yaris_deformer")
    assert corrector.model._target_ == "residual_model.ResidualTimeConditional"
    assert OmegaConf.to_container(corrector.model.teacher) == OmegaConf.to_container(
        network.model
    )
    # Same optimizer, loss and data path as the network; its own budget and warm-up.
    for key in (
        "start_lr",
        "end_lr",
        "loss",
        "loss_space",
        "loss_scale_mm",
        "optimizer",
        "muon_min_dim",
        "amp_dtype",
        "num_time_steps",
        "num_training_samples",
        "num_validation_samples",
        "trainer_class",
        "seed",
    ):
        assert corrector.training[key] == network.training[key], key
    assert corrector.training.epochs == 30 and corrector.training.lr_warmup_epochs == 1
    assert (
        corrector.training.validation_freq == 2
        and corrector.training.save_checkpoint_freq == 1
    )
    assert corrector.training.coarse_loss_weight == 0.0
    assert corrector.datapipe.contact_surface and corrector.datapipe.mesh_edges
    assert corrector.datapipe.rigid_wall_position == 100.0
    assert corrector.datapipe.frame_pairs is False
    assert corrector.model.final_clamp_to_wall is True
    assert corrector.model.final_projection_iterations == 0
    student = OmegaConf.to_container(corrector.model, resolve=True)
    for key in WRAPPER_KEYS:
        student.pop(key)
    assert set(student) <= set(_mgn_kwargs()) | {
        "num_time_steps",
        "dt",
        "local_knn_k",
        "local_neighborhood",
        "aggregation",
        "num_processor_checkpoint_segments",
        "time_embedding",
        "contact_surface_max_pairs",
        "contact_activation_distance",
        "node_contact_radius",
        "contact_normal_epsilon",
        "contact_surface_material_fan",
        "contact_side_from_reference",
        "contact_side_symmetric",
        "contact_crossing_radius",
        "contact_clearance_approach_only",
        "contact_bidirectional",
        "teacher_window",
    }
