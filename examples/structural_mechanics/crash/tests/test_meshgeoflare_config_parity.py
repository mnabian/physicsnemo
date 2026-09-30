# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_bumper_meshgeoflare_uses_geoflare_recipe_with_flarepp_attention():
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        geoflare = compose(config_name="bumper_geoflare_oneshot")
        meshgeoflare = compose(config_name="bumper_meshgeoflare_oneshot")

    assert OmegaConf.to_container(
        meshgeoflare.training, resolve=False
    ) == OmegaConf.to_container(geoflare.training, resolve=False)
    assert OmegaConf.to_container(
        meshgeoflare.inference, resolve=False
    ) == OmegaConf.to_container(geoflare.inference, resolve=False)

    shared_backbone_keys = (
        "functional_dim",
        "out_dim",
        "geometry_dim",
        "global_dim",
        "n_hidden",
        "n_head",
        "slice_num",
        "n_layers",
        "use_te",
        "time_input",
        "include_local_features",
        "attn_scale",
        "num_time_steps",
    )
    for key in shared_backbone_keys:
        assert meshgeoflare.model[key] == geoflare.model[key]

    assert geoflare.training.epochs == 10_000
    assert meshgeoflare.training.epochs == 10_000
    assert geoflare.training.early_stopping_patience == 0
    assert meshgeoflare.training.early_stopping_patience == 0
    assert geoflare.model.attention_type == "GALE_FA"
    assert meshgeoflare.model.attention_type == "GALE_FPP"
    assert geoflare.model.attn_scale is None
    assert meshgeoflare.model.attn_scale is None
    assert meshgeoflare.model.mesh_context_fusion == "none"
    assert meshgeoflare.model.mesh_context_use_global is False


def test_bumper_meshgeoflare_context_ablation_configs_only_change_fusion():
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    names_and_expected = (
        ("bumper_meshgeoflare_oneshot", "none", False),
        ("bumper_meshgeoflarepp_post_geometry_oneshot", "post", False),
        ("bumper_meshgeoflarepp_pre_post_geometry_oneshot", "pre_post", False),
        (
            "bumper_meshgeoflarepp_pre_post_geometry_global_oneshot",
            "pre_post",
            True,
        ),
    )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        configs = [
            (compose(config_name=name), stage, use_global)
            for name, stage, use_global in names_and_expected
        ]

    baseline = configs[0][0]
    for config, stage, use_global in configs:
        assert OmegaConf.to_container(
            config.training, resolve=False
        ) == OmegaConf.to_container(baseline.training, resolve=False)
        assert OmegaConf.to_container(
            config.inference, resolve=False
        ) == OmegaConf.to_container(baseline.inference, resolve=False)
        assert config.model.mesh_context_fusion == stage
        assert config.model.mesh_context_use_global is use_global

        comparable_model = OmegaConf.to_container(config.model, resolve=False)
        baseline_model = OmegaConf.to_container(baseline.model, resolve=False)
        comparable_model.pop("mesh_context_fusion")
        comparable_model.pop("mesh_context_use_global")
        baseline_model.pop("mesh_context_fusion")
        baseline_model.pop("mesh_context_use_global")
        assert comparable_model == baseline_model


def test_bumper_meshgeoflare_teacher_forced_rollout_is_controlled_ablation():
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        baseline = compose(
            config_name="bumper_meshgeoflare_autoregressive_teacher_forced"
        )
        context = compose(
            config_name=(
                "bumper_meshgeoflarepp_pre_post_geometry_global_"
                "autoregressive_teacher_forced"
            )
        )

    assert OmegaConf.to_container(
        context.training, resolve=False
    ) == OmegaConf.to_container(baseline.training, resolve=False)
    assert OmegaConf.to_container(
        context.inference, resolve=False
    ) == OmegaConf.to_container(baseline.inference, resolve=False)
    assert OmegaConf.to_container(
        context.datapipe, resolve=False
    ) == OmegaConf.to_container(baseline.datapipe, resolve=False)

    baseline_model = OmegaConf.to_container(baseline.model, resolve=False)
    context_model = OmegaConf.to_container(context.model, resolve=False)
    assert baseline_model.pop("mesh_context_fusion") == "none"
    assert baseline_model.pop("mesh_context_use_global") is False
    assert context_model.pop("mesh_context_fusion") == "pre_post"
    assert context_model.pop("mesh_context_use_global") is True
    assert context_model == baseline_model

    assert baseline.training.optimizer == "muon"
    assert baseline.training.amp is True
    assert baseline.training.amp_dtype == "float16"
    assert baseline.training.seed == 42
    assert baseline.training.num_training_samples == 121
    assert baseline.training.num_validation_samples == 5
    assert baseline.training.epochs == 2_500
    assert baseline.training.validation_freq == 10
    assert baseline.training.save_checkpoint_freq == 10
    assert baseline.training.early_stopping_patience == 0

    assert baseline.datapipe.sample_type == "all_time_steps"
    assert baseline.model.functional_dim == 9
    assert baseline.model.out_dim == 5
    assert baseline.model.n_hidden == 256
    assert baseline.model.n_head == 8
    assert baseline.model.slice_num == 128
    assert baseline.model.n_layers == 6
    assert baseline.model.num_pre_processor_layers == 1
    assert baseline.model.num_post_processor_layers == 2
    assert baseline.model.teacher_forcing is True
    assert baseline.model.use_contact is False
    assert baseline.model.enable_contact is False
    assert baseline.model.enable_node_contact is False
    assert baseline.model.enable_cylinder_contact is False


def test_bumper_geoflare_teacher_forced_rollout_is_plain_point_cloud_flare():
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        geoflare = compose(config_name="bumper_geoflare_autoregressive_teacher_forced")
        meshgeoflare = compose(
            config_name="bumper_meshgeoflare_autoregressive_teacher_forced"
        )

    assert OmegaConf.to_container(
        geoflare.training, resolve=False
    ) == OmegaConf.to_container(meshgeoflare.training, resolve=False)
    assert OmegaConf.to_container(
        geoflare.inference, resolve=False
    ) == OmegaConf.to_container(meshgeoflare.inference, resolve=False)
    assert geoflare.datapipe._target_ == "datapipe.CrashPointCloudDataset"
    assert geoflare.datapipe.dynamic_targets == [
        "effective_plastic_strain",
        "stress_vm",
    ]
    assert geoflare.datapipe.global_features == [
        "velocity_x",
        "thickness_scale",
        "rwall_origin_y",
    ]
    assert geoflare.model._target_ == "rollout.GeoTransolverAutoregressive"
    assert geoflare.model.functional_dim == 3
    assert geoflare.model.out_dim == 5
    assert geoflare.model.attention_type == "GALE_FA"
    assert geoflare.model.teacher_forcing is True
    for mesh_only_key in (
        "input_dim_edges",
        "mesh_hidden_dim",
        "num_pre_processor_layers",
        "num_post_processor_layers",
        "mesh_context_fusion",
        "mesh_context_use_global",
        "use_contact",
        "enable_contact",
    ):
        assert mesh_only_key not in geoflare.model


def test_autoregressive_parity_probe_matrix_is_controlled():
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        mesh_flare_reference = compose(
            config_name="bumper_meshgeoflare_parity_flare_reference_ar"
        )
        mesh_flarepp_reference = compose(
            config_name="bumper_meshgeoflare_parity_flarepp_reference_ar"
        )
        geoflarepp = compose(
            config_name="bumper_geoflarepp_autoregressive_teacher_forced_probe"
        )

    for config in (
        mesh_flare_reference,
        mesh_flarepp_reference,
        geoflarepp,
    ):
        assert config.training.epochs == 200
        assert config.training.optimizer == "muon"
        assert config.training.seed == 42
        assert config.training.num_training_samples == 121
        assert config.training.num_validation_samples == 5
        assert config.model.functional_dim == 3
        assert config.model.out_dim == 5
        assert config.model.teacher_forcing is True

    shared_backbone_keys = (
        "functional_dim",
        "out_dim",
        "geometry_dim",
        "global_dim",
        "n_hidden",
        "n_head",
        "slice_num",
        "n_layers",
        "use_te",
        "time_input",
        "include_local_features",
        "attn_scale",
    )
    for key in shared_backbone_keys:
        assert mesh_flare_reference.model[key] == geoflarepp.model[key]

    assert mesh_flare_reference.model.attention_type == "GALE_FA"
    assert mesh_flarepp_reference.model.attention_type == "GALE_FPP"
    assert geoflarepp.model.attention_type == "GALE_FPP"
    assert mesh_flare_reference.model.node_input_mode == "velocity"
    assert "structural_edge_mode" not in mesh_flare_reference.model
    assert mesh_flare_reference.model.mesh_pre_residual_gate_init == 0.0
    assert mesh_flare_reference.model.mesh_post_residual_gate_init == 0.0


def test_latent_pre_mpnn_flare_autoregressive_recipe():
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name="bumper_meshgeoflare_adapter_flare_autoregressive")
        baseline = compose(config_name="bumper_geoflare_flare_autoregressive")

    assert config.model._target_ == "rollout.MeshGeoFLAREAutoregressive"
    assert baseline.model._target_ == "rollout.GeoTransolverAutoregressive"
    assert config.model.attention_type == "GALE_FA"
    assert baseline.model.attention_type == "GALE_FA"
    assert config.model.functional_dim == baseline.model.functional_dim == 3
    assert config.model.out_dim == 5
    assert config.model.node_input_mode == "velocity"
    assert config.model.mesh_context_fusion == "pre_post"
    assert config.model.mesh_context_use_global is True
    assert config.model.num_pre_processor_layers == 1
    assert config.model.num_post_processor_layers == 2
    assert config.model.mesh_pre_residual_gate_init == 0.0
    assert config.model.mesh_post_residual_gate_init == 0.0
    assert config.model.use_contact is False
    assert config.model.enable_contact is False

    for key in (
        "optimizer",
        "amp",
        "amp_dtype",
        "seed",
        "num_time_steps",
        "num_training_samples",
        "num_validation_samples",
        "epochs",
        "validation_freq",
        "save_checkpoint_freq",
        "early_stopping_patience",
        "acceleration_loss_weight",
    ):
        assert config.training[key] == baseline.training[key]
    assert config.training.epochs == baseline.training.epochs == 10_000
    assert config.training.optimizer == baseline.training.optimizer == "muon"
    assert config.training.save_checkpoint_freq == 100
    assert config.training.acceleration_loss_weight == 0.0
    assert "teacher_forcing_schedule" not in config.training
    assert "teacher_forcing_schedule" not in baseline.training
    assert config.model.teacher_forcing is False
    assert baseline.model.teacher_forcing is False


def test_full_car_deformer_uses_closed_loop_tbptt_without_early_stopping():
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name="gm_crash_deformer_autoregressive_tbptt")

    assert config.model._target_ == "rollout.MeshGeoFLAREAutoregressive"
    assert config.model.attention_type == "GALE_FA"
    assert config.model.functional_dim == 4
    assert config.model.out_dim == 3
    assert config.model.initial_velocity_mode == "previous_coords"
    assert config.model.rollout_steps_from_target is True
    assert config.model.teacher_forcing is False
    assert config.model.enable_contact is False
    assert config.datapipe.sample_type == "random_time_window"
    assert config.datapipe.initial_history_steps == 2
    assert config.datapipe.rollout_window_steps == 4
    assert config.training.epochs == 500
    assert config.training.start_lr == 2.0e-4
    assert config.training.end_lr == 1.0e-6
    assert config.training.save_checkpoint_freq == 10
    assert config.training.manual_gradient_allreduce is True
    assert config.training.early_stopping_patience == 0
    assert "teacher_forcing_schedule" not in config.training


def test_full_car_geoflare_is_matched_point_cloud_flare_baseline():
    config_dir = str(Path(__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        deformer = compose(config_name="gm_crash_deformer_autoregressive_tbptt")
        geoflare = compose(config_name="gm_crash_geoflare_autoregressive_tbptt")

    for key in (
        "optimizer",
        "amp",
        "amp_dtype",
        "manual_gradient_allreduce",
        "seed",
        "num_time_steps",
        "num_training_samples",
        "num_validation_samples",
        "epochs",
        "start_lr",
        "end_lr",
        "validation_freq",
        "save_checkpoint_freq",
        "early_stopping_patience",
        "acceleration_loss_weight",
    ):
        assert geoflare.training[key] == deformer.training[key]

    assert geoflare.datapipe._target_ == "datapipe.CrashPointCloudDataset"
    assert geoflare.datapipe.sample_type == "random_time_window"
    assert geoflare.datapipe.initial_history_steps == 2
    assert geoflare.datapipe.rollout_window_steps == 4
    assert geoflare.model._target_ == "rollout.GeoTransolverAutoregressive"
    for key in (
        "functional_dim",
        "out_dim",
        "geometry_dim",
        "global_dim",
        "n_hidden",
        "n_head",
        "slice_num",
        "n_layers",
        "use_te",
        "time_input",
        "include_local_features",
        "attention_type",
        "attn_scale",
        "initial_velocity_mode",
        "rollout_steps_from_target",
        "checkpoint_rollout",
        "teacher_forcing",
    ):
        assert geoflare.model[key] == deformer.model[key]
    for mesh_only_key in (
        "input_dim_edges",
        "mesh_hidden_dim",
        "num_pre_processor_layers",
        "num_post_processor_layers",
        "mesh_context_fusion",
        "mesh_context_use_global",
        "use_contact",
        "enable_contact",
    ):
        assert mesh_only_key not in geoflare.model
