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
"""GeoTransolver/FLARE time-conditional baseline on the Yaris datapipe (CPU)."""

import math
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_yaris_full_recipe import CONF, _compose_public, yaris_root  # noqa: E402,F401

RECIPE = "yaris_geotransolver"
CONTROL = "yaris_deformer"


def _compose_tiny(root, work_dir, extra=()):
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose(
            config_name=RECIPE,
            overrides=[
                f"yaris.data_root={root}",
                f"yaris.split_manifest={root / 'splits.csv'}",
                f"yaris.outlier_mask={root / 'outliers.npz'}",
                f"yaris.work_dir={work_dir}",
                "training.num_training_samples=3",
                "training.num_validation_samples=2",
                "training.num_time_steps=7",
                "training.amp=false",
                "training.num_dataloader_workers=0",
                "model.n_hidden=16",
                "model.n_head=4",
                "model.slice_num=4",
                "model.n_layers=1",
                "model.functional_dim=12",
                "model.global_dim=8",
                "model.time_dim=8",
                "model.time_fourier_bands=3",
                "datapipe.local_knn_k=4",
                *extra,
            ],
        )


def test_recipe_matches_the_deformer_control_protocol():
    baseline = _compose_public(RECIPE)
    control = _compose_public(CONTROL)
    assert baseline.model._target_ == "rollout.GeoTransolverTimeConditional"
    for key in (
        "epochs",
        "start_lr",
        "end_lr",
        "optimizer",
        "amp",
        "amp_dtype",
        "seed",
        "sampler_seed",
        "reproducible_sampling",
        "muon_min_dim",
        "num_time_steps",
        "num_training_samples",
        "num_validation_samples",
        "validation_freq",
        "save_checkpoint_freq",
        "loss",
        "loss_space",
        "loss_scale_mm",
        "trainer_class",
    ):
        assert baseline.training[key] == control.training[key], key
    assert baseline.training.coarse_loss_weight == 0.0
    # Same data path: reader and datapipe resolve identically (same kNN cache key).
    for group in ("reader", "datapipe"):
        assert OmegaConf.to_container(
            getattr(baseline, group), resolve=True
        ) == OmegaConf.to_container(getattr(control, group), resolve=True), group
    # Same backbone size and time code as the DeFormer, with its memory policy.
    for key in (
        "functional_dim",
        "global_dim",
        "n_hidden",
        "n_head",
        "slice_num",
        "n_layers",
        "attention_type",
        "time_embedding",
        "time_fourier_bands",
        "time_dim",
        "time_as_global",
        "dt",
        "activation_checkpointing",
        "checkpointing_ratio",
    ):
        assert baseline.model[key] == control.model[key], key
    assert set(baseline.model.activation_checkpointing_components) == {
        "blocks",
        "context",
    }
    assert "local_knn_k" not in baseline.model


def test_geoflare_yaris_trainer_end_to_end(yaris_root, tmp_path):  # noqa: F811
    """Train one step and run the exact validation over variable horizons."""
    from yaris_full_trainer import YarisFullTrainer

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

    cfg = _compose_tiny(yaris_root, tmp_path / "work")
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    logger0 = RankZeroLoggingWrapper(
        PythonLogger("geoflare-yaris-test"), DistributedManager()
    )
    trainer = YarisFullTrainer(cfg, logger0)
    assert type(trainer.model).__name__ == "GeoTransolverTimeConditional"
    assert trainer.model.dt == pytest.approx(0.005)
    sample = next(iter(trainer.dataloader))[0]
    loss = trainer.train(sample, 0)
    assert torch.isfinite(loss)
    assert set(trainer.last_loss_components) == {"trajectory"}
    stats = trainer.validate(0)
    assert torch.isfinite(stats["MSE"])
    horizons = [
        len(
            trainer.validation_dataset.reader.frames(
                trainer.validation_dataset.reader.open(path)
            )
        )
        - 1
        for path in trainer.validation_dataset.paths
    ]
    assert len(set(horizons)) > 1, "the fixture must have cases of different horizons"
    assert stats["MSE_w_time"].numel() == max(horizons)
    # Eval mode follows each case's horizon and restores the time label.
    trainer.model.eval()
    for batch in trainer.val_dataloader:
        case = batch[0].to(trainer.dist.device)
        before = case.node_features.get("time")
        with torch.no_grad():
            out = trainer.model(sample=case, data_stats=trainer.data_stats)
        assert out.shape == case.node_target.shape
        assert case.node_features.get("time") is before
        assert math.isfinite(float(out.abs().sum()))
