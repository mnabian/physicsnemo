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
"""Release contracts of the three final Yaris recipes.

The models the recipes build must have the parameter counts and the state-dict
layouts (tensor names and shapes) of the released checkpoints, so that those
checkpoints keep loading strictly. The fingerprints were taken from the
checkpoints of the DeFormer paper: the network at epoch 60, the corrector at
epoch 30 and the GeoTransolver baseline at epoch 60.
"""

import hashlib
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))
CONF = CRASH_DIR / "conf"

CONTRACTS = {
    # recipe: (trainable parameters, tensors in the state dict, fingerprint)
    "yaris_deformer": (9_672_505, 299, "e37e646a70ad1575"),
    "yaris_geotransolver": (6_731_545, 197, "fc9d6c19f878463b"),
    "yaris_deformer_corrector": (1_250_659, 454, "f30a32ed46b17776"),
}


def _compose(name, overrides=()):
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose(
            config_name=name,
            overrides=[
                "yaris.data_root=/fixtures/yaris",
                "yaris.split_manifest=/fixtures/splits.csv",
                "yaris.outlier_mask=/fixtures/outliers.npz",
                "yaris.work_dir=/fixtures/work",
                *overrides,
            ],
        )


def _fingerprint(state_dict):
    lines = sorted(f"{k}:{tuple(v.shape)}" for k, v in state_dict.items())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def _model(name, tmp_path):
    overrides = []
    if name == "yaris_deformer_corrector":
        teacher = instantiate(
            OmegaConf.create(
                OmegaConf.to_container(_compose("yaris_deformer").model, resolve=True)
            )
        )
        teacher_file = tmp_path / "teacher.mdlus"
        teacher.save(str(teacher_file))
        overrides.append(f"model.teacher_weights={teacher_file}")
    cfg = _compose(name, overrides)
    return instantiate(
        OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    )


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_final_recipe_builds_the_released_model(name, tmp_path):
    torch.manual_seed(0)
    model = _model(name, tmp_path)
    parameters, tensors, fingerprint = CONTRACTS[name]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == parameters
    state = model.state_dict()
    assert len(state) == tensors
    assert _fingerprint(state) == fingerprint
