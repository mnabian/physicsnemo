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
"""RNG capture/restore across restarts and the deterministic-execution policy (training_state.py)."""

import copy
import io
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))

from training_state import (  # noqa: E402
    capture_rng_state,
    configure_deterministic_training,
    restore_rank_rng,
)


def test_stochastic_optimizer_restart_replays_exactly():
    """A restart from a snapshot replays the same losses and weights."""
    torch.manual_seed(83)
    random.seed(83)
    np.random.seed(83)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 1)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(94)

    def step():
        x = torch.randn(8, 3, generator=generator) + random.random() + np.random.rand()
        optimizer.zero_grad()
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()

    step()
    snapshot = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng_state({"loader": generator}),
        },
        snapshot,
    )
    losses = [step() for _ in range(5)]
    reference = copy.deepcopy(model.state_dict())
    torch.randn(1000)
    snapshot.seek(0)
    saved = torch.load(snapshot, weights_only=True)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    restore_rank_rng([saved["rng"]], 0, 1, {"loader": generator})
    for expected in losses:
        assert torch.equal(step(), expected)
    for name, value in model.state_dict().items():
        assert torch.equal(value, reference[name])
    with pytest.raises(ValueError, match="world size"):
        restore_rank_rng([saved["rng"]], 0, 2, {"loader": generator})


def test_deterministic_execution_requires_valid_workspace(monkeypatch):
    """The deterministic policy is set before CUDA starts and checks the cuBLAS workspace."""
    old = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        with pytest.raises(RuntimeError, match="before CUDA"):
            configure_deterministic_training(True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
        configure_deterministic_training(True)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
        with pytest.raises(ValueError, match="Unsupported"):
            configure_deterministic_training(True)
        configure_deterministic_training(False)  # the default policy is not altered
    finally:
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[2:]
