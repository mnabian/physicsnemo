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

r"""Combined optimizer wrapper (ported from
``examples/cfd/external_aerodynamics/transformer_models/src/train.py`` for
the Muon+AdamW split used by the reference GeoTransolver recipe; that copy
is marked for upstreaming into physicsnemo — replace this one with the
library version when it lands)."""

from __future__ import annotations

from collections.abc import Sequence

from torch.optim import Optimizer


class CombinedOptimizer(Optimizer):
    r"""Combine multiple optimizers behind a single Optimizer interface.

    Concatenates *param_groups* from all contained optimizers so LR
    schedulers operate transparently across every parameter. Minimal
    subset of the ``torch.optim.Optimizer`` API.
    """

    def __init__(self, optimizers: Sequence[Optimizer]):
        if not optimizers:
            raise ValueError("`optimizers` must contain at least one optimizer.")
        self.optimizers = list(optimizers)
        param_groups = [g for opt in self.optimizers for g in opt.param_groups]
        super().__init__(param_groups, defaults={})

    def zero_grad(self, *args, **kwargs) -> None:
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    def step(self, closure=None) -> None:
        for opt in self.optimizers:
            if closure is None:
                opt.step()
            else:
                opt.step(closure)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)
        self.param_groups = [
            g for opt in self.optimizers for g in opt.param_groups
        ]


__all__ = ["CombinedOptimizer"]
