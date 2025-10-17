# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from physicsnemo.models.meshgraphnet import MeshGraphNet, HybridMeshGraphNet
from physicsnemo.models.transolver import Transolver
from physicsnemo.utils.profiling import profile

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.utils.neighbors.radius_search import radius_search

import torch

from torch import Tensor
from typing import Callable, List, Tuple, Union, Dict

from torch.utils.checkpoint import checkpoint as ckpt
from physicsnemo import Module
import torch.nn as nn

from dataclasses import dataclass
from physicsnemo.models.meta import ModelMetaData

from omegaconf import DictConfig
from hydra.utils import instantiate
from physicsnemo.models.transolver import Transolver


class TransolverAutoregressiveRolloutTraining(Transolver):
    """
    Transolver model with autoregressive rollout training.
    """

    def __init__(self, *args, **kwargs):
        self.dt = kwargs.pop("dt")
        self.initial_vel = kwargs.pop("initial_vel")
        self.rollout_steps = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, node_features, data_stats):
        outputs = []
        y_t1 = node_features[..., :3]
        thickness = node_features[..., -1:]
        y_t0 = y_t1 - self.initial_vel * self.dt

        for t in range(self.rollout_steps):
            time_t = 0.0 if self.rollout_steps <= 1 else t / (self.rollout_steps - 1)
            time_t = torch.tensor([time_t], device=y_t1.device)
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / data_stats["node"][
                "norm_vel_std"
            ]
            fx_t = torch.cat(
                [vel_norm, thickness, time_t.expand(y_t1.shape[0], 1)], dim=-1
            )

            if self.training:

                def step_fn(fx, embedding):
                    return super(TransolverAutoregressiveRolloutTraining, self).forward(
                        fx=fx, embedding=embedding
                    )

                outf = ckpt(
                    step_fn, fx_t.unsqueeze(0), y_t1.unsqueeze(0), use_reentrant=False
                ).squeeze(0)
            else:
                outf = (
                    super(TransolverAutoregressiveRolloutTraining, self)
                    .forward(fx=fx_t.unsqueeze(0), embedding=y_t1.unsqueeze(0))
                    .squeeze(0)
                )

            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel = self.dt * acc + vel
            y_t2 = self.dt * vel + y_t1
            outputs.append(y_t2.clone())
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)


class TransolverTimeConditionalRollout(Transolver):
    """
    Transolver model with time-conditional rollout.
    """

    def __init__(self, *args, **kwargs):
        self.rollout_steps = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, node_features, data_stats) -> Tensor:
        x = node_features[..., :3]
        thickness = node_features[..., -1:]
        outputs: List[Tensor] = []
        time_seq = torch.linspace(0.0, 1.0, self.rollout_steps, device=x.device)
        for time in time_seq:
            fx_t = thickness
            if self.training:

                def step_fn(fx, embedding, time):
                    return super(TransolverTimeConditionalRollout, self).forward(
                        fx=fx, embedding=embedding, time=time
                    )

                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    x.unsqueeze(0),
                    time=time.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = (
                    super(TransolverTimeConditionalRollout, self)
                    .forward(
                        fx=fx_t.unsqueeze(0),
                        embedding=x.unsqueeze(0),
                        time=time.unsqueeze(0),
                    )
                    .squeeze(0)
                )
            y_t2 = x + outf
            outputs.append(y_t2.clone())
        return torch.stack(outputs, dim=0)


class TransolverOneStepRollout(Transolver):
    """
    Transolver model with one-step rollout.
    """

    def __init__(self, *args, **kwargs):
        self.dt = kwargs.pop("dt")
        self.initial_vel = kwargs.pop("initial_vel")
        self.rollout_steps = kwargs.pop("num_time_steps") - 1
        self.sigma = kwargs.pop("sigma")
        super().__init__(*args, **kwargs)
        self.add_noise = lambda y: y + torch.randn_like(y) * self.sigma

    def forward(self, node_features, data_stats):
        outputs = []
        y_t1 = node_features[..., :3]
        thickness = node_features[..., -1:]
        y_t0 = y_t1 - self.initial_vel * self.dt
        y_t0, y_t1 = self.add_noise(y_t0), self.add_noise(y_t1)

        for t in range(self.rollout_steps):
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / data_stats["node"][
                "norm_vel_std"
            ]
            fx_t = torch.cat([vel_norm, thickness], dim=-1)
            outf = (
                super(TransolverAutoregressiveRolloutTraining, self)
                .forward(fx=fx_t.unsqueeze(0), embedding=y_t1.unsqueeze(0))
                .squeeze(0)
            )

            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel = self.dt * acc + vel
            y_t2 = self.dt * vel + y_t1
            outputs.append(y_t2.clone())
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)
