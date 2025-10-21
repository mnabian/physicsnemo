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

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt
from typing import List
from physicsnemo.models.transolver import Transolver

EPS = 1e-8


class TransolverAutoregressiveRolloutTraining(Transolver):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(
        self,
        node_features: torch.Tensor,
        data_stats: dict,
        edge_index=None,
        edge_features=None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N,F_in]
            data_stats: dict containing normalization stats
        Returns:
            [T, N, F_out] rollout of predicted positions
        """

        N = node_features.size(0)
        device = node_features.device

        # Initial states
        y_t1 = node_features[..., :3]  # [N,3]
        thickness = node_features[..., -1:]  # [N,1]
        y_t0 = y_t1 - self.initial_vel * self.dt  # backstep using initial velocity

        outputs: List[torch.Tensor] = []
        for t in range(self.rollout_steps):
            time_t = 0.0 if self.rollout_steps <= 1 else t / (self.rollout_steps - 1)
            time_t = torch.tensor([time_t], device=device, dtype=torch.float32)

            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            # Model input
            fx_t = torch.cat(
                [vel_norm, thickness, time_t.expand(N, 1)], dim=-1
            )  # [N, 3+1+1]

            def step_fn(fx, embedding):
                return super(TransolverAutoregressiveRolloutTraining, self).forward(
                    fx=fx, embedding=embedding
                )

            if self.training:
                outf = ckpt(
                    step_fn, fx_t.unsqueeze(0), y_t1.unsqueeze(0), use_reentrant=False
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), y_t1.unsqueeze(0)).squeeze(0)

            # De-normalize acceleration
            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel = self.dt * acc + vel
            y_t2 = self.dt * vel + y_t1

            outputs.append(y_t2.clone())
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverTimeConditionalRollout(Transolver):
    """
    Transolver model with time-conditional rollout.

    Predicts each time step independently, conditioned on normalized time.
    """

    def __init__(self, *args, **kwargs):
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(
        self,
        node_features: torch.Tensor,
        data_stats: dict,
        edge_index=None,
        edge_features=None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N,4] (pos(3) + thickness(1))
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        assert node_features.ndim == 2 and node_features.shape[1] == 4, (
            f"Expected node_features [N,4], got {node_features.shape}"
        )

        x = node_features[..., :3]  # initial pos
        thickness = node_features[..., -1:]
        outputs: List[torch.Tensor] = []

        time_seq = torch.linspace(0.0, 1.0, self.rollout_steps, device=x.device)
        N = x.shape[0]

        def step_fn(fx, embedding, time):
            return super(TransolverTimeConditionalRollout, self).forward(
                fx=fx, embedding=embedding, time=time
            )

        for time in time_seq:
            fx_t = thickness  # [N,1]

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    x.unsqueeze(0),
                    time.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(
                    fx_t.unsqueeze(0), x.unsqueeze(0), time.unsqueeze(0)
                ).squeeze(0)

            y_t2 = x + outf
            outputs.append(y_t2.clone())

        return torch.stack(outputs, dim=0)  # [T,N,3]


# class TransolverOneStepRollout(Transolver):
#     """
#     Transolver with single-step rollout compatible with CrashDataset datapipe.
#     Uses t=0 and t=1 from node_features, then autoregressively predicts the rest.
#     """

#     def __init__(self, *args, **kwargs):
#         self.dt: float = kwargs.pop("dt", 5e-3)
#         self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
#         super().__init__(*args, **kwargs)

#     def forward(
#         self,
#         node_features: Tensor,   # [N,7] -> pos_t0(3), pos_t1(3), thickness(1)
#         data_stats: dict,
#         edge_index=None,
#         edge_features=None,
#     ) -> Tensor:
#         """
#         Args:
#             node_features: [N,7] containing pos_t0, pos_t1, thickness
#             data_stats: normalization stats
#         Returns:
#             [T, N, 3] rollout of predicted positions (T = rollout_steps)
#         """
#         dt = self.dt
#         N = node_features.size(0)

#         # Unpack features
#         y_t0 = node_features[:, 0:3]   # initial pos
#         y_t1 = node_features[:, 3:6]   # next pos
#         thickness = node_features[:, -1:].contiguous()

#         outputs: List[Tensor] = []
#         for t in range(self.rollout_steps):
#             vel = (y_t1 - y_t0) / dt
#             vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
#                 data_stats["node"]["norm_vel_std"] + EPS
#             )

#             fx_t = torch.cat([vel_norm, thickness], dim=-1)

#             def step_fn(fx, embedding):
#                 return super(TransolverOneStepRollout, self).forward(
#                     fx=fx, embedding=embedding
#                 )

#             if self.training:
#                 outf = ckpt(step_fn, fx_t.unsqueeze(0), y_t1.unsqueeze(0),
#                             use_reentrant=False).squeeze(0)
#             else:
#                 outf = step_fn(fx_t.unsqueeze(0), y_t1.unsqueeze(0)).squeeze(0)

#             acc = (
#                 outf * data_stats["node"]["norm_acc_std"]
#                 + data_stats["node"]["norm_acc_mean"]
#             )
#             vel = dt * acc + vel
#             y_t2 = dt * vel + y_t1
#             outputs.append(y_t2.clone())

#             # autoregressive update
#             y_t0, y_t1 = y_t1, y_t2

#         return torch.stack(outputs, dim=0)  # [T,N,3]
