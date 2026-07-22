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

r"""FlareJEPA predictor: context latent + conditions -> predicted target latent."""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core.module import Module

from ._metadata import FlareJEPAMetaData
from .layers import CondGALEBlock, reshape_context


class Predictor(Module):
    r"""GALE_FA predictor — infers the field-aware latent from geometry alone.

    Target-query slots are learned parameters (slot ``i`` on the predictor
    and teacher sides corresponds by shared index, enforced by the per-slot
    latent loss), optionally initialised additively from ``Z_ctx``, then
    refined by ``gale_layers`` AdaLN-Zero-conditioned
    :class:`CondGALEBlock` cross-attending to the geometry memory
    ``reshape_context(Z_ctx)``.

    Runs at inference; trained to match the stop-grad teacher ``Z_tgt``.
    """

    def __init__(
        self,
        token_dim: int = 256,
        heads: int = 8,
        slots: int = 128,
        gale_layers: int = 6,
        cond_embed_dim: int = 64,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        state_mixing_mode: str = "weighted",
        init_from_context: bool = True,
    ) -> None:
        super().__init__(meta=FlareJEPAMetaData())
        self.heads = heads
        self.init_from_context = init_from_context
        self.slot_queries = nn.Parameter(torch.randn(1, slots, token_dim) * 0.02)
        self.blocks = nn.ModuleList(
            CondGALEBlock(
                token_dim,
                heads,
                n_global_queries=slots,
                context_dim=token_dim // heads,
                cond_embed_dim=cond_embed_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                state_mixing_mode=state_mixing_mode,
            )
            for _ in range(gale_layers)
        )
        # Non-affine: Z_hat lives natively in the CANONICAL latent space
        # (token-wise zero-mean/unit-var), the same space as the
        # normalised teacher target and the teacher-forced decoder input
        # — no implicit reliance on affine params drifting to identity.
        self.out_norm = nn.LayerNorm(token_dim, elementwise_affine=False)

    def forward(
        self,
        z_ctx: Float[torch.Tensor, "B S C"],
        cond_embed: Float[torch.Tensor, "B E"],
    ) -> Float[torch.Tensor, "B S C"]:
        q = self.slot_queries.expand(z_ctx.shape[0], -1, -1)
        if self.init_from_context:
            q = q + z_ctx
        context = reshape_context(z_ctx, self.heads)
        for block in self.blocks:
            q = block(q, context, cond_embed)
        return self.out_norm(q)
