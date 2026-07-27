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

r"""FlareJEPA encoders: geometry (FLARE) and target (teacher, train only).

Both encoders share :class:`_SlotPool`: ``S`` learned slot queries
(Perceiver latent array — position-free, content-addressed) pooled from
point tokens via cross-attention. Slot ``i`` corresponds by shared index
on the student and teacher sides — the JEPA latent-alignment contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core.module import Module
from physicsnemo.nn import FourierPositionalEmbedding

from ._metadata import FlareJEPAMetaData
from .layers import (
    CondGALEBlock,
    CrossAttentionPool,
    FlarePointBlock,
    SlotSelfAttentionBlock,
    reshape_context,
)


class _PointEmbed(nn.Module):
    r"""Linear feature embed + projected Fourier positional encoding.

    ``x = feature_in(features) + pos_proj(pe(positions))``.
    """

    def __init__(self, in_dim: int, token_dim: int, pe_bands: int) -> None:
        super().__init__()
        self.pe = FourierPositionalEmbedding(
            in_dim=3, num_bands=pe_bands, include_input=True
        )
        self.feature_in = nn.Linear(in_dim, token_dim)
        self.pos_proj = nn.Linear(self.pe.out_dim, token_dim)

    def forward(
        self,
        positions: Float[torch.Tensor, "B N 3"],
        features: Float[torch.Tensor, "B N F"],
    ) -> Float[torch.Tensor, "B N C"]:
        # Fourier PE at high bands is precision-fragile: ensure at
        # least fp32 (upcast bf16/fp16, never downcast fp64).
        pos = positions.to(torch.promote_types(positions.dtype, torch.float32))
        return self.feature_in(features) + self.pos_proj(self.pe(pos))


class _SlotPool(nn.Module):
    r"""Learned slot queries pooled from point tokens via cross-attention.

    The ``S`` slot queries are a learned parameter array (Perceiver latent
    array) — position-free, content-addressed.
    """

    def __init__(
        self,
        token_dim: int,
        heads: int,
        slots: int,
        mlp_ratio: int,
        dropout: float,
        adaptive_temp: bool = False,
    ) -> None:
        super().__init__()
        self.slot_queries = nn.Parameter(torch.randn(1, slots, token_dim) * 0.02)
        self.pool = CrossAttentionPool(
            token_dim, heads, mlp_ratio, dropout, adaptive_temp=adaptive_temp
        )

    def forward(
        self, tokens: Float[torch.Tensor, "B N C"]
    ) -> Float[torch.Tensor, "B S C"]:
        q = self.slot_queries.expand(tokens.shape[0], -1, -1)
        return self.pool(q, tokens)


class GeometryEncoder(Module):
    r"""FLARE geometry encoder — encode-once, produces the slot latent.

    Pipeline: point embed + Fourier PE -> ``flare_layers`` x
    :class:`FlarePointBlock` -> cross-attention pool to ``slots`` learned
    slot queries -> ``slot_layers`` x slot MHSA (+ optional re-pool rounds).

    Parameters
    ----------
    in_dim : int
        Per-point geometry feature dim (3 for xyz; +3 normals, +1 SDF).
    token_dim, heads, slots, pe_bands : int
        Latent geometry (C, H, S, Fourier bands).
    flare_layers, slot_layers : int
        Depth of the point-level FLARE stack (may be 0 for a
        Perceiver-pure variant) and of the slot self-attention stack.
    pool_repeats : int
        Total pooling rounds. Both strong reference models re-acquire
        geometry at depth; each extra round re-reads the point tokens with
        the current slots as queries (residual), then re-mixes the slots.
    """

    def __init__(
        self,
        in_dim: int = 3,
        token_dim: int = 256,
        heads: int = 8,
        slots: int = 128,
        pe_bands: int = 16,
        flare_layers: int = 4,
        slot_layers: int = 2,
        pool_repeats: int = 1,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        pool_adaptive_temp: bool = False,
    ) -> None:
        super().__init__(meta=FlareJEPAMetaData())
        if pool_repeats < 1:
            raise ValueError(f"pool_repeats must be >= 1, got {pool_repeats}")
        self.slots = slots
        self.embed = _PointEmbed(in_dim, token_dim, pe_bands)
        self.point_blocks = nn.ModuleList(
            FlarePointBlock(token_dim, heads, slots, mlp_ratio, dropout)
            for _ in range(flare_layers)
        )
        self.slot_pool = _SlotPool(
            token_dim, heads, slots, mlp_ratio, dropout,
            adaptive_temp=pool_adaptive_temp,
        )
        self.slot_blocks = nn.ModuleList(
            SlotSelfAttentionBlock(token_dim, heads, mlp_ratio, dropout)
            for _ in range(slot_layers)
        )
        self.repool = nn.ModuleList(
            CrossAttentionPool(
                token_dim, heads, mlp_ratio, dropout,
                adaptive_temp=pool_adaptive_temp,
            )
            for _ in range(pool_repeats - 1)
        )
        self.repool_slot_blocks = nn.ModuleList(
            SlotSelfAttentionBlock(token_dim, heads, mlp_ratio, dropout)
            for _ in range(pool_repeats - 1)
        )
        self.out_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        positions: Float[torch.Tensor, "B N_g 3"],
        features: Float[torch.Tensor, "B N_g F"] | None = None,
        return_point_tokens: bool = False,
    ):
        r"""Encode geometry points into the slot latent ``Z_ctx``.

        ``features`` defaults to the positions themselves (xyz-only input).
        With ``return_point_tokens=True`` also returns the pre-pool point
        tokens ``(B, N_g, C)`` (position-carrying, deterministic from
        geometry) for the dual-read decoder.
        """
        if features is None:
            features = positions
        x = self.embed(positions, features)
        for block in self.point_blocks:
            x = block(x)
        z = self.slot_pool(x)
        for block in self.slot_blocks:
            z = block(z)
        for pool, blk in zip(self.repool, self.repool_slot_blocks):
            z = pool(z, x)
            z = blk(z)
        if return_point_tokens:
            return self.out_norm(z), x
        return self.out_norm(z)


class TargetEncoder(Module):
    r"""Field target encoder — the JEPA teacher (train only).

    Pipeline (pool-first): point embed on field points -> pool to slots ->
    ``gale_layers`` refinement blocks. With ``context_cross=True`` the
    refinement blocks are unconditioned :class:`CondGALEBlock` s
    cross-attending to the geometry memory ``reshape_context(Z_ctx)``.
    With ``context_cross=False`` (copy-proof teacher) the geometry memory
    is removed entirely — the teacher CANNOT copy ``Z_ctx`` by
    construction, which closes the copy path a joint (non-stop-grad)
    latent loss would otherwise exploit — and the slots are refined by
    self-attention instead.

    ``in_dim`` is the per-point input dim: 3 (xyz) + field channels
    (+ optional normals/SDF, mirroring the geometry encoder flags).
    """

    def __init__(
        self,
        in_dim: int = 6,
        token_dim: int = 256,
        heads: int = 8,
        slots: int = 128,
        pe_bands: int = 16,
        flare_layers: int = 0,
        gale_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        state_mixing_mode: str = "weighted",
        context_cross: bool = True,
        pool_adaptive_temp: bool = False,
    ) -> None:
        super().__init__(meta=FlareJEPAMetaData())
        self.heads = heads
        self.context_cross = context_cross
        self.embed = _PointEmbed(in_dim, token_dim, pe_bands)
        # flare_layers>0 gives field tokens point-level context before the
        # depth-1 compression of the pool.
        self.point_blocks = nn.ModuleList(
            FlarePointBlock(token_dim, heads, slots, mlp_ratio, dropout)
            for _ in range(flare_layers)
        )
        self.slot_pool = _SlotPool(
            token_dim, heads, slots, mlp_ratio, dropout,
            adaptive_temp=pool_adaptive_temp,
        )
        if context_cross:
            self.blocks = nn.ModuleList(
                CondGALEBlock(
                    token_dim,
                    heads,
                    n_global_queries=slots,
                    context_dim=token_dim // heads,
                    cond_embed_dim=None,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    state_mixing_mode=state_mixing_mode,
                )
                for _ in range(gale_layers)
            )
        else:
            self.blocks = nn.ModuleList(
                SlotSelfAttentionBlock(token_dim, heads, mlp_ratio, dropout)
                for _ in range(gale_layers)
            )
        self.out_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        positions: Float[torch.Tensor, "B N_t 3"],
        features: Float[torch.Tensor, "B N_t F"],
        z_ctx: Float[torch.Tensor, "B S C"],
    ) -> Float[torch.Tensor, "B S C"]:
        r"""Encode field points into the teacher latent ``Z_tgt``.

        The caller applies stop-grad where ``Z_tgt`` feeds the latent loss.
        """
        x = self.embed(positions, features)
        for block in self.point_blocks:
            x = block(x)
        z = self.slot_pool(x)
        if self.context_cross:
            context = reshape_context(z_ctx, self.heads)
            for block in self.blocks:
                z = block(z, context)
        else:
            for block in self.blocks:
                z = block(z)
        return self.out_norm(z)
