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

r"""Building-block layers for FlareJEPA.

Design notes:

- ``scale=1.0`` is a constraint on the *internals* of the reused
  :class:`~physicsnemo.experimental.nn.flare_attention.FLARE` /
  :class:`~physicsnemo.experimental.models.geotransolver.gale.GALE_FA`
  modules only. All NEW attention here uses conventional ``1/sqrt(d)``
  scaling plus QK-norm.
- Conditioning uses AdaLN-Zero (DiT-style): scale/shift after each
  pre-norm plus a zero-initialised gate on each residual branch, all
  regressed from a global conditioning embedding.
- The GALE context memory layout is split-per-head:
  ``(B, S, C) -> (B, H, S, C/H)`` with ``context_dim = C/H`` (resolved
  design decision; matches GeoTransolver's ContextProjector convention).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from physicsnemo.experimental.models.geotransolver.gale import GALE_FA
from physicsnemo.experimental.nn.flare_attention import FLARE
from physicsnemo.nn import FourierPositionalEmbedding, Mlp


def reshape_context(
    z: Float[torch.Tensor, "B S C"], heads: int
) -> Float[torch.Tensor, "B H S C_over_H"]:
    r"""Reshape a slot latent into the GALE_FA cross-attention memory layout.

    Split-per-head: ``(B, S, C) -> (B, H, S, C/H)``. The consuming GALE_FA
    stack must be constructed with ``context_dim = C // heads``.
    """
    B, S, C = z.shape
    if C % heads != 0:
        raise ValueError(f"token dim {C} not divisible by heads {heads}")
    return z.view(B, S, heads, C // heads).permute(0, 2, 1, 3)


class CondEmbed(nn.Module):
    r"""Fourier-feature + MLP embedding of global conditions (aoa, mach).

    Raw scalars are NOT fed directly: Mach nonlinearity in the transonic
    regime warrants a richer embedding than raw scalars.

    Forward: ``cond (B, cond_dim)`` -> ``(B, embed_dim)``.
    """

    def __init__(self, cond_dim: int, embed_dim: int, num_bands: int = 8) -> None:
        super().__init__()
        self.pe = FourierPositionalEmbedding(
            in_dim=cond_dim, num_bands=num_bands, include_input=True
        )
        self.mlp = Mlp(
            in_features=self.pe.out_dim,
            hidden_features=embed_dim,
            out_features=embed_dim,
            act_layer="silu",
        )

    def forward(
        self, cond: Float[torch.Tensor, "B cond_dim"]
    ) -> Float[torch.Tensor, "B embed_dim"]:
        return self.mlp(self.pe(cond))


class AdaLNZero(nn.Module):
    r"""AdaLN-Zero modulation head (DiT-style).

    Regresses ``n_chunks`` groups of per-channel (shift, scale, gate)
    parameters from a conditioning embedding. The final linear layer is
    zero-initialised so every block starts as identity.
    """

    def __init__(self, embed_dim: int, dim: int, n_chunks: int = 6) -> None:
        super().__init__()
        self.n_chunks = n_chunks
        self.proj = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dim, n_chunks * dim)
        )
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, cond_embed: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Each chunk: (B, 1, dim) so it broadcasts over the token axis.
        return self.proj(cond_embed).unsqueeze(1).chunk(self.n_chunks, dim=-1)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class _QKNormCrossAttention(nn.Module):
    r"""Multi-head cross-attention with per-head RMS QK-norm and ``1/sqrt(d)``
    scaling.

    Forward: ``q (B, Nq, C)``, ``kv (B, Nkv, C)`` -> ``(B, Nq, C)``.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        key_adaptive_temp: bool = False,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.dim_head = dim // heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.dim_head)
        self.k_norm = nn.RMSNorm(self.dim_head)
        # Per-KEY adaptive assignment temperature (Transolver++ "eidetic
        # state" mechanism, arXiv:2502.02414, transplanted to slot pooling):
        # each TOKEN carries a learned per-head sharpness tau(x_n) dividing
        # its post-QK-norm key, so tokens on sharp features can assign
        # themselves peakily to slots while bland tokens spread. Weight is
        # zero-init and the bias solves softplus(b) = 1, so tau == 1 at
        # init — the flag only ADDS capacity (exact legacy behaviour).
        self.to_tau = nn.Linear(dim, heads) if key_adaptive_temp else None
        if self.to_tau is not None:
            nn.init.zeros_(self.to_tau.weight)
            nn.init.constant_(self.to_tau.bias, math.log(math.e - 1.0))
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        B, N, _ = t.shape
        return t.view(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)

    def forward(
        self,
        q: Float[torch.Tensor, "B Nq C"],
        kv: Float[torch.Tensor, "B Nkv C"],
    ) -> Float[torch.Tensor, "B Nq C"]:
        qh = self.q_norm(self._split(self.to_q(q)))
        kh = self.k_norm(self._split(self.to_k(kv)))
        if self.to_tau is not None:
            # tau: (B, Nkv, H) -> (B, H, Nkv, 1); dividing the key scales
            # that token's logit COLUMN, i.e. per-token assignment
            # sharpness. fp32 softplus so bf16 cannot quantise tau near 1.
            tau = F.softplus(self.to_tau(kv).float()).clamp_min(0.05)
            kh = kh / tau.permute(0, 2, 1).unsqueeze(-1).to(kh.dtype)
        vh = self._split(self.to_v(kv))
        y = F.scaled_dot_product_attention(qh, kh, vh)
        B, _, Nq, _ = y.shape
        y = y.permute(0, 2, 1, 3).reshape(B, Nq, self.heads * self.dim_head)
        return self.dropout(self.out(y))


class CrossAttentionPool(nn.Module):
    r"""Perceiver-style pooling: slot queries cross-attend to point tokens.

    Standard pre-LN cross-attention block (QK-norm, ``1/sqrt(d)``) followed
    by an MLP, both with residuals on the query stream.

    Forward: ``slot_queries (B, S, C)``, ``tokens (B, N, C)`` -> ``(B, S, C)``.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        adaptive_temp: bool = False,
    ) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = _QKNormCrossAttention(
            dim, heads, dropout, key_adaptive_temp=adaptive_temp
        )
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim * mlp_ratio,
            out_features=dim,
            act_layer="gelu",
            drop=dropout,
        )

    def forward(
        self,
        slot_queries: Float[torch.Tensor, "B S C"],
        tokens: Float[torch.Tensor, "B N C"],
    ) -> Float[torch.Tensor, "B S C"]:
        x = slot_queries + self.attn(self.ln_q(slot_queries), self.ln_kv(tokens))
        return x + self.mlp(self.ln_mlp(x))


class FlarePointBlock(nn.Module):
    r"""Pre-LN transformer block with FLARE attention over point tokens.

    ``x = x + FLARE(norm(x)); x = x + MLP(norm(x))``.
    FLARE is reused untouched (``use_te=False``, its internal ``scale=1.0``).
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        n_global_queries: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = FLARE(
            dim=dim,
            heads=heads,
            dim_head=dim // heads,
            dropout=dropout,
            n_global_queries=n_global_queries,
            use_te=False,
        )
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim * mlp_ratio,
            out_features=dim,
            act_layer="gelu",
            drop=dropout,
        )

    def forward(self, x: Float[torch.Tensor, "B N C"]) -> Float[torch.Tensor, "B N C"]:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class SlotSelfAttentionBlock(nn.Module):
    r"""Plain pre-LN MHSA block for the (small) slot set. QK-norm, SDPA
    default scaling. Optionally AdaLN-Zero-conditioned.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        cond_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, elementwise_affine=cond_embed_dim is None)
        self.attn = _QKNormCrossAttention(dim, heads, dropout)
        self.ln_2 = nn.LayerNorm(dim, elementwise_affine=cond_embed_dim is None)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim * mlp_ratio,
            out_features=dim,
            act_layer="gelu",
            drop=dropout,
        )
        self.ada = AdaLNZero(cond_embed_dim, dim) if cond_embed_dim else None

    def forward(
        self,
        x: Float[torch.Tensor, "B S C"],
        cond_embed: Float[torch.Tensor, "B E"] | None = None,
    ) -> Float[torch.Tensor, "B S C"]:
        if self.ada is not None:
            if cond_embed is None:
                raise ValueError("cond_embed required for a conditioned block")
            sh1, sc1, g1, sh2, sc2, g2 = self.ada(cond_embed)
            h = _modulate(self.ln_1(x), sh1, sc1)
            x = x + g1 * self.attn(h, h)
            h = _modulate(self.ln_2(x), sh2, sc2)
            return x + g2 * self.mlp(h)
        h = self.ln_1(x)
        x = x + self.attn(h, h)
        return x + self.mlp(self.ln_2(x))


class CondGALEBlock(nn.Module):
    r"""AdaLN-Zero-conditioned transformer block around an untouched GALE_FA.

    Structure: pre-LN -> GALE_FA(x, context) -> gated residual -> pre-LN ->
    MLP -> gated residual. When ``cond_embed_dim`` is ``None`` the block is
    unconditioned (plain pre-LN residual block, used by the TargetEncoder).

    ``context_dim`` must equal ``dim // heads`` when a context memory of
    layout ``(B, H, S_c, C/H)`` (from :func:`reshape_context`) is passed.
    GALE_FA requires ``context_dim > 0`` at construction for its cross
    branch to exist — this block enforces that.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        n_global_queries: int,
        context_dim: int,
        cond_embed_dim: int | None = None,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        state_mixing_mode: str = "weighted",
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError(
                "CondGALEBlock requires context_dim > 0; GALE_FA builds its "
                "cross-attention branch only when context_dim is set."
            )
        self.ln_1 = nn.LayerNorm(dim, elementwise_affine=cond_embed_dim is None)
        self.attn = GALE_FA(
            dim,
            heads=heads,
            dim_head=dim // heads,
            dropout=dropout,
            n_global_queries=n_global_queries,
            use_te=False,
            context_dim=context_dim,
            state_mixing_mode=state_mixing_mode,
        )
        self.ln_2 = nn.LayerNorm(dim, elementwise_affine=cond_embed_dim is None)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim * mlp_ratio,
            out_features=dim,
            act_layer="gelu",
            drop=dropout,
        )
        self.ada = AdaLNZero(cond_embed_dim, dim) if cond_embed_dim else None

    def forward(
        self,
        x: Float[torch.Tensor, "B S C"],
        context: Float[torch.Tensor, "B H S_c D_c"],
        cond_embed: Float[torch.Tensor, "B E"] | None = None,
    ) -> Float[torch.Tensor, "B S C"]:
        if self.ada is not None:
            if cond_embed is None:
                raise ValueError("cond_embed required for a conditioned block")
            sh1, sc1, g1, sh2, sc2, g2 = self.ada(cond_embed)
            h = _modulate(self.ln_1(x), sh1, sc1)
            x = x + g1 * self.attn((h,), context)[0]
            h = _modulate(self.ln_2(x), sh2, sc2)
            return x + g2 * self.mlp(h)
        x = x + self.attn((self.ln_1(x),), context)[0]
        return x + self.mlp(self.ln_2(x))


class SlotReadCrossAttentionBlock(nn.Module):
    r"""Decoder cross-attention block: queries <- slots (global slot read).

    Plain Perceiver-IO cross-attention (QK-norm, ``1/sqrt(d)``) followed by
    an MLP, with AdaLN-Zero conditioning when ``cond_embed_dim`` is set.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        cond_embed_dim: int | None = None,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(dim, elementwise_affine=cond_embed_dim is None)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = _QKNormCrossAttention(dim, heads, dropout)
        self.ln_mlp = nn.LayerNorm(dim, elementwise_affine=cond_embed_dim is None)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim * mlp_ratio,
            out_features=dim,
            act_layer="gelu",
            drop=dropout,
        )
        self.ada = AdaLNZero(cond_embed_dim, dim) if cond_embed_dim else None

    def forward(
        self,
        x: Float[torch.Tensor, "B Nq C"],
        z: Float[torch.Tensor, "B S C"],
        cond_embed: Float[torch.Tensor, "B E"] | None = None,
    ) -> Float[torch.Tensor, "B Nq C"]:
        kv = self.ln_kv(z)
        if self.ada is not None:
            if cond_embed is None:
                raise ValueError("cond_embed required for a conditioned block")
            sh1, sc1, g1, sh2, sc2, g2 = self.ada(cond_embed)
            h = _modulate(self.ln_q(x), sh1, sc1)
            x = x + g1 * self.attn(h, kv)
            h = _modulate(self.ln_mlp(x), sh2, sc2)
            return x + g2 * self.mlp(h)
        x = x + self.attn(self.ln_q(x), kv)
        return x + self.mlp(self.ln_mlp(x))
