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

r"""FlareJEPA Perceiver-IO decoder: query positions x slot latent -> field."""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core.module import Module
from physicsnemo.nn import FourierPositionalEmbedding, Mlp

from physicsnemo.nn.module.point_transformer_attention import (
    LocalTokenCrossAttentionBlock,
    _dilated_knn,
)

from ._metadata import FlareJEPAMetaData
from .layers import SlotReadCrossAttentionBlock


class Decoder(Module):
    r"""Continuous field decoder (dual-read).

    Each of the ``cross_layers`` blocks performs a GLOBAL slot read (the
    query cross-attends to the slot latent through an AdaLN-Zero-conditioned
    Perceiver-IO block) followed, when ``point_read=True``, by a LOCAL point
    read: the query cross-attends to its ``point_neighbor_k`` nearest
    geometry point tokens via kNN relative-position vector attention. The
    point read supplies the local geometric evidence that gives strong
    baselines their high-frequency fidelity; it is AdaLN-Zero-gated, so it
    is an exact identity at initialisation.

    Every query attends only to its own kNN neighbourhood and the (global)
    slot latent — never to other queries — so the decode is provably
    invariant to how the query set is chunked (`forward_chunked`).
    """

    def __init__(
        self,
        token_dim: int = 256,
        heads: int = 8,
        out_dim: int = 3,
        pe_bands: int = 16,
        cross_layers: int = 4,
        cond_embed_dim: int = 64,
        mlp_ratio: int = 4,
        head_mlp_ratio: int = 1,
        dropout: float = 0.0,
        query_chunk_size: int = 4096,
        point_read: bool = False,
        point_neighbor_k: int = 8,
    ) -> None:
        super().__init__(meta=FlareJEPAMetaData())
        self.query_chunk_size = query_chunk_size
        self.point_read = point_read
        self.pe = FourierPositionalEmbedding(
            in_dim=3, num_bands=pe_bands, include_input=True
        )
        self.query_proj = nn.Linear(self.pe.out_dim, token_dim)
        self.blocks = nn.ModuleList(
            SlotReadCrossAttentionBlock(
                token_dim,
                heads,
                cond_embed_dim=cond_embed_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(cross_layers)
        )
        self.point_blocks = (
            nn.ModuleList(
                LocalTokenCrossAttentionBlock(
                    dim=token_dim,
                    num_heads=heads,
                    neighbor_k=point_neighbor_k,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    conditioning_dim=cond_embed_dim,
                    adaln_zero=True,
                    use_te=False,
                )
                for _ in range(cross_layers)
            )
            if point_read
            else None
        )
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            Mlp(
                in_features=token_dim,
                hidden_features=token_dim * head_mlp_ratio,
                out_features=out_dim,
                act_layer="gelu",
            ),
        )

    def forward(
        self,
        z: Float[torch.Tensor, "B S C"],
        query_positions: Float[torch.Tensor, "B N_q 3"],
        cond_embed: Float[torch.Tensor, "B E"],
        point_tokens: Float[torch.Tensor, "B N_p C"] | None = None,
        point_positions: Float[torch.Tensor, "B N_p 3"] | None = None,
    ) -> Float[torch.Tensor, "B N_q out_dim"]:
        if self.point_blocks is not None and (
            point_tokens is None or point_positions is None
        ):
            raise ValueError(
                "point_read=True requires point_tokens and point_positions"
            )
        # Fourier PE at high bands is precision-fragile: ensure at
        # least fp32 (upcast bf16/fp16, never downcast fp64).
        q_pos = query_positions.to(
            torch.promote_types(query_positions.dtype, torch.float32)
        )
        x = self.query_proj(self.pe(q_pos))
        knn_idx = None
        if self.point_blocks is not None:
            # The kNN graph depends only on (query, point) coordinates —
            # identical for every point-read block. Compute once per
            # sample per decode and thread precomputed_idx through.
            p_pos32 = point_positions.to(
                torch.promote_types(point_positions.dtype, torch.float32)
            )
            k = int(self.point_blocks[0].neighbor_k)
            knn_idx = [
                _dilated_knn(
                    query_coords=q_pos[b],
                    key_coords=p_pos32[b],
                    k=min(k, int(p_pos32.shape[1])),
                    dilation=1,
                )
                for b in range(q_pos.shape[0])
            ]
        for i, block in enumerate(self.blocks):
            x = block(x, z, cond_embed=cond_embed)
            if self.point_blocks is not None:
                x = self._point_read(
                    self.point_blocks[i],
                    x,
                    q_pos,
                    point_tokens,
                    point_positions,
                    cond_embed,
                    knn_idx=knn_idx,
                )
        return self.head(x)

    @staticmethod
    def _point_read(
        block: LocalTokenCrossAttentionBlock,
        x: Float[torch.Tensor, "B N_q C"],
        q_pos: Float[torch.Tensor, "B N_q 3"],
        point_tokens: Float[torch.Tensor, "B N_p C"],
        point_positions: Float[torch.Tensor, "B N_p 3"],
        cond_embed: Float[torch.Tensor, "B E"],
        knn_idx: list | None = None,
    ) -> Float[torch.Tensor, "B N_q C"]:
        r"""Per-sample packed calls into the kNN cross block.

        The block is packed-format ``(N, C)``; looping over the batch keeps
        each kNN search inside one sample (no cross-sample neighbours, no
        B*N x B*N distance blow-up). ``knn_idx`` (one per sample) skips the
        block-internal neighbour search.
        """
        p_pos = point_positions.to(
            torch.promote_types(point_positions.dtype, torch.float32)
        )
        out = [
            block(
                x[b],
                q_pos[b],
                point_tokens[b],
                p_pos[b],
                cond=cond_embed[b],
                precomputed_idx=None if knn_idx is None else knn_idx[b],
            )
            for b in range(x.shape[0])
        ]
        return torch.stack(out, dim=0)

    def forward_chunked(
        self,
        z: Float[torch.Tensor, "B S C"],
        query_positions: Float[torch.Tensor, "B N_q 3"],
        cond_embed: Float[torch.Tensor, "B E"],
        chunk_size: int | None = None,
        point_tokens: Float[torch.Tensor, "B N_p C"] | None = None,
        point_positions: Float[torch.Tensor, "B N_p 3"] | None = None,
    ) -> Float[torch.Tensor, "B N_q out_dim"]:
        r"""Memory-bounded decode: chunk the query axis.

        Deliberately NOT decorated with ``torch.no_grad`` — a training path
        that chunks the decode for memory must keep gradients. Inference
        callers (``FlareJEPA.predict``) supply their own no-grad context.
        """
        chunk = chunk_size or self.query_chunk_size
        outputs = [
            self.forward(
                z,
                query_positions[:, i : i + chunk],
                cond_embed,
                point_tokens=point_tokens,
                point_positions=point_positions,
            )
            for i in range(0, query_positions.shape[1], chunk)
        ]
        return torch.cat(outputs, dim=1)
