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

r"""FLARE-family Transolver models.

These Transolver variants replace physics attention with either fixed-query
FLARE or input-conditioned FLARE++ attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.experimental.nn import FLARE as FLAREAttention
from physicsnemo.experimental.nn import FLAREPlusPlus as FLAREPlusPlusAttention
from physicsnemo.models.transolver import Transolver as CoreTransolver
from physicsnemo.models.transolver.transolver import _TransolverMlp


class _FLAREBlock(nn.Module):
    r"""Transformer block with FLARE-family attention.

    Mirrors ``TransolverBlock`` while replacing physics attention with a
    supplied FLARE-family layer. Transformer Engine is not supported.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        n_global_queries: int = 32,
        attention_cls: type[nn.Module] = FLAREAttention,
        attn_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.last_layer = last_layer
        dim_head = hidden_dim // num_heads

        self.ln_1 = nn.LayerNorm(hidden_dim)
        attention_kwargs = {}
        if attn_scale is not None:
            attention_kwargs["attn_scale"] = attn_scale
        self.Attn = attention_cls(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            n_global_queries=n_global_queries,
            use_te=False,
            **attention_kwargs,
        )
        self.ln_mlp1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            _TransolverMlp(
                in_features=hidden_dim,
                hidden_features=hidden_dim * mlp_ratio,
                out_features=hidden_dim,
                act_layer=act,
                use_te=False,
            ),
        )
        if last_layer:
            self.ln_mlp2 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(
        self, fx: Float[torch.Tensor, "B N C"]
    ) -> Float[torch.Tensor, "B N C_out"]:
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.ln_mlp1(fx) + fx
        if self.last_layer:
            return self.ln_mlp2(fx)
        return fx


def _make_flare_blocks(
    *,
    attention_cls: type[nn.Module],
    n_layers: int,
    n_head: int,
    n_hidden: int,
    dropout: float,
    act: str,
    mlp_ratio: int,
    out_dim: int,
    n_global_queries: int,
    attn_scale: float | None = None,
) -> nn.ModuleList:
    """Build a matched residual stack for a FLARE-family attention layer."""
    return nn.ModuleList(
        [
            _FLAREBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                last_layer=(i == n_layers - 1),
                out_dim=out_dim,
                n_global_queries=n_global_queries,
                attention_cls=attention_cls,
                attn_scale=attn_scale,
            )
            for i in range(n_layers)
        ]
    )


class FLARE(CoreTransolver):
    r"""Transolver with FLARE attention.

    Inherits from the core Transolver and replaces all physics attention blocks
    with FLARE (Fast Low-rank Attention Routing Engine) blocks. Transformer
    Engine is not supported (use_te is forced to False).

    Parameters
    ----------
    functional_dim : int
        Dimension of input values, not including embeddings.
    out_dim : int
        Dimension of model output.
    embedding_dim : int | None, optional
        Dimension of input embeddings. Required if ``unified_pos=False``.
    n_layers : int, optional
        Number of transformer blocks. Default is 4.
    n_hidden : int, optional
        Hidden dimension. Default is 256.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    n_head : int, optional
        Number of attention heads. Default is 8.
    act : str, optional
        Activation function name. Default is ``"gelu"``.
    mlp_ratio : int, optional
        MLP hidden ratio. Default is 4.
    slice_num : int, optional
        Number of global queries for FLARE attention. Default is 32.
    unified_pos : bool, optional
        Whether to use unified positional embeddings. Default is ``False``.
    ref : int, optional
        Reference grid size for unified position. Default is 8.
    structured_shape : None | tuple[int, ...], optional
        Shape of structured data. ``None`` for unstructured. Default is ``None``.
    time_input : bool, optional
        Whether to include time embeddings. Default is ``False``.
    attn_scale : float, optional
        Attention-logit scale. Default is ``1.0`` for backward compatibility
        with the original FLARE implementation.

    Forward
    -------
    Same as :class:`~physicsnemo.models.transolver.Transolver`.

    Outputs
    -------
    Same as :class:`~physicsnemo.models.transolver.Transolver`.

    See Also
    --------
    :class:`~physicsnemo.models.transolver.Transolver` : Core Transolver model.
    :class:`~physicsnemo.experimental.nn.flare_attention.FLARE` : FLARE attention layer.
    """

    def __init__(
        self,
        functional_dim: int,
        out_dim: int,
        embedding_dim: int | None = None,
        n_layers: int = 4,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 4,
        slice_num: int = 32,
        unified_pos: bool = False,
        ref: int = 8,
        structured_shape: None | tuple[int, ...] = None,
        time_input: bool = False,
        attn_scale: float = 1.0,
    ) -> None:
        super().__init__(
            functional_dim=functional_dim,
            out_dim=out_dim,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
            unified_pos=unified_pos,
            ref=ref,
            structured_shape=structured_shape,
            use_te=False,
            time_input=time_input,
            plus=False,
        )

        # Replace physics attention blocks with FLARE blocks
        self.blocks = _make_flare_blocks(
            attention_cls=FLAREAttention,
            n_layers=n_layers,
            n_head=n_head,
            n_hidden=n_hidden,
            dropout=dropout,
            act=act,
            mlp_ratio=mlp_ratio,
            out_dim=out_dim,
            n_global_queries=slice_num,
            attn_scale=attn_scale,
        )
        self.initialize_weights()


class FLAREPlusPlus(CoreTransolver):
    r"""Transolver with FLARE++ dynamic-routing attention.

    FLARE++ synthesizes its routing queries from each block's current input
    before applying FLARE's low-rank gather-scatter operator. This changes only
    the token mixer: the Transolver residual stream, normalization, and
    feed-forward blocks are identical to :class:`FLARE`.

    For architecture details, see:

    - `FLARE++: Low-rank attention with dynamic attention routing
      <https://arxiv.org/abs/2608.11519>`_

    Parameters
    ----------
    functional_dim : int
        Dimension of input values, not including embeddings.
    out_dim : int
        Dimension of model output.
    embedding_dim : int | None, optional
        Dimension of input embeddings. Required if ``unified_pos=False``.
    n_layers : int, optional
        Number of transformer blocks. Default is 4.
    n_hidden : int, optional
        Hidden dimension. Default is 256.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    n_head : int, optional
        Number of attention heads. Default is 8.
    act : str, optional
        Activation function name. Default is ``"gelu"``.
    mlp_ratio : int, optional
        MLP hidden ratio. Default is 4.
    slice_num : int, optional
        Number of dynamic routing queries. Default is 32.
    unified_pos : bool, optional
        Whether to use unified positional embeddings. Default is ``False``.
    ref : int, optional
        Reference grid size for unified position. Default is 8.
    structured_shape : None | tuple[int, ...], optional
        Shape of structured data. ``None`` for unstructured data. Default is
        ``None``.
    time_input : bool, optional
        Whether to include time embeddings. Default is ``False``.
    attn_scale : float | None, optional
        Attention-logit scale. ``None`` uses ``(n_hidden / n_head) ** -0.5``
        as specified by FLARE++. Pass ``1.0`` for historical FLARE scaling.

    Forward
    -------
    Same as :class:`~physicsnemo.models.transolver.Transolver`.

    Outputs
    -------
    Same as :class:`~physicsnemo.models.transolver.Transolver`.

    See Also
    --------
    :class:`FLARE` : Fixed-query FLARE model.
    :class:`~physicsnemo.experimental.nn.flare_attention.FLAREPlusPlus` :
        FLARE++ attention layer.
    """

    def __init__(
        self,
        functional_dim: int,
        out_dim: int,
        embedding_dim: int | None = None,
        n_layers: int = 4,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 4,
        slice_num: int = 32,
        unified_pos: bool = False,
        ref: int = 8,
        structured_shape: None | tuple[int, ...] = None,
        time_input: bool = False,
        attn_scale: float | None = None,
    ) -> None:
        super().__init__(
            functional_dim=functional_dim,
            out_dim=out_dim,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
            unified_pos=unified_pos,
            ref=ref,
            structured_shape=structured_shape,
            use_te=False,
            time_input=time_input,
            plus=False,
        )

        self.blocks = _make_flare_blocks(
            attention_cls=FLAREPlusPlusAttention,
            n_layers=n_layers,
            n_head=n_head,
            n_hidden=n_hidden,
            dropout=dropout,
            act=act,
            mlp_ratio=mlp_ratio,
            out_dim=out_dim,
            n_global_queries=slice_num,
            attn_scale=(n_hidden // n_head) ** -0.5
            if attn_scale is None
            else attn_scale,
        )
        self.initialize_weights()
