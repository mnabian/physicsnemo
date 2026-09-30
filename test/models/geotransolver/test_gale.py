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

import pytest
import torch

from physicsnemo.experimental.models.geotransolver import GALE_FPP
from physicsnemo.experimental.models.geotransolver.gale import (
    GALE,
    GALE_FA,
    GALE_block,
)
from physicsnemo.experimental.nn import FLAREPlusPlus

# =============================================================================
# GALE (Geometry-Aware Latent Embeddings) Attention Tests
# =============================================================================


def test_gale_forward_basic(device):
    """Test GALE attention layer forward pass without context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens = 100

    gale = GALE(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=dim_head,  # Must match dim_head for cross attention
    ).to(device)

    # Single input tensor wrapped in tuple
    x = torch.randn(batch_size, n_tokens, dim).to(device)

    outputs = gale((x,), context=None)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_forward_with_context(device):
    """Test GALE attention layer forward pass with cross-attention context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens = 100
    context_tokens = 32
    context_dim = dim_head

    gale = GALE(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
    ).to(device)

    x = torch.randn(batch_size, n_tokens, dim).to(device)
    context = torch.randn(batch_size, heads, context_tokens, context_dim).to(device)

    outputs = gale((x,), context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_forward_multiple_inputs(device):
    """Test GALE attention layer with multiple input tensors."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens_1 = 100
    n_tokens_2 = 150
    context_dim = dim_head

    gale = GALE(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
    ).to(device)

    x1 = torch.randn(batch_size, n_tokens_1, dim).to(device)
    x2 = torch.randn(batch_size, n_tokens_2, dim).to(device)

    outputs = gale((x1, x2), context=None)

    assert len(outputs) == 2
    assert outputs[0].shape == (batch_size, n_tokens_1, dim)
    assert outputs[1].shape == (batch_size, n_tokens_2, dim)
    assert not torch.isnan(outputs[0]).any()
    assert not torch.isnan(outputs[1]).any()


# =============================================================================
# GALE_FA Attention Tests
# =============================================================================


def test_gale_fa_forward_basic(device):
    """Test GALE_FA attention layer pass without context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    n_global_queries = 8
    batch_size = 2
    n_tokens = 100

    gale_fa = GALE_FA(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        n_global_queries=n_global_queries,
        use_te=False,
        context_dim=dim_head,  # Must match dim_head for cross attention
    ).to(device)

    # Single input tensor wrapped in tuple
    x = torch.randn(batch_size, n_tokens, dim).to(device)

    outputs = gale_fa((x,), context=None)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_fa_forward_with_context(device):
    """Test GALE_FA attention layer with cross-attention context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    n_global_queries = 8
    batch_size = 2
    n_tokens = 100
    context_tokens = 32
    context_dim = dim_head

    gale_fa = GALE_FA(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        n_global_queries=n_global_queries,
        use_te=False,
        context_dim=context_dim,
    ).to(device)

    x = torch.randn(batch_size, n_tokens, dim).to(device)
    context = torch.randn(batch_size, heads, context_tokens, context_dim).to(device)

    outputs = gale_fa((x,), context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_fa_forward_multiple_inputs(device):
    """Test GALE_FA attention layer with multiple input tensors."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    n_global_queries = 8
    batch_size = 2
    n_tokens_1 = 100
    n_tokens_2 = 150
    context_dim = dim_head

    gale_fa = GALE_FA(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        n_global_queries=n_global_queries,
        use_te=False,
        context_dim=context_dim,
    ).to(device)

    x1 = torch.randn(batch_size, n_tokens_1, dim).to(device)
    x2 = torch.randn(batch_size, n_tokens_2, dim).to(device)

    outputs = gale_fa((x1, x2), context=None)

    assert len(outputs) == 2
    assert outputs[0].shape == (batch_size, n_tokens_1, dim)
    assert outputs[1].shape == (batch_size, n_tokens_2, dim)
    assert not torch.isnan(outputs[0]).any()
    assert not torch.isnan(outputs[1]).any()


# =============================================================================
# GALE_FPP Attention Tests
# =============================================================================


@pytest.mark.parametrize("with_context", [False, True])
def test_gale_fpp_forward(device, with_context):
    """FLARE++ backend preserves token shape with optional Geo context."""
    torch.manual_seed(42)
    dim = 64
    heads = 4
    dim_head = 16
    batch_size = 2
    n_tokens = 100
    context_dim = dim_head

    attention = GALE_FPP(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        n_global_queries=8,
        use_te=False,
        context_dim=context_dim,
    ).to(device)
    x = torch.randn(batch_size, n_tokens, dim, device=device)
    context = (
        torch.randn(batch_size, heads, 8, context_dim, device=device)
        if with_context
        else None
    )

    (output,) = attention((x,), context=context)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_gale_fpp_forward_multiple_inputs(device):
    """Each GeoTransolver stream synthesizes its own dynamic routes."""
    torch.manual_seed(42)
    attention = GALE_FPP(
        dim=32,
        heads=4,
        dim_head=8,
        n_global_queries=6,
        use_te=False,
        context_dim=8,
    ).to(device)
    x1 = torch.randn(2, 31, 32, device=device)
    x2 = torch.randn(2, 47, 32, device=device)
    context = torch.randn(2, 4, 6, 8, device=device)

    outputs = attention((x1, x2), context=context)

    assert [output.shape for output in outputs] == [x1.shape, x2.shape]
    assert all(torch.isfinite(output).all() for output in outputs)


def test_gale_fpp_backward_with_context(device):
    """Gradients reach dynamic routing, context projections, and both inputs."""
    torch.manual_seed(9)
    attention = GALE_FPP(
        dim=32,
        heads=4,
        dim_head=8,
        n_global_queries=6,
        use_te=False,
        context_dim=8,
    ).to(device)
    x = torch.randn(2, 19, 32, device=device, requires_grad=True)
    context = torch.randn(2, 4, 7, 8, device=device, requires_grad=True)

    (output,) = attention((x,), context=context)
    output.square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    for parameter in (
        attention.q_seed,
        attention.query_synthesis_k.weight,
        attention.query_synthesis_v.weight,
        attention.cross_k.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_gale_fpp_matches_standalone_flare_plus_plus(device):
    """The Geo backend must reuse the standalone FLARE++ self-attention contract."""
    torch.manual_seed(7)
    kwargs = {
        "dim": 32,
        "heads": 4,
        "dim_head": 8,
        "n_global_queries": 6,
        "dropout": 0.0,
        "use_te": False,
    }
    standalone = FLAREPlusPlus(**kwargs).to(device).eval()
    backend = GALE_FPP(**kwargs, context_dim=0).to(device).eval()
    backend.load_state_dict(standalone.state_dict(), strict=True)
    x = torch.randn(2, 29, 32, device=device)

    expected = standalone(x)
    (actual,) = backend((x,), context=None)

    torch.testing.assert_close(actual, expected)


def test_gale_fpp_scale_and_te_contract():
    """FLARE++ backend defaults to paper scaling and explicitly rejects TE."""
    attention = GALE_FPP(dim=32, heads=4, dim_head=8, use_te=False)
    assert attention.scale == pytest.approx(8**-0.5)

    with pytest.raises(ValueError, match="does not support Transformer Engine"):
        GALE_FPP(dim=32, heads=4, dim_head=8, use_te=True)

    for bad_scale in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="attn_scale"):
            GALE_FPP(
                dim=32,
                heads=4,
                dim_head=8,
                use_te=False,
                attn_scale=bad_scale,
            )


# =============================================================================
# concat_project state mixing mode
# =============================================================================


def test_gale_concat_project_forward(device):
    """Test GALE with state_mixing_mode='concat_project' and cross-attention context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    slice_num = 8
    batch_size = 2
    n_tokens = 100
    context_tokens = 32
    context_dim = dim_head

    gale = GALE(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
        state_mixing_mode="concat_project",
    ).to(device)

    x = torch.randn(batch_size, n_tokens, dim).to(device)
    context = torch.randn(batch_size, heads, context_tokens, context_dim).to(device)

    outputs = gale((x,), context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


def test_gale_fa_concat_project_forward(device):
    """Test GALE_FA with state_mixing_mode='concat_project' and cross-attention context."""
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = 16
    n_global_queries = 8
    batch_size = 2
    n_tokens = 100
    context_tokens = 32
    context_dim = dim_head

    gale_fa = GALE_FA(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        n_global_queries=n_global_queries,
        use_te=False,
        context_dim=context_dim,
        state_mixing_mode="concat_project",
    ).to(device)

    x = torch.randn(batch_size, n_tokens, dim).to(device)
    context = torch.randn(batch_size, heads, context_tokens, context_dim).to(device)

    outputs = gale_fa((x,), context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, dim)
    assert not torch.isnan(outputs[0]).any()


# =============================================================================
# GALE_block Tests
# =============================================================================


@pytest.mark.parametrize("attention_type", ["GALE", "GALE_FA", "GALE_FPP"])
def test_gale_block_forward(device, attention_type):
    """Test GALE_block forward pass for every attention backend."""
    torch.manual_seed(42)

    hidden_dim = 64
    n_head = 4
    batch_size = 2
    n_tokens = 100
    slice_num = 8
    context_dim = hidden_dim // n_head

    block = GALE_block(
        num_heads=n_head,
        hidden_dim=hidden_dim,
        dropout=0.0,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
        attention_type=attention_type,
    ).to(device)

    x = torch.randn(batch_size, n_tokens, hidden_dim).to(device)
    context = torch.randn(batch_size, n_head, slice_num, context_dim).to(device)

    outputs = block((x,), global_context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, hidden_dim)
    assert not torch.isnan(outputs[0]).any()


@pytest.mark.parametrize("attention_type", ["GALE", "GALE_FA", "GALE_FPP"])
def test_gale_block_multiple_inputs(device, attention_type):
    """Test every GALE_block backend with multiple input tensors."""
    torch.manual_seed(42)

    hidden_dim = 64
    n_head = 4
    batch_size = 2
    n_tokens_1 = 100
    n_tokens_2 = 150
    slice_num = 8
    context_dim = hidden_dim // n_head

    block = GALE_block(
        num_heads=n_head,
        hidden_dim=hidden_dim,
        dropout=0.0,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
        attention_type=attention_type,
    ).to(device)

    x1 = torch.randn(batch_size, n_tokens_1, hidden_dim).to(device)
    x2 = torch.randn(batch_size, n_tokens_2, hidden_dim).to(device)
    context = torch.randn(batch_size, n_head, slice_num, context_dim).to(device)

    outputs = block((x1, x2), global_context=context)

    assert len(outputs) == 2
    assert outputs[0].shape == (batch_size, n_tokens_1, hidden_dim)
    assert outputs[1].shape == (batch_size, n_tokens_2, hidden_dim)


@pytest.mark.parametrize("attention_type", ["GALE", "GALE_FA", "GALE_FPP"])
def test_gale_block_concat_project(device, attention_type):
    """Test GALE_block with state_mixing_mode='concat_project'."""
    torch.manual_seed(42)

    hidden_dim = 64
    n_head = 4
    batch_size = 2
    n_tokens = 100
    slice_num = 8
    context_dim = hidden_dim // n_head

    block = GALE_block(
        num_heads=n_head,
        hidden_dim=hidden_dim,
        dropout=0.0,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=slice_num,
        use_te=False,
        plus=False,
        context_dim=context_dim,
        attention_type=attention_type,
        state_mixing_mode="concat_project",
    ).to(device)

    x = torch.randn(batch_size, n_tokens, hidden_dim).to(device)
    context = torch.randn(batch_size, n_head, slice_num, context_dim).to(device)

    outputs = block((x,), global_context=context)

    assert len(outputs) == 1
    assert outputs[0].shape == (batch_size, n_tokens, hidden_dim)
    assert not torch.isnan(outputs[0]).any()
