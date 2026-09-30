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

"""Tests for FLARE attention layer."""

import pytest
import torch
import torch.nn.functional as F

from physicsnemo.experimental.nn import FLARE, FLAREPlusPlus


def test_flare_forward(device):
    """Test FLARE forward pass and output shape."""
    torch.manual_seed(42)
    flare = FLARE(dim=64, heads=4, dim_head=16, n_global_queries=32, use_te=False).to(
        device
    )
    x = torch.randn(2, 100, 64).to(device)
    out = flare(x)
    assert out.shape == (2, 100, 64)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("heads,dim_head", [(2, 32), (8, 8), (4, 16)])
def test_flare_configs(device, heads, dim_head):
    """Test FLARE with different head configurations."""
    torch.manual_seed(42)
    dim = heads * dim_head
    flare = FLARE(
        dim=dim, heads=heads, dim_head=dim_head, n_global_queries=16, use_te=False
    ).to(device)
    x = torch.randn(2, 50, dim).to(device)
    out = flare(x)
    assert out.shape == x.shape


@pytest.mark.parametrize("attention_cls", [FLARE, FLAREPlusPlus])
def test_flare_use_te_raises(attention_cls):
    """Test that the FLARE-family layers reject Transformer Engine."""
    with pytest.raises(ValueError, match="does not support Transformer Engine"):
        attention_cls(dim=64, heads=4, dim_head=16, use_te=True)


def test_flare_gradient_flow(device):
    """Test gradient flow through FLARE."""
    torch.manual_seed(42)
    flare = FLARE(dim=32, heads=4, dim_head=8, use_te=False).to(device)
    x = torch.randn(2, 20, 32, device=device, requires_grad=True)
    out = flare(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_flare_plus_plus_forward(device):
    """Test FLARE++ forward pass and output shape."""
    torch.manual_seed(42)
    flare = FLAREPlusPlus(
        dim=64,
        heads=4,
        dim_head=16,
        n_global_queries=32,
        use_te=False,
    ).to(device)
    x = torch.randn(2, 100, 64, device=device)
    out = flare(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_flare_plus_plus_matches_three_sdpa_reference(device):
    """Check FLARE++ against the three-SDPA formulation in the paper."""
    torch.manual_seed(7)
    batch_size, n_tokens, dim = 2, 11, 24
    heads, dim_head, n_queries = 3, 8, 5
    flare = FLAREPlusPlus(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        n_global_queries=n_queries,
        dropout=0.0,
        use_te=False,
    ).to(device)
    flare.eval()
    x = torch.randn(batch_size, n_tokens, dim, device=device)

    actual = flare(x)
    assert flare.scale == pytest.approx(dim_head**-0.5)

    def project_to_heads(projection):
        return (
            projection(x).reshape(batch_size, n_tokens, heads, dim_head).transpose(1, 2)
        )

    x_mid = project_to_heads(flare.in_project_x)
    query_k = project_to_heads(flare.query_synthesis_k)
    query_v = project_to_heads(flare.query_synthesis_v)
    seeds = flare.q_seed.expand(batch_size, -1, -1, -1)
    queries = F.scaled_dot_product_attention(seeds, query_k, query_v)
    keys = flare.self_k(x_mid)
    values = flare.self_v(x_mid)
    latent_values = F.scaled_dot_product_attention(queries, keys, values)
    expected = F.scaled_dot_product_attention(keys, queries, latent_values)
    expected = expected.transpose(1, 2).reshape(batch_size, n_tokens, dim)
    expected = flare.out_dropout(flare.out_linear(expected))

    torch.testing.assert_close(actual, expected)


def test_flare_plus_plus_attention_scale_contract():
    """FLARE++ defaults to paper scaling and permits an explicit override."""
    fixed = FLARE(dim=32, heads=4, dim_head=8, use_te=False)
    paper = FLAREPlusPlus(dim=32, heads=4, dim_head=8, use_te=False)
    overridden = FLAREPlusPlus(
        dim=32,
        heads=4,
        dim_head=8,
        use_te=False,
        attn_scale=1.0,
    )

    assert fixed.scale == 1.0
    assert paper.scale == pytest.approx(8**-0.5)
    assert overridden.scale == 1.0

    for bad_scale in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="attn_scale"):
            FLAREPlusPlus(
                dim=32,
                heads=4,
                dim_head=8,
                use_te=False,
                attn_scale=bad_scale,
            )


def test_flare_plus_plus_routing_queries_are_input_conditioned(device):
    """Different samples should synthesize different routing templates."""
    torch.manual_seed(11)
    flare = FLAREPlusPlus(
        dim=16,
        heads=2,
        dim_head=8,
        n_global_queries=4,
        use_te=False,
    ).to(device)
    x = torch.stack(
        [
            torch.zeros(9, 16, device=device),
            torch.ones(9, 16, device=device),
        ]
    )
    query_k = flare.query_synthesis_k(x).reshape(2, 9, 2, 8).transpose(1, 2)
    query_v = flare.query_synthesis_v(x).reshape(2, 9, 2, 8).transpose(1, 2)
    queries = F.scaled_dot_product_attention(
        flare.q_seed.expand(2, -1, -1, -1),
        query_k,
        query_v,
        scale=flare.scale,
    )

    assert not torch.allclose(queries[0], queries[1])


def test_flare_plus_plus_adds_only_query_synthesis_projections():
    """FLARE++ should add exactly two warm-startable projection layers."""
    dim = 32
    flare = FLARE(dim=dim, heads=4, dim_head=8, n_global_queries=6, use_te=False)
    flare_pp = FLAREPlusPlus(
        dim=dim, heads=4, dim_head=8, n_global_queries=6, use_te=False
    )

    n_flare = sum(parameter.numel() for parameter in flare.parameters())
    n_flare_pp = sum(parameter.numel() for parameter in flare_pp.parameters())
    expected_difference = 2 * (dim * dim + dim)
    assert n_flare_pp - n_flare == expected_difference

    incompatible_keys = flare_pp.load_state_dict(flare.state_dict(), strict=False)
    assert incompatible_keys.unexpected_keys == []
    assert set(incompatible_keys.missing_keys) == {
        "query_synthesis_k.bias",
        "query_synthesis_k.weight",
        "query_synthesis_v.bias",
        "query_synthesis_v.weight",
    }


def test_flare_plus_plus_gradient_flow(device):
    """Gradients should reach inputs, seeds, and both synthesis projections."""
    torch.manual_seed(42)
    flare = FLAREPlusPlus(
        dim=32,
        heads=4,
        dim_head=8,
        n_global_queries=7,
        use_te=False,
    ).to(device)
    x = torch.randn(2, 20, 32, device=device, requires_grad=True)
    flare(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in (
        flare.q_seed,
        flare.query_synthesis_k.weight,
        flare.query_synthesis_v.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
