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

"""Shape/contract tests for FlareJEPA on random inputs (CPU-friendly)."""

import os

import pytest
import torch

from physicsnemo.experimental.models.flarejepa import (
    FlareJEPA,
    FlareJEPAMetaData,
    reshape_context,
)

# Small config so the tests run fast on CPU.
B, N_G, N_T, N_Q = 2, 96, 64, 48
S, C, H = 16, 64, 4
COND_DIM, OUT_DIM = 2, 3


def small_model(use_normals: bool = False, **over) -> FlareJEPA:
    kw = dict(
        slots=S,
        token_dim=C,
        heads=H,
        cond_dim=COND_DIM,
        cond_embed_dim=32,
        out_dim=OUT_DIM,
        pe_bands=4,
        use_normals=use_normals,
        use_sdf=False,
        geometry_encoder={"flare_layers": 1, "slot_layers": 1},
        target_encoder={"gale_layers": 1},
        predictor={"gale_layers": 1},
        decoder={"cross_layers": 2, "query_chunk_size": 16},
        mlp_ratio=2,
    )
    kw.update(over)
    torch.manual_seed(0)
    return FlareJEPA(**kw)


def random_inputs(use_normals: bool = False):
    torch.manual_seed(1)
    geo_pos = torch.randn(B, N_G, 3)
    geo_feat = (
        torch.cat([geo_pos, torch.randn(B, N_G, 3)], dim=-1) if use_normals else None
    )
    tgt_pos = torch.randn(B, N_T, 3)
    tgt_in_dim = (6 if use_normals else 3) + OUT_DIM
    tgt_feat = torch.randn(B, N_T, tgt_in_dim)
    q_pos = torch.randn(B, N_Q, 3)
    cond = torch.randn(B, COND_DIM)
    return geo_pos, geo_feat, tgt_pos, tgt_feat, q_pos, cond


def test_metadata_present():
    model = small_model()
    assert isinstance(model.meta, FlareJEPAMetaData)


# --------------------------------------------------------------------------- #
# Core paths
# --------------------------------------------------------------------------- #


def test_encode_geometry_shape():
    model = small_model()
    geo_pos, *_ = random_inputs()
    z_ctx = model.encode_geometry(geo_pos)
    assert z_ctx.shape == (B, S, C)


def test_full_forward_shapes():
    model = small_model()
    geo_pos, geo_feat, tgt_pos, tgt_feat, q_pos, cond = random_inputs()
    field, z_hat, z_tgt = model(
        geo_pos, tgt_pos, tgt_feat, q_pos, cond, geometry_features=geo_feat
    )
    assert field.shape == (B, N_Q, OUT_DIM)
    # JEPA contract: predictor and teacher latents share layout.
    assert z_hat.shape == z_tgt.shape == (B, S, C)


def test_forward_with_normals():
    model = small_model(use_normals=True)
    geo_pos, geo_feat, tgt_pos, tgt_feat, q_pos, cond = random_inputs(
        use_normals=True
    )
    field, _, _ = model(
        geo_pos, tgt_pos, tgt_feat, q_pos, cond, geometry_features=geo_feat
    )
    assert field.shape == (B, N_Q, OUT_DIM)


def test_decode_from_predictor_path():
    model = small_model()
    geo_pos, _, tgt_pos, tgt_feat, q_pos, cond = random_inputs()
    field, _, _ = model(
        geo_pos, tgt_pos, tgt_feat, q_pos, cond, decode_from="predictor"
    )
    assert field.shape == (B, N_Q, OUT_DIM)
    with pytest.raises(ValueError):
        model(geo_pos, tgt_pos, tgt_feat, q_pos, cond, decode_from="nope")


def test_forward_teacher_forced_decodes_canonical_latent():
    # forward(decode_from="target") must feed the LayerNorm-normalised
    # teacher latent to the decoder — the same space the latent loss
    # regresses Z_hat toward — so inference decode(Z_hat) is consistent.
    model = small_model().eval()
    geo_pos, _, tgt_pos, tgt_feat, q_pos, cond = random_inputs()
    with torch.no_grad():
        field, _, z_tgt = model(
            geo_pos, tgt_pos, tgt_feat, q_pos, cond, decode_from="target"
        )
        canonical = torch.nn.functional.layer_norm(z_tgt, z_tgt.shape[-1:])
        expected = model.decode_field(canonical, q_pos, cond)
    assert torch.allclose(field, expected, atol=1e-6)


def test_forward_run_target_false_skips_teacher():
    # Supervised path: the target encoder must be skippable from the
    # canonical forward (single source of truth with the recipe).
    model = small_model()
    geo_pos, _, tgt_pos, tgt_feat, q_pos, cond = random_inputs()
    field, z_hat, z_tgt = model(
        geo_pos, tgt_pos, tgt_feat, q_pos, cond,
        decode_from="predictor", run_target=False,
    )
    assert field.shape == (B, N_Q, OUT_DIM)
    assert z_tgt is None
    with pytest.raises(ValueError):
        model(
            geo_pos, tgt_pos, tgt_feat, q_pos, cond,
            decode_from="target", run_target=False,
        )


def test_predictor_output_is_canonical_space():
    # Non-affine final norm: Z_hat must be token-wise zero-mean/unit-var —
    # natively in the same canonical space as the normalised teacher target.
    model = small_model().eval()
    geo_pos, _, _, _, _, cond = random_inputs()
    with torch.no_grad():
        z_ctx = model.encode_geometry(geo_pos)
        z_hat = model.predict_latent(z_ctx, cond)
    assert torch.allclose(z_hat.mean(dim=-1), torch.zeros(B, S), atol=1e-5)
    assert torch.allclose(
        z_hat.std(dim=-1, unbiased=False), torch.ones(B, S), atol=1e-2
    )


def test_chunked_decode_matches_full():
    model = small_model().eval()
    geo_pos, _, _, _, q_pos, cond = random_inputs()
    with torch.no_grad():
        z_ctx = model.encode_geometry(geo_pos)
        z_hat = model.predict_latent(z_ctx, cond)
        full = model.decode_field(z_hat, q_pos, cond)
        chunked = model.decode_field_chunked(z_hat, q_pos, cond, chunk_size=16)
    assert torch.allclose(full, chunked, atol=1e-4)


def test_inference_predict_path():
    model = small_model().eval()
    geo_pos, _, _, _, q_pos, cond = random_inputs()
    out = model.predict(geo_pos, q_pos, cond)
    assert out.shape == (B, N_Q, OUT_DIM)


def test_gradients_flow_to_all_submodules():
    # AdaLN-Zero gates are zero at init, which (by design) blocks gradient
    # THROUGH conditioned residual branches at step 0. Nudge the gates off
    # zero to test the fully-connected regime.
    from physicsnemo.experimental.models.flarejepa.layers import AdaLNZero

    model = small_model()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, AdaLNZero):
                m.proj[1].weight.normal_(std=0.02)
    geo_pos, _, tgt_pos, tgt_feat, q_pos, cond = random_inputs()
    field, z_hat, z_tgt = model(geo_pos, tgt_pos, tgt_feat, q_pos, cond)
    latent = torch.nn.functional.mse_loss(z_hat, z_tgt.detach())
    (field.pow(2).mean() + latent).backward()
    for name in ("geometry_encoder", "target_encoder", "predictor", "decoder"):
        grads = [
            p.grad.abs().sum()
            for p in getattr(model, name).parameters()
            if p.grad is not None
        ]
        assert grads and torch.stack(grads).sum() > 0, f"no grads in {name}"
    assert model.geometry_encoder.slot_pool.slot_queries.grad is not None
    assert model.geometry_encoder.slot_pool.slot_queries.grad.abs().sum() > 0
    assert model.predictor.slot_queries.grad is not None
    assert model.predictor.slot_queries.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# Layers / conditioning contracts
# --------------------------------------------------------------------------- #


def test_reshape_context_layout():
    z = torch.arange(B * S * C, dtype=torch.float32).reshape(B, S, C)
    ctx = reshape_context(z, H)
    assert ctx.shape == (B, H, S, C // H)
    # Head h of slot s must hold the h-th channel chunk of that slot.
    assert torch.equal(ctx[:, 1, 2], z[:, 2, C // H : 2 * C // H])


def test_conditioning_changes_output():
    # At init AdaLN-Zero is identity (zero gates), so cond has no effect by
    # design. Nudge every modulation projection off zero to verify the cond
    # plumbing actually reaches the output.
    from physicsnemo.experimental.models.flarejepa.layers import AdaLNZero

    model = small_model().eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, AdaLNZero):
                m.proj[1].weight.normal_(std=0.02)
    geo_pos, _, _, _, q_pos, cond = random_inputs()
    out1 = model.predict(geo_pos, q_pos, cond)
    out2 = model.predict(geo_pos, q_pos, cond + 1.0)
    assert not torch.allclose(out1, out2)


def test_adaln_zero_identity_at_init():
    # Zero-init contract: a conditioned block must be exactly identity at
    # init, so the cond-embedding path cannot destabilise early training.
    from physicsnemo.experimental.models.flarejepa.layers import (
        SlotSelfAttentionBlock,
    )

    torch.manual_seed(0)
    block = SlotSelfAttentionBlock(C, H, cond_embed_dim=32)
    x = torch.randn(B, S, C)
    out = block(x, cond_embed=torch.randn(B, 32))
    assert torch.allclose(out, x)


def test_decoder_head_mlp_ratio():
    m = small_model(decoder={"cross_layers": 1, "head_mlp_ratio": 4})
    hidden = [mod for mod in m.decoder.head.modules()
              if isinstance(mod, torch.nn.Linear)][0]
    assert hidden.out_features == C * 4


def test_target_encoder_pre_pool_blocks():
    m = small_model(target_encoder={"gale_layers": 1, "flare_layers": 2})
    assert len(m.target_encoder.point_blocks) == 2
    geo_pos, _, tgt_pos, tgt_feat, q_pos, cond = random_inputs()
    field, z_hat, z_tgt = m(geo_pos, tgt_pos, tgt_feat, q_pos, cond)
    assert z_tgt.shape == (B, S, C)


def test_pool_repeats_modules_and_forward():
    m = small_model(
        geometry_encoder={"flare_layers": 1, "slot_layers": 1, "pool_repeats": 3}
    )
    assert len(m.geometry_encoder.repool) == 2
    assert len(m.geometry_encoder.repool_slot_blocks) == 2
    geo_pos, _, tgt_pos, tgt_feat, q_pos, cond = random_inputs()
    field, z_hat, z_tgt = m(geo_pos, tgt_pos, tgt_feat, q_pos, cond)
    assert field.shape == (B, N_Q, OUT_DIM)
    with pytest.raises(ValueError):
        small_model(
            geometry_encoder={"flare_layers": 1, "slot_layers": 1,
                              "pool_repeats": 0}
        )


# --------------------------------------------------------------------------- #
# Slot-correspondence tying
# --------------------------------------------------------------------------- #


def test_share_slot_queries_ties_parameter():
    m = small_model(share_slot_queries=True)
    assert (
        m.target_encoder.slot_pool.slot_queries
        is m.geometry_encoder.slot_pool.slot_queries
    )


def test_share_slot_queries_off_keeps_separate_banks():
    m = small_model()
    assert (
        m.target_encoder.slot_pool.slot_queries
        is not m.geometry_encoder.slot_pool.slot_queries
    )


# --------------------------------------------------------------------------- #
# Dual-read decoder (kNN point read)
# --------------------------------------------------------------------------- #


def test_point_read_decoder_forward_backward():
    m = small_model(
        decoder={"cross_layers": 2, "query_chunk_size": 16,
                 "point_read": True, "point_neighbor_k": 4}
    )
    assert m.decoder.point_blocks is not None
    assert len(m.decoder.point_blocks) == 2
    geo_pos, _, tgt_pos, tgt_feat, q_pos, cond = random_inputs()
    field, z_hat, z_tgt = m(geo_pos, tgt_pos, tgt_feat, q_pos, cond)
    assert field.shape == (B, N_Q, OUT_DIM)
    (field.square().mean() + z_hat.square().mean()).backward()
    # Point tokens feed the decode: geometry embed must get decoder-loss grad.
    g = m.geometry_encoder.embed.feature_in.weight.grad
    assert g is not None and torch.isfinite(g).all()


def test_point_read_identity_at_init():
    # AdaLN-Zero gating: at init the point-read blocks must be exact
    # no-ops, so enabling point_read cannot perturb the base decode path.
    m = small_model(
        decoder={"cross_layers": 2, "query_chunk_size": 16, "point_read": True}
    ).eval()
    geo_pos, _, _, _, q_pos, cond = random_inputs()
    with torch.no_grad():
        z_ctx, pts = m.encode_geometry(geo_pos, return_point_tokens=True)
        z_hat = m.predict_latent(z_ctx, cond)
        a = m.decode_field(
            z_hat, q_pos, cond, point_tokens=pts, point_positions=geo_pos
        )
        b = m.decode_field(
            z_hat, q_pos, cond,
            point_tokens=torch.randn_like(pts), point_positions=geo_pos,
        )
    assert torch.allclose(a, b, atol=1e-5)


def test_point_read_requires_point_tokens():
    m = small_model(
        decoder={"cross_layers": 1, "query_chunk_size": 16, "point_read": True}
    )
    geo_pos, _, _, _, q_pos, cond = random_inputs()
    z = torch.randn(B, S, C)
    with pytest.raises(ValueError):
        m.decode_field(z, q_pos, cond)


def test_point_read_chunked_matches_full():
    m = small_model(
        decoder={"cross_layers": 2, "query_chunk_size": 16, "point_read": True}
    ).eval()
    geo_pos, _, _, _, q_pos, cond = random_inputs()
    with torch.no_grad():
        z_ctx, pts = m.encode_geometry(geo_pos, return_point_tokens=True)
        z_hat = m.predict_latent(z_ctx, cond)
        full = m.decode_field(
            z_hat, q_pos, cond, point_tokens=pts, point_positions=geo_pos
        )
        chunked = m.decode_field_chunked(
            z_hat, q_pos, cond, chunk_size=16,
            point_tokens=pts, point_positions=geo_pos,
        )
    assert torch.allclose(full, chunked, atol=1e-4)


def test_point_read_inference_predict_path():
    m = small_model(
        decoder={"cross_layers": 1, "query_chunk_size": 16, "point_read": True}
    ).eval()
    geo_pos, _, _, _, q_pos, cond = random_inputs()
    out = m.predict(geo_pos, q_pos, cond)
    assert out.shape == (B, N_Q, OUT_DIM)


# --------------------------------------------------------------------------- #
# JEPA latent kit: copy-proof teacher, teacher-forced decode, cond-blind
# --------------------------------------------------------------------------- #


def test_teacher_context_cross_false_builds_self_attn_teacher():
    from physicsnemo.experimental.models.flarejepa.layers import (
        SlotSelfAttentionBlock,
    )

    m = small_model(target_encoder={"gale_layers": 2, "context_cross": False})
    assert m.target_encoder.context_cross is False
    assert all(
        isinstance(b, SlotSelfAttentionBlock) for b in m.target_encoder.blocks
    )
    geo_pos, _, tgt_pos, tgt_feat, q_pos, cond = random_inputs()
    field, z_hat, z_tgt = m(geo_pos, tgt_pos, tgt_feat, q_pos, cond)
    assert z_tgt.shape == (B, S, C)
    # No-copy construction: z_tgt must NOT depend on z_ctx.
    with torch.no_grad():
        z1 = m.encode_target(tgt_pos, tgt_feat, torch.randn(B, S, C))
        z2 = m.encode_target(tgt_pos, tgt_feat, torch.randn(B, S, C))
    assert torch.allclose(z1, z2, atol=1e-6)


def test_teacher_context_cross_true_depends_on_z_ctx():
    m = small_model()
    geo_pos, _, tgt_pos, tgt_feat, _, _ = random_inputs()
    with torch.no_grad():
        z1 = m.encode_target(tgt_pos, tgt_feat, torch.randn(B, S, C))
        z2 = m.encode_target(tgt_pos, tgt_feat, torch.randn(B, S, C))
    assert not torch.allclose(z1, z2)


def test_also_decode_target_returns_four():
    m = small_model()
    # Decoder blocks are AdaLN-Zero: at init decode(z) ignores z entirely,
    # so field == field_tf trivially. Open the gates so the two decodes
    # actually consume their (different) latents.
    with torch.no_grad():
        for blk in m.decoder.blocks:
            blk.ada.proj[1].weight.fill_(0.01)
    geo_pos, _, tgt_pos, tgt_feat, q_pos, cond = random_inputs()
    out = m(
        geo_pos, tgt_pos, tgt_feat, q_pos, cond,
        decode_from="predictor", also_decode_target=True,
    )
    assert len(out) == 4
    field, z_hat, z_tgt, field_tf = out
    assert field_tf.shape == field.shape == (B, N_Q, OUT_DIM)
    # Teacher-forced decode consumes the canonical teacher latent — must
    # differ from the predictor decode of an untrained model.
    assert not torch.allclose(field, field_tf)
    with pytest.raises(ValueError):
        m(
            geo_pos, tgt_pos, tgt_feat, q_pos, cond,
            decode_from="predictor", run_target=False,
            also_decode_target=True,
        )


def test_cond_blind_decoder():
    # use_cond=false: the decoder output must be INVARIANT to cond (the
    # only cond route left is the predictor latent), even with decoder
    # gates opened; and the default model must still respond to cond.
    m = small_model(
        decoder={"cross_layers": 2, "query_chunk_size": 16, "use_cond": False}
    )
    with torch.no_grad():
        for blk in m.decoder.blocks:
            blk.ada.proj[1].weight.fill_(0.01)
    m = m.eval()
    geo_pos, _, _, _, q_pos, cond = random_inputs()
    with torch.no_grad():
        z = torch.randn(B, S, C)
        a = m.decode_field(z, q_pos, cond)
        b = m.decode_field(z, q_pos, cond + 5.0)
    assert torch.allclose(a, b, atol=1e-6)
    m2 = small_model(decoder={"cross_layers": 2, "query_chunk_size": 16})
    with torch.no_grad():
        for blk in m2.decoder.blocks:
            blk.ada.proj[1].weight.fill_(0.01)
    m2 = m2.eval()
    with torch.no_grad():
        a2 = m2.decode_field(z, q_pos, cond)
        b2 = m2.decode_field(z, q_pos, cond + 5.0)
    assert not torch.allclose(a2, b2)


# --------------------------------------------------------------------------- #
# Checkpoint compatibility (integration; needs a trained checkpoint on disk)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not os.environ.get("FLAREJEPA_CKPT"),
    reason="set FLAREJEPA_CKPT=/path/to/best.pt to run",
)
def test_flagship_checkpoint_loads():
    # The cleaned model must load flagship training checkpoints unchanged
    # (state-dict-key compatible), so published numbers stay reproducible.
    # Build from the checkpoint's own stored config, filtered to the
    # cleaned ctor signature (older configs carry since-removed keys).
    import inspect

    ckpt = torch.load(
        os.environ["FLAREJEPA_CKPT"], map_location="cpu", weights_only=False
    )
    stored = dict(ckpt["config"]["model"])
    accepted = set(inspect.signature(FlareJEPA.__init__).parameters)
    kwargs = {k: v for k, v in stored.items() if k in accepted}
    dropped = sorted(set(stored) - accepted - {"_target_", "_convert_"})
    if "decoder" in kwargs:
        dec_accepted = {
            "cross_layers", "query_chunk_size", "head_mlp_ratio",
            "point_read", "point_neighbor_k", "use_cond",
        }
        kwargs["decoder"] = {
            k: v for k, v in dict(kwargs["decoder"]).items()
            if k in dec_accepted
        }
    torch.manual_seed(0)
    model = FlareJEPA(**kwargs)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert not missing, f"missing keys: {missing[:8]} (dropped cfg: {dropped})"
    assert not unexpected, f"unexpected keys: {unexpected[:8]}"
