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

r"""FlareJEPA: geometry-conditioned JEPA with a FLARE/GALE backbone.

A learned slot latent (Perceiver pooling over FLARE-refined geometry point
tokens), a GALE_FA target encoder (teacher, train only) + predictor with
geometry cross-attention, and a dual-read Perceiver-IO decoder (global slot
read + local kNN point read) under a JEPA training objective. Fixed input
shapes on the hot path.

Inference path: ``encode_geometry -> predict_latent -> decode_field``.
The target encoder runs at training time only, so condition sweeps over a
fixed geometry amortise the encode: only the (cheap) predictor + decoder
re-run per condition.
"""

from __future__ import annotations

import torch
from jaxtyping import Float

from physicsnemo.core.module import Module

from ._metadata import FlareJEPAMetaData
from .decoder import Decoder
from .encoders import GeometryEncoder, TargetEncoder
from .layers import CondEmbed
from .predictor import Predictor


class FlareJEPA(Module):
    r"""Top-level FlareJEPA model.

    Parameters mirror ``conf/model/flarejepa.yaml``. The nested
    ``geometry_encoder`` / ``target_encoder`` / ``predictor`` / ``decoder``
    arguments are plain dicts so the kwargs stay JSON-serialisable for
    ``.mdlus`` checkpointing.

    Parameters
    ----------
    slots : int
        Number of latent slots ``S``.
    token_dim, heads : int
        Latent width ``C`` and attention heads ``H`` (``C % H == 0``).
    cond_dim : int
        Raw global-condition dim (aoa, mach -> 2).
    cond_embed_dim : int
        Width of the Fourier+MLP condition embedding.
    out_dim : int
        Output field channels (Cp, Cf_tau, Cf_z -> 3).
    pe_bands : int
        Fourier bands for all positional encodings.
    use_normals, use_sdf : bool
        Extra per-point geometry channels; they set the encoder input dims
        (geometry: ``3 + 3*normals + 1*sdf``; target adds field channels).
    field_dim : int
        Field channels seen by the target encoder (defaults to ``out_dim``).
    geometry_encoder, target_encoder, predictor, decoder : dict or None
        Per-module overrides:
        ``{flare_layers, slot_layers, pool_repeats}``,
        ``{flare_layers, gale_layers, state_mixing_mode, context_cross}``,
        ``{gale_layers, state_mixing_mode, init_from_context}``,
        ``{cross_layers, query_chunk_size, head_mlp_ratio, point_read,
        point_neighbor_k, use_cond}``.
    mlp_ratio, dropout : int, float
        Shared across all sub-modules.
    share_slot_queries : bool
        Tie the teacher's slot-query bank to the geometry encoder's.
        Without tying the per-slot latent loss has NO slot correspondence —
        nothing defines which teacher slot student slot ``i`` must match,
        and the joint system can satisfy the loss with per-sample slot
        codes. With tying, slot ``i`` on both sides is "what query ``i``
        attends to".
    """

    def __init__(
        self,
        slots: int = 128,
        token_dim: int = 256,
        heads: int = 8,
        cond_dim: int = 2,
        cond_embed_dim: int = 64,
        out_dim: int = 3,
        pe_bands: int = 16,
        use_normals: bool = True,
        use_sdf: bool = False,
        field_dim: int | None = None,
        geometry_encoder: dict | None = None,
        target_encoder: dict | None = None,
        predictor: dict | None = None,
        decoder: dict | None = None,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        share_slot_queries: bool = False,
    ) -> None:
        super().__init__(meta=FlareJEPAMetaData())
        if token_dim % heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by heads ({heads})"
            )
        geometry_encoder = dict(geometry_encoder or {})
        target_encoder = dict(target_encoder or {})
        predictor = dict(predictor or {})
        decoder = dict(decoder or {})
        # Module.__new__ snapshots the ctor args as passed — under Hydra
        # without _convert_=all those are DictConfig objects, which would
        # only fail at .mdlus save time (json.dumps), AFTER training.
        # Re-record the plain-dict copies so serialisation cannot break.
        for _k, _v in (
            ("geometry_encoder", geometry_encoder),
            ("target_encoder", target_encoder),
            ("predictor", predictor),
            ("decoder", decoder),
        ):
            if _k in self._args.get("__args__", {}):
                self._args["__args__"][_k] = _v

        self.slots = slots
        self.out_dim = out_dim
        field_dim = out_dim if field_dim is None else field_dim
        geo_in_dim = 3 + (3 if use_normals else 0) + (1 if use_sdf else 0)
        tgt_in_dim = geo_in_dim + field_dim
        self.geo_in_dim = geo_in_dim
        self.tgt_in_dim = tgt_in_dim

        self.cond_embed = CondEmbed(cond_dim, cond_embed_dim)
        self.geometry_encoder = GeometryEncoder(
            in_dim=geo_in_dim,
            token_dim=token_dim,
            heads=heads,
            slots=slots,
            pe_bands=pe_bands,
            flare_layers=geometry_encoder.get("flare_layers", 4),
            slot_layers=geometry_encoder.get("slot_layers", 2),
            pool_repeats=geometry_encoder.get("pool_repeats", 1),
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.target_encoder = TargetEncoder(
            in_dim=tgt_in_dim,
            token_dim=token_dim,
            heads=heads,
            slots=slots,
            pe_bands=pe_bands,
            flare_layers=target_encoder.get("flare_layers", 0),
            gale_layers=target_encoder.get("gale_layers", 4),
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            state_mixing_mode=target_encoder.get("state_mixing_mode", "weighted"),
            context_cross=target_encoder.get("context_cross", True),
        )
        self.predictor = Predictor(
            token_dim=token_dim,
            heads=heads,
            slots=slots,
            gale_layers=predictor.get("gale_layers", 6),
            cond_embed_dim=cond_embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            state_mixing_mode=predictor.get("state_mixing_mode", "weighted"),
            init_from_context=predictor.get("init_from_context", True),
        )
        self.decoder = Decoder(
            token_dim=token_dim,
            heads=heads,
            out_dim=out_dim,
            pe_bands=pe_bands,
            cross_layers=decoder.get("cross_layers", 4),
            cond_embed_dim=cond_embed_dim,
            mlp_ratio=mlp_ratio,
            head_mlp_ratio=decoder.get("head_mlp_ratio", 1),
            dropout=dropout,
            query_chunk_size=decoder.get("query_chunk_size", 4096),
            point_read=decoder.get("point_read", False),
            point_neighbor_k=decoder.get("point_neighbor_k", 8),
        )
        self.decoder_point_read = bool(decoder.get("point_read", False))

        # Cond-blind decoder (the causal latent probe): zero the decoder's
        # condition embedding so (aoa, mach) can reach the field ONLY
        # through predictor -> Z_hat -> slot read. AdaLN then degenerates
        # to learned (condition-independent) gates; the point-read path
        # stays fully functional.
        self.decoder_use_cond = bool(decoder.get("use_cond", True))
        if share_slot_queries:
            self.target_encoder.slot_pool.slot_queries = (
                self.geometry_encoder.slot_pool.slot_queries
            )

    def encode_geometry(
        self,
        geometry_positions: Float[torch.Tensor, "B N_g 3"],
        geometry_features: Float[torch.Tensor, "B N_g F_g"] | None = None,
        return_point_tokens: bool = False,
    ):
        r"""Geometry -> slot latent ``Z_ctx``. Runs once per sample.

        With ``return_point_tokens=True`` also returns the pre-pool point
        tokens for the dual-read decoder (``decoder.point_read``).
        """
        return self.geometry_encoder(
            geometry_positions,
            geometry_features,
            return_point_tokens=return_point_tokens,
        )

    def encode_target(
        self,
        target_positions: Float[torch.Tensor, "B N_t 3"],
        target_features: Float[torch.Tensor, "B N_t F_t"],
        z_ctx: Float[torch.Tensor, "B S C"],
    ) -> Float[torch.Tensor, "B S C"]:
        r"""Field + geometry memory -> teacher latent ``Z_tgt`` (train only)."""
        return self.target_encoder(target_positions, target_features, z_ctx)

    def predict_latent(
        self,
        z_ctx: Float[torch.Tensor, "B S C"],
        cond: Float[torch.Tensor, "B cond_dim"],
    ) -> Float[torch.Tensor, "B S C"]:
        r"""Context latent + conditions -> predicted target latent ``Z_hat``."""
        return self.predictor(z_ctx, self.cond_embed(cond))

    def decode_field(
        self,
        z: Float[torch.Tensor, "B S C"],
        query_positions: Float[torch.Tensor, "B N_q 3"],
        cond: Float[torch.Tensor, "B cond_dim"],
        point_tokens: Float[torch.Tensor, "B N_p C"] | None = None,
        point_positions: Float[torch.Tensor, "B N_p 3"] | None = None,
    ) -> Float[torch.Tensor, "B N_q out_dim"]:
        r"""Latent + query positions -> field values."""
        ce = self.cond_embed(cond)
        if not self.decoder_use_cond:
            ce = ce * 0.0
        return self.decoder(
            z,
            query_positions,
            ce,
            point_tokens=point_tokens,
            point_positions=point_positions,
        )

    def decode_field_chunked(
        self,
        z: Float[torch.Tensor, "B S C"],
        query_positions: Float[torch.Tensor, "B N_q 3"],
        cond: Float[torch.Tensor, "B cond_dim"],
        chunk_size: int | None = None,
        point_tokens: Float[torch.Tensor, "B N_p C"] | None = None,
        point_positions: Float[torch.Tensor, "B N_p 3"] | None = None,
    ) -> Float[torch.Tensor, "B N_q out_dim"]:
        r"""Memory-bounded decode over a large query set (inference only)."""
        ce = self.cond_embed(cond)
        if not self.decoder_use_cond:
            ce = ce * 0.0
        return self.decoder.forward_chunked(
            z,
            query_positions,
            ce,
            chunk_size,
            point_tokens=point_tokens,
            point_positions=point_positions,
        )

    def forward(
        self,
        geometry_positions: Float[torch.Tensor, "B N_g 3"],
        target_positions: Float[torch.Tensor, "B N_t 3"],
        target_features: Float[torch.Tensor, "B N_t F_t"],
        query_positions: Float[torch.Tensor, "B N_q 3"],
        cond: Float[torch.Tensor, "B cond_dim"],
        geometry_features: Float[torch.Tensor, "B N_g F_g"] | None = None,
        decode_from: str = "target",
        run_target: bool = True,
        normalize_target: bool = True,
        also_decode_target: bool = False,
    ):
        r"""Training-step forward: returns ``(field_pred, Z_hat, Z_tgt)``.

        This is the SINGLE canonical composition of the training step — the
        recipe's ``train_flarejepa._forward_batch`` delegates here, so the
        tested path IS the training path.

        ``decode_from`` selects the latent fed to the decoder: ``"target"``
        (teacher-forced recon) or ``"predictor"`` (end-to-end path). With
        ``run_target=False`` the target encoder is skipped entirely
        (``Z_tgt`` is ``None``; requires ``decode_from="predictor"``).

        With ``normalize_target=True`` the teacher-forced decoder input is
        the CANONICAL LayerNorm-normalised teacher latent — the same space
        the latent loss regresses ``Z_hat`` toward — so inference-time
        ``decode(Z_hat)`` stays consistent with training. The returned
        ``Z_tgt`` is raw (for SIGReg); apply the same normalisation when
        building the latent-loss target. ``Z_tgt`` is returned WITHOUT
        stop-grad; the loss applies it.

        With ``also_decode_target=True`` a fourth output is returned: an
        auxiliary teacher-forced decode, so a reconstruction term can
        field-ground the teacher latent even when the main decode path is
        ``decode_from="predictor"``.
        """
        point_tokens = None
        if self.decoder_point_read:
            z_ctx, point_tokens = self.encode_geometry(
                geometry_positions, geometry_features, return_point_tokens=True
            )
        else:
            z_ctx = self.encode_geometry(geometry_positions, geometry_features)
        z_hat = self.predict_latent(z_ctx, cond)
        z_tgt = None
        if run_target:
            z_tgt = self.encode_target(target_positions, target_features, z_ctx)
        if decode_from == "target":
            if z_tgt is None:
                raise ValueError("decode_from='target' requires run_target=True")
            z_dec = (
                torch.nn.functional.layer_norm(z_tgt, z_tgt.shape[-1:])
                if normalize_target
                else z_tgt
            )
        elif decode_from == "predictor":
            z_dec = z_hat
        else:
            raise ValueError(
                f"decode_from must be 'target' or 'predictor', got {decode_from!r}"
            )
        point_positions = (
            geometry_positions if point_tokens is not None else None
        )
        field_pred = self.decode_field(
            z_dec,
            query_positions,
            cond,
            point_tokens=point_tokens,
            point_positions=point_positions,
        )
        if also_decode_target:
            if z_tgt is None:
                raise ValueError("also_decode_target requires run_target=True")
            z_dec_t = torch.nn.functional.layer_norm(z_tgt, z_tgt.shape[-1:])
            field_pred_teacher = self.decode_field(
                z_dec_t,
                query_positions,
                cond,
                point_tokens=point_tokens,
                point_positions=point_positions,
            )
            return field_pred, z_hat, z_tgt, field_pred_teacher
        return field_pred, z_hat, z_tgt

    @torch.no_grad()
    def predict(
        self,
        geometry_positions: Float[torch.Tensor, "B N_g 3"],
        query_positions: Float[torch.Tensor, "B N_q 3"],
        cond: Float[torch.Tensor, "B cond_dim"],
        geometry_features: Float[torch.Tensor, "B N_g F_g"] | None = None,
        chunk_size: int | None = None,
    ) -> Float[torch.Tensor, "B N_q out_dim"]:
        r"""Inference path: geometry -> context -> predictor -> decoder."""
        point_tokens = None
        if self.decoder_point_read:
            z_ctx, point_tokens = self.encode_geometry(
                geometry_positions, geometry_features, return_point_tokens=True
            )
        else:
            z_ctx = self.encode_geometry(geometry_positions, geometry_features)
        z_hat = self.predict_latent(z_ctx, cond)
        return self.decode_field_chunked(
            z_hat,
            query_positions,
            cond,
            chunk_size,
            point_tokens=point_tokens,
            point_positions=(
                geometry_positions if point_tokens is not None else None
            ),
        )
