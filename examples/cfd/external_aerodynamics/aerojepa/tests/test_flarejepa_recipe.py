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

"""Tests for the FlareJEPA recipe pieces: grid normals, the datapipe's
``include_normals`` path, and the dense-batched training-step helpers."""

import json

import numpy as np
import pytest
import torch

from src.datapipes import (
    SuperWingDataset,
    compute_grid_normals,
    superwing_collate,
)
from src.datapipes.superwing import SUPERWING_GRID_SHAPE

H, W = SUPERWING_GRID_SHAPE
N_GRID = H * W


# --------------------------------------------------------------------------- #
# compute_grid_normals
# --------------------------------------------------------------------------- #


def test_grid_normals_flat_plane():
    # A flat plane z=0: every normal must be +/- z-hat.
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    geom = np.stack(
        [ii.astype(np.float64), jj.astype(np.float64), np.zeros((H, W))], axis=0
    )
    n = compute_grid_normals(geom)
    assert n.shape == (N_GRID, 3)
    assert np.allclose(np.abs(n[:, 2]), 1.0, atol=1e-6)
    assert np.allclose(n[:, :2], 0.0, atol=1e-6)
    # Sign must be consistent across the whole grid.
    assert np.all(n[:, 2] == n[0, 2])


def test_grid_normals_sphere_patch_radial():
    # A patch of the unit sphere: normals must be radial (parallel to xyz).
    theta = np.linspace(0.4, 1.2, H)
    phi = np.linspace(0.3, 1.5, W)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    geom = np.stack(
        [np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], axis=0
    )
    n = compute_grid_normals(geom)
    xyz = geom.reshape(3, -1).T
    # |cos| of angle between normal and radial direction ~ 1 in the interior
    # (edges use one-sided differences and are slightly less accurate).
    cos = np.abs((n * xyz).sum(-1) / np.linalg.norm(xyz, axis=-1))
    interior = np.ones((H, W), dtype=bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    assert cos[interior.reshape(-1)].min() > 0.999


def test_grid_normals_degenerate_cells_are_zero():
    # A grid collapsed to a single point has zero-area cells everywhere.
    geom = np.ones((3, H, W))
    n = compute_grid_normals(geom)
    assert np.allclose(n, 0.0)


def test_grid_normals_rejects_bad_shape():
    with pytest.raises(ValueError):
        compute_grid_normals(np.zeros((2, H, W)))


# --------------------------------------------------------------------------- #
# SuperWingDataset include_normals (synthetic files)
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_superwing(tmp_path):
    """Minimal on-disk SuperWing layout: 2 geometries x 2 conditions."""
    n_shape, n_samples = 2, 4
    rng = np.random.default_rng(0)
    geom = rng.normal(size=(n_shape, 3, H, W)).astype(np.float32)
    data = rng.normal(size=(n_samples, 3, H, W)).astype(np.float32)
    index = np.zeros((n_samples, 12), dtype=np.float32)
    index[:, 0] = [0, 0, 1, 1]  # geom_idx
    index[:, 1] = [0, 1, 0, 1]  # cond_idx
    index[:, 2] = rng.uniform(0, 10, n_samples)  # aoa
    index[:, 3] = rng.uniform(0.6, 0.9, n_samples)  # mach
    np.save(tmp_path / "geom0.npy", geom)
    np.save(tmp_path / "data.npy", data)
    np.save(tmp_path / "index.npy", index)

    manifest = {
        "train_sample_idx": [0, 1, 2],
        "val_sample_idx": [3],
        "test_sample_idx": [],
    }
    (tmp_path / "split.json").write_text(json.dumps(manifest))

    stats = {
        "schema": "superwing",
        "target_mean": [0.0, 0.0, 0.0],
        "target_std": [1.0, 1.0, 1.0],
        "gen_params_columns": [2, 3],
        "gen_params_names": ["aoa", "mach"],
        "gen_params_mean": [5.0, 0.75],
        "gen_params_std": [3.0, 0.1],
        "xyz_min": [-4.0, -4.0, -4.0],
        "xyz_max": [4.0, 4.0, 4.0],
    }
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    return tmp_path


def _make_dataset(root, **kwargs):
    defaults = dict(
        root_dir=str(root),
        split="train",
        split_manifest_path=str(root / "split.json"),
        normalization_stats_path=str(root / "stats.json"),
        surface_points=64,
        target_encoder_points=32,
        query_points=16,
        eval_full_grid_query=False,
        deterministic_sampling=True,
    )
    defaults.update(kwargs)
    return SuperWingDataset(**defaults)


def test_dataset_emits_normals(synthetic_superwing):
    ds = _make_dataset(synthetic_superwing, include_normals=True)
    sample = ds[0]
    assert sample["context_normals"].shape == (64, 3)
    assert sample["target_surface_normals"].shape == (32, 3)
    # Unit (or zero for degenerate cells) vectors.
    norms = sample["context_normals"].norm(dim=-1)
    assert torch.all((norms - 1.0).abs() < 1e-4)
    # Normals are aligned with the same subsample as the positions:
    # rebuilding from the full grid at the dataset's deterministic indices
    # must reproduce them.
    ds._ensure_loaded()
    geom_chw = np.asarray(ds._geom_mm[0], dtype=np.float32)
    full = compute_grid_normals(geom_chw)
    rng = ds._make_rng(0)
    s_idx = ds._sample_indices(rng, 64)
    assert np.allclose(sample["context_normals"].numpy(), full[s_idx])


def test_dataset_without_normals_unchanged(synthetic_superwing):
    ds = _make_dataset(synthetic_superwing, include_normals=False)
    sample = ds[0]
    assert "context_normals" not in sample
    assert "target_surface_normals" not in sample


def test_collate_with_normals(synthetic_superwing):
    ds = _make_dataset(synthetic_superwing, include_normals=True)
    batch = superwing_collate([ds[0], ds[1]])
    assert batch["context_normals"].shape == (2, 64, 3)
    assert batch["target_surface_normals"].shape == (2, 32, 3)
    assert batch["context_pos"].shape == (2, 64, 3)


# --------------------------------------------------------------------------- #
# train_flarejepa helpers
# --------------------------------------------------------------------------- #


def _small_flarejepa():
    from physicsnemo.experimental.models.flarejepa import FlareJEPA

    torch.manual_seed(0)
    return FlareJEPA(
        slots=8,
        token_dim=32,
        heads=4,
        cond_dim=2,
        cond_embed_dim=16,
        out_dim=3,
        pe_bands=2,
        use_normals=True,
        geometry_encoder={"flare_layers": 1, "slot_layers": 1},
        target_encoder={"gale_layers": 1},
        predictor={"gale_layers": 1},
        decoder={"cross_layers": 1},
        mlp_ratio=1,
    )


def _fake_batch(B=2, n_ctx=48, n_tgt=24, n_q=16, with_normals=True):
    torch.manual_seed(1)
    batch = {
        "context_pos": torch.randn(B, n_ctx, 3),
        "target_surface_pos": torch.randn(B, n_tgt, 3),
        "target_surface_main_feat": torch.randn(B, n_tgt, 6),
        "query_pos": torch.randn(B, n_q, 3),
        "query_target": torch.randn(B, n_q, 3),
        "gen_params": torch.randn(B, 2),
    }
    if with_normals:
        batch["context_normals"] = torch.nn.functional.normalize(
            torch.randn(B, n_ctx, 3), dim=-1
        )
        batch["target_surface_normals"] = torch.nn.functional.normalize(
            torch.randn(B, n_tgt, 3), dim=-1
        )
    return batch


def test_assemble_features_ordering():
    from train_flarejepa import _assemble_features

    batch = _fake_batch()
    geo_feat, tgt_feat = _assemble_features(batch, use_normals=True)
    assert geo_feat.shape[-1] == 6
    assert tgt_feat.shape[-1] == 9
    # Layout contracts: geo=[xyz, normals]; tgt=[xyz, normals, field].
    assert torch.equal(geo_feat[..., :3], batch["context_pos"])
    assert torch.equal(geo_feat[..., 3:], batch["context_normals"])
    assert torch.equal(tgt_feat[..., :3], batch["target_surface_main_feat"][..., :3])
    assert torch.equal(tgt_feat[..., 3:6], batch["target_surface_normals"])
    assert torch.equal(tgt_feat[..., 6:], batch["target_surface_main_feat"][..., 3:])


def test_assemble_features_missing_normals_raises():
    from train_flarejepa import _assemble_features

    with pytest.raises(KeyError):
        _assemble_features(_fake_batch(with_normals=False), use_normals=True)


def test_forward_batch_supervised_path():
    from train_flarejepa import _forward_batch

    model = _small_flarejepa()
    batch = _fake_batch()
    out = _forward_batch(
        model,
        batch,
        use_normals=True,
        run_target=False,
        decode_from="predictor",
        normalize_target=True,
    )
    assert out["field_pred"].shape == (2, 16, 3)
    assert out["z_tgt"] is None and out["z_tgt_canonical"] is None


def test_forward_batch_teacher_forced_uses_canonical_latent():
    from train_flarejepa import _forward_batch

    model = _small_flarejepa()
    batch = _fake_batch()
    out = _forward_batch(
        model,
        batch,
        use_normals=True,
        run_target=True,
        decode_from="target",
        normalize_target=True,
    )
    # The canonical latent is the LayerNorm of the raw teacher latent —
    # the SAME tensor the latent loss regresses and the decoder consumed.
    expected = torch.nn.functional.layer_norm(
        out["z_tgt"], out["z_tgt"].shape[-1:]
    )
    assert torch.allclose(out["z_tgt_canonical"], expected, atol=1e-6)
    # decode_from=target without a target latent must fail loudly.
    with pytest.raises(ValueError):
        _forward_batch(
            model,
            batch,
            use_normals=True,
            run_target=False,
            decode_from="target",
            normalize_target=True,
        )


def test_phase2_config_composes_with_inheritance():
    import os

    from hydra import compose, initialize_config_dir

    conf_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "conf")
    )
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(
            config_name="config_flarejepa",
            overrides=[
                "training=superwing_flarejepa_phase2",
                "data.path=/nonexistent",
            ],
        )
    # Phase-2 overrides.
    assert cfg.training.decode_from == "target"
    assert float(cfg.training.loss.latent.weight) == 1.0
    assert float(cfg.training.loss.sigreg.weight) == pytest.approx(1e-2)
    # Inherited from the Phase-1 schedule (defaults composition intact).
    assert bool(cfg.training.loss.latent.stop_grad) is True
    assert bool(cfg.training.loss.latent.normalize_target) is True
    assert int(cfg.training.epochs) == 200
    # Entry-point config still forces normals emission.
    assert bool(cfg.data.include_normals) is True


def test_total_loss_stop_grad_blocks_teacher_gradient():
    from omegaconf import OmegaConf

    from src.losses import build_recon_loss_from_config, build_sigreg_from_config
    from train_flarejepa import _compute_total_loss, _forward_batch

    loss_cfg = OmegaConf.create(
        {
            "recon": {
                "kind": "relative_l2_mse",
                "eps": 1e-2,
                "relative_l2_weight": 1.0,
                "mse_weight": 0.2,
            },
            "latent": {
                "mse_weight": 0.5,
                "cosine_weight": 0.5,
                "stop_grad": True,
                "normalize_target": True,
            },
            "sigreg": {"knots": 5, "num_proj": 8},
        }
    )
    model = _small_flarejepa()
    batch = _fake_batch()
    out = _forward_batch(
        model,
        batch,
        use_normals=True,
        run_target=True,
        decode_from="predictor",
        normalize_target=True,
    )
    weights = {"recon": 0.0, "latent": 1.0, "sigreg": 0.0}
    total, parts = _compute_total_loss(
        out,
        batch,
        recon_loss_fn=build_recon_loss_from_config(loss_cfg.recon),
        sigreg_loss_fn=build_sigreg_from_config(loss_cfg.sigreg),
        loss_cfg=loss_cfg,
        term_weights=weights,
    )
    total.backward()
    # With stop_grad + zero recon/sigreg weights, the latent loss must reach
    # the predictor but NOT the target encoder.
    pred_grads = sum(
        p.grad.abs().sum().item()
        for p in model.predictor.parameters()
        if p.grad is not None
    )
    tgt_grads = sum(
        p.grad.abs().sum().item()
        for p in model.target_encoder.parameters()
        if p.grad is not None
    )
    assert pred_grads > 0
    assert tgt_grads == 0


def test_optimizer_no_decay_groups():
    from omegaconf import OmegaConf

    from src.training import build_optimizer

    model = _small_flarejepa()
    cfg = OmegaConf.create({
        "_target_": "torch.optim.AdamW", "lr": 1e-3,
        "weight_decay": 1e-3, "no_decay_norms_and_gains": True,
    })
    opt = build_optimizer(model, cfg)
    assert len(opt.param_groups) == 2
    decay, no_decay = opt.param_groups
    assert no_decay["weight_decay"] == 0.0
    assert decay["weight_decay"] == 1e-3
    # Every 1-D param (norm gains, biases, gates) must be in the no-decay set.
    no_decay_ids = {id(p) for p in no_decay["params"]}
    for name, p in model.named_parameters():
        if p.ndim <= 1 or "slot_queries" in name or "q_global" in name:
            assert id(p) in no_decay_ids, name
    # Flag off -> single group, unchanged legacy behaviour.
    cfg2 = OmegaConf.create({
        "_target_": "torch.optim.AdamW", "lr": 1e-3, "weight_decay": 1e-3,
    })
    assert len(build_optimizer(model, cfg2).param_groups) == 1


def test_teacher_forced_recon_term_and_per_slot_sigreg():
    from omegaconf import OmegaConf

    from train_flarejepa import _compute_total_loss, _forward_batch
    from src.losses.builders import (
        build_recon_loss_from_config,
        build_sigreg_from_config,
    )

    model = _small_flarejepa()
    # Decoder blocks are AdaLN-Zero: at init d(field)/d(latent) == 0, so
    # the teacher-forced recon gradient to the teacher is exactly zero.
    # Open the gates so this test checks CONNECTIVITY, not init state.
    with torch.no_grad():
        for blk in model.decoder.blocks:
            blk.ada.proj[1].weight.fill_(0.01)
    batch = _fake_batch()
    loss_cfg = OmegaConf.create({
        "recon": {"kind": "relative_l2_mse", "relative_mse_mode": "pointwise",
                  "eps": 1e-2, "relative_l2_weight": 1.0, "mse_weight": 0.2,
                  "teacher_forced_weight": 0.3},
        "latent": {"stop_grad": False, "mse_weight": 0.5,
                   "cosine_weight": 0.5},
        "sigreg": {"per_slot": True, "knots": 17, "num_proj": 64},
    })
    outputs = _forward_batch(
        model, batch, use_normals=True, run_target=True,
        decode_from="predictor", normalize_target=True,
        teacher_forced_decode=True,
    )
    assert outputs["field_pred_teacher"] is not None
    recon_fn = build_recon_loss_from_config(loss_cfg.recon)
    sigreg_fn = build_sigreg_from_config(loss_cfg.sigreg)
    total, parts = _compute_total_loss(
        outputs, batch, recon_loss_fn=recon_fn, sigreg_loss_fn=sigreg_fn,
        loss_cfg=loss_cfg,
        term_weights={"recon": 1.0, "latent": 1.0, "sigreg": 0.01},
    )
    assert parts["recon_tf"].item() > 0.0
    assert torch.isfinite(total)
    # Teacher must now receive gradient from the tf-recon term even with a
    # detached latent target: rebuild with stop_grad=True to isolate it.
    loss_cfg.latent.stop_grad = True
    total2, _ = _compute_total_loss(
        outputs, batch, recon_loss_fn=recon_fn, sigreg_loss_fn=sigreg_fn,
        loss_cfg=loss_cfg,
        term_weights={"recon": 1.0, "latent": 1.0, "sigreg": 0.0},
    )
    model.zero_grad(set_to_none=True)
    total2.backward()
    g = next(
        p.grad for n, p in model.named_parameters()
        if n.startswith("target_encoder") and p.grad is not None
    )
    assert g.abs().sum() > 0


def test_per_slot_sigreg_sees_per_slot_constants():
    # Audit C1 regression: the folded default flattens (B,S,C) into one
    # token pool and is blind to per-slot-constant collapse; the per_slot
    # path must NOT be equivalent (the wrapped transpose used to be a
    # silent no-op).
    from omegaconf import OmegaConf

    from src.losses.builders import build_sigreg_from_config
    from train_flarejepa import _compute_total_loss, _forward_batch

    torch.manual_seed(0)
    B, S, C = 8, 16, 32
    # Per-slot constants: every sample shares slot i's code — zero
    # cross-sample information, healthy-looking cross-slot spread.
    z_collapsed = torch.randn(1, S, C).expand(B, S, C).contiguous()
    sigreg_fn = build_sigreg_from_config(
        OmegaConf.create({"knots": 17, "num_proj": 128})
    )
    z_healthy = torch.randn(B, S, C)
    # Cross-mode magnitudes are not comparable (SIGReg scales its statistic
    # by the per-group sample count: B*S folded vs B per-slot). The
    # property that matters: within the per-slot mode, per-slot-constant
    # collapse must be penalised far harder than a healthy latent — the
    # folded mode cannot make that distinction (both look like a spread
    # of rows).
    ps_collapsed = sigreg_fn.regularizer(z_collapsed.transpose(0, 1).float())
    ps_healthy = sigreg_fn.regularizer(z_healthy.transpose(0, 1).float())
    assert ps_collapsed.item() > 3.0 * ps_healthy.item()
    # (No folded-blindness assertion: with EXACT duplicates the folded
    # statistic reacts too; the empirically observed blindness on trained
    # checkpoints involves jittered near-constants. The guarantee under
    # test is that the per-slot path — previously a silent no-op — now
    # actually separates collapsed from healthy latents.)


def test_dataset_mach_range_and_max_samples(synthetic_superwing):
    import numpy as np

    idx = np.load(synthetic_superwing / "index.npy")
    machs = idx[:, 3]
    lo = float(machs.min())
    mid = float(np.median(machs))
    ds_in = _make_dataset(synthetic_superwing, mach_range=[lo, mid])
    ds_out = _make_dataset(
        synthetic_superwing, mach_range=[lo, mid], mach_range_invert=True
    )
    all_train = set(_make_dataset(synthetic_superwing).sample_indices)
    assert set(ds_in.sample_indices) | set(ds_out.sample_indices) == all_train
    assert set(ds_in.sample_indices) & set(ds_out.sample_indices) == set()
    for si in ds_in.sample_indices:
        assert lo <= machs[si] <= mid
    ds_few = _make_dataset(
        synthetic_superwing, max_samples=1, subset_seed=7
    )
    assert len(ds_few.sample_indices) == 1
    ds_few2 = _make_dataset(
        synthetic_superwing, max_samples=1, subset_seed=7
    )
    assert ds_few.sample_indices == ds_few2.sample_indices  # seeded
