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

r"""Hydra-driven training entry point for the FlareJEPA SuperWing recipe.

Differences from ``train.py`` (the AeroJEPA loop), by design:

* **Dense batching.** FlareJEPA runs on fixed shapes (SuperWing subsamples
  are exact-size), so the whole ``(B, N, ...)`` batch goes through the model
  in one forward — no per-sample slicing, no padding masks in the hot path.
* **Latent-space consistency.** The decoder's training-time input space is
  kept identical to the predictor's regression target: when
  ``loss.latent.normalize_target=true``, the teacher latent is
  LayerNorm-normalised once, and BOTH the latent loss target and the
  teacher-forced decoder input use that same tensor. Decoding at inference
  from ``Z_hat`` is then consistent with training.
* The target encoder runs only when a loss term needs it (latent / SIGReg
  weights > 0, or ``decode_from=target``).

Usage::

    python train_flarejepa.py data.path=/path/to/SuperWing_Dataset

See ``conf/config_flarejepa.yaml`` for the configuration surface.
"""

from __future__ import annotations

import itertools
import logging
import time
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.distributed import DistributedManager
from src.datapipes import SuperWingDataset, superwing_collate
from src.losses import (
    build_recon_loss_from_config,
    build_sigreg_from_config,
    compute_latent_loss,
)
from src.training import (
    ExponentialMovingAverage,
    build_lr_scheduler,
    build_optimizer,
    get_autocast_context,
    linear_warmup_weight,
    move_batch_to_device,
    set_seed,
)
from train import (
    _ensure_superwing_artifacts,
)


log = logging.getLogger(__name__)


def _all_reduce_grads(model: torch.nn.Module, world_size: int) -> None:
    """Average gradients across ranks (manual data parallel).

    Unlike train.py's version, params whose grad is None are SKIPPED, not
    zero-materialized: AdamW applies decoupled weight decay to any param
    that has a grad (even zero), so materializing zeros silently decays
    grad-less params on multi-GPU while single-GPU leaves them untouched
    — verified as an exact 1.8e-4 fp64 divergence on the detached teacher.
    The None-grad set is identical on every rank (same graph, same phase),
    so skipping keeps the collective balanced.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset : offset + n].view_as(g))
        offset += n


# --------------------------------------------------------------------------- #
# Checkpointing
#
# Deliberately NOT reusing train.py's _save_checkpoint/_load_initial_state:
# those persist the EMA shadow AS the model state, so a resumed run restarts
# from EMA-smoothed weights with an optimizer state that tracked different
# parameters — a silent trajectory change on every walltime-chained slot.
# Here the LIVE weights and the EMA shadow are stored separately and both
# restored on resume; evaluation/model-selection still runs on EMA weights.
# --------------------------------------------------------------------------- #


def _save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    ema: ExponentialMovingAverage | None,
    epoch: int,
    best_val: float,
    cfg: DictConfig,
) -> None:
    from omegaconf import OmegaConf

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val),
        "model": model.state_dict(),  # LIVE weights, matching the optimizer
        "optimizer": optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    if ema is not None:
        payload["ema_shadow"] = ema.shadow
        payload["ema_decay"] = ema.decay
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic on POSIX: no truncated latest.pt on preemption
    log.info("Saved checkpoint to %s", path)


def _load_initial_state(
    cfg: DictConfig,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    ema: ExponentialMovingAverage | None,
    device: torch.device,
) -> tuple[int, float]:
    """Resume (live weights + optimizer + scheduler + EMA shadow) or
    init-from-checkpoint (model weights only). Both off by default.

    ``training.resume.checkpoint_path`` may point to a not-yet-existing
    ``latest.pt``: the first slot of a walltime chain starts fresh and
    later slots resume — so a missing file is a clean cold start, not an
    error (``training.resume.strict_path=true`` restores erroring).
    """
    train_cfg = cfg.training

    resume_cfg = train_cfg.get("resume", None)
    resume_missing = False
    if resume_cfg is not None and bool(resume_cfg.get("enabled", False)):
        ckpt_path = resume_cfg.get("checkpoint_path")
        if not ckpt_path:
            raise ValueError("training.resume.enabled=true requires checkpoint_path.")
        if not Path(str(ckpt_path)).exists():
            if bool(resume_cfg.get("strict_path", False)):
                raise FileNotFoundError(f"resume checkpoint missing: {ckpt_path}")
            # First slot of a chain: fall through to init_from_checkpoint
            # (warm start) if configured, else cold start.
            log.info(
                "Resume enabled but %s does not exist — first slot of a "
                "chain; falling through to init_from_checkpoint/cold start.",
                ckpt_path,
            )
            resume_missing = True
    if (
        resume_cfg is not None
        and bool(resume_cfg.get("enabled", False))
        and not resume_missing
    ):
        ckpt_path = resume_cfg.get("checkpoint_path")
        payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(
            payload["model"], strict=bool(resume_cfg.get("strict", True))
        )
        if bool(resume_cfg.get("load_optimizer", True)) and payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        if (
            bool(resume_cfg.get("load_scheduler", True))
            and lr_scheduler is not None
            and payload.get("lr_scheduler")
        ):
            lr_scheduler.load_state_dict(payload["lr_scheduler"])
        if ema is not None and payload.get("ema_shadow"):
            ema.shadow = {k: v.to(device) for k, v in payload["ema_shadow"].items()}
        start_epoch = int(payload.get("epoch", 0))
        best_val = float(payload.get("best_val_loss", float("inf")))
        log.info(
            "Resumed from %s at epoch %d (best_val=%.4e)",
            ckpt_path,
            start_epoch,
            best_val,
        )
        return start_epoch, best_val

    init_cfg = train_cfg.get("init_from_checkpoint", None)
    if init_cfg is not None and init_cfg.get("path"):
        ckpt_path = init_cfg.get("path")
        payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        result = model.load_state_dict(
            payload["model"], strict=bool(init_cfg.get("strict", True))
        )
        if ema is not None:
            # Re-snapshot the shadow AFTER the checkpoint load. The EMA is
            # constructed on the randomly initialised model; leaving that
            # stale shadow in place poisons EMA-weight evaluation and
            # best-checkpoint selection for short fine-tunes (the shadow
            # only converges to the loaded weights after ~1/(1-decay)
            # steps).
            ema.shadow = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
        log.info(
            "Initialised model weights from %s (missing=%d, unexpected=%d)",
            ckpt_path,
            len(result.missing_keys),
            len(result.unexpected_keys),
        )
        return 0, float("inf")

    return 0, float("inf")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def _build_loader(
    data_cfg: DictConfig,
    *,
    split: str,
    split_manifest_path: str,
    normalization_stats_path: str,
    batch_size: int,
    shuffle: bool,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[DataLoader, Any]:
    """FlareJEPA loader: same as train.py's but passes ``include_normals``."""
    deterministic = (
        bool(data_cfg.train_deterministic_sampling)
        if split == "train"
        else bool(data_cfg.eval_deterministic_sampling)
    )
    dataset = SuperWingDataset(
        root_dir=str(data_cfg.path),
        split=split,
        split_manifest_path=split_manifest_path,
        normalization_stats_path=normalization_stats_path,
        surface_points=int(data_cfg.surface_points),
        target_encoder_points=int(data_cfg.target_encoder_points),
        query_points=int(data_cfg.query_points),
        eval_full_grid_query=bool(data_cfg.eval_full_grid_query),
        return_origingeom=False,
        return_full_fields=False,
        deterministic_sampling=deterministic,
        normalize_xyz=bool(data_cfg.normalize_xyz),
        include_normals=bool(data_cfg.get("include_normals", False)),
        mach_range=data_cfg.get("mach_range", None),
        mach_range_invert=bool(data_cfg.get("mach_range_invert", False)),
        max_samples=(
            int(data_cfg.get("max_train_samples"))
            if split == "train" and data_cfg.get("max_train_samples") is not None
            else None
        ),
        subset_seed=int(data_cfg.get("subset_seed", 0)),
    )
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=bool(shuffle),
            drop_last=(split == "train"),
        )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(data_cfg.num_workers),
        pin_memory=bool(data_cfg.pin_memory),
        collate_fn=superwing_collate,
        drop_last=False,
        persistent_workers=int(data_cfg.num_workers) > 0,
        prefetch_factor=(
            int(data_cfg.get("prefetch_factor", 4))
            if int(data_cfg.num_workers) > 0
            else None
        ),
    )
    return loader, sampler


# --------------------------------------------------------------------------- #
# Batched forward + loss
# --------------------------------------------------------------------------- #


def _assemble_features(
    batch: dict[str, torch.Tensor], *, use_normals: bool
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Build geometry / target feature tensors matching the model's input dims.

    Geometry features: ``[xyz, normals]`` (or ``None`` for xyz-only, which
    the model defaults to positions). Target features: ``[xyz, normals,
    field]`` — the datapipe's ``target_surface_main_feat`` is ``[xyz(3),
    field(3)]``, so normals are spliced between them.
    """
    main_feat = batch["target_surface_main_feat"]
    if not use_normals:
        return None, main_feat
    if "context_normals" not in batch or "target_surface_normals" not in batch:
        raise KeyError(
            "model.use_normals=true requires data.include_normals=true "
            "(the datapipe must emit context_normals / "
            "target_surface_normals)."
        )
    geo_feat = torch.cat([batch["context_pos"], batch["context_normals"]], dim=-1)
    tgt_feat = torch.cat(
        [main_feat[..., :3], batch["target_surface_normals"], main_feat[..., 3:]],
        dim=-1,
    )
    return geo_feat, tgt_feat


def _forward_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    use_normals: bool,
    run_target: bool,
    decode_from: str,
    normalize_target: bool,
    teacher_forced_decode: bool = False,
) -> dict[str, torch.Tensor | None]:
    """One dense batched forward through the FlareJEPA stack.

    Thin adapter over ``FlareJEPA.forward`` — the model owns the canonical
    composition (single source of truth; the unit-tested path IS the
    training path). This wrapper only assembles the feature tensors from
    the batch dict and derives ``z_tgt_canonical`` (the normalised teacher
    latent — the latent-loss target, numerically identical to the
    teacher-forced decoder input computed inside ``forward``).
    """
    geo_feat, tgt_feat = _assemble_features(batch, use_normals=use_normals)
    if decode_from == "target" and not run_target:
        raise ValueError(
            "decode_from=target requires the target encoder to run; "
            "enable a latent/sigreg loss term."
        )
    kwargs = {}
    if teacher_forced_decode:
        kwargs["also_decode_target"] = True
    out = model(
        batch["context_pos"],
        batch["target_surface_pos"],
        tgt_feat,
        batch["query_pos"],
        batch["gen_params"],
        geometry_features=geo_feat,
        decode_from=decode_from,
        run_target=run_target,
        normalize_target=normalize_target,
        **kwargs,
    )
    field_pred_teacher = None
    if teacher_forced_decode:
        field_pred, z_hat, z_tgt, field_pred_teacher = out
    else:
        field_pred, z_hat, z_tgt = out
    z_tgt_canonical = None
    if z_tgt is not None:
        z_tgt_canonical = (
            F.layer_norm(z_tgt, z_tgt.shape[-1:]) if normalize_target else z_tgt
        )
    return {
        "field_pred": field_pred,
        "z_hat": z_hat,
        "z_tgt": z_tgt,
        "z_tgt_canonical": z_tgt_canonical,
        "field_pred_teacher": field_pred_teacher,
    }


def _compute_total_loss(
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, torch.Tensor],
    *,
    recon_loss_fn: torch.nn.Module,
    sigreg_loss_fn: torch.nn.Module,
    loss_cfg: DictConfig,
    term_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine recon + latent + SIGReg with phase-resolved weights."""
    field_pred = outputs["field_pred"]
    recon_term = recon_loss_fn(field_pred, batch["query_target"])
    total = term_weights["recon"] * recon_term

    zeros = torch.zeros((), device=field_pred.device)
    latent_term = zeros
    sigreg_term = zeros
    z_tgt = outputs["z_tgt"]
    if z_tgt is not None:
        target = outputs["z_tgt_canonical"]
        # NOTE: compute_latent_loss's docstring describes AeroJEPA's joint
        # (non-stop-grad) training. FlareJEPA defaults to stop_grad=true:
        # the teacher trains via teacher-forced recon + SIGReg,
        # NOT via this term. Set loss.latent.stop_grad=false to reproduce
        # the AeroJEPA joint dynamics.
        if bool(loss_cfg.latent.get("stop_grad", True)):
            target = target.detach()
        latent_term = compute_latent_loss(
            outputs["z_hat"],
            target,
            mse_weight=float(loss_cfg.latent.mse_weight),
            cosine_weight=float(loss_cfg.latent.cosine_weight),
        )
        # SIGReg regularises the RAW teacher latent (pre-normalisation),
        # mirroring the AeroJEPA recipe.
        # With per_slot=true the (B, S, C) latent is transposed
        # to (S, B, C) so SIGReg enforces Gaussianity ACROSS SAMPLES per
        # slot — the default (slots folded into batch) is blind to
        # per-slot-constant collapse. The TokenLatentSIGReg WRAPPER must be
        # bypassed here: its reshape flattens any rank-3 input to
        # (1, B*S, C), which silently undoes the transpose (the wrapped
        # call would be a no-op). Feed (S, B, C) straight into the
        # SIGReg module so T=S groups survive. NOTE: SIGReg's internal
        # statistic scaling multiplies by the per-group sample count (B,
        # not B*S) — per-slot magnitudes are NOT comparable with the
        # folded default; tune loss.sigreg.weight per mode. Slot-latent
        # models only.
        if bool(loss_cfg.sigreg.get("per_slot", False)):
            sigreg_term = sigreg_loss_fn.regularizer(
                z_tgt.transpose(0, 1).float()
            )
        else:
            sigreg_term = sigreg_loss_fn(z_tgt, None)
        total = (
            total
            + term_weights["latent"] * latent_term
            + term_weights["sigreg"] * sigreg_term
        )

    tf_term = zeros
    tf_weight = float(loss_cfg.recon.get("teacher_forced_weight", 0.0))
    if tf_weight > 0.0 and outputs.get("field_pred_teacher") is not None:
        # Field-ground the teacher — decode the canonical
        # teacher latent too and take a (small) recon loss on it, so the
        # latent target must carry field information even when the main
        # decode path is decode_from=predictor.
        tf_term = recon_loss_fn(
            outputs["field_pred_teacher"], batch["query_target"]
        )
        total = total + tf_weight * term_weights["recon"] * tf_term

    return total, {
        "recon": recon_term.detach(),
        "latent": latent_term.detach(),
        "sigreg": sigreg_term.detach(),
        "recon_tf": tf_term.detach(),
    }


def _compute_term_weights(epoch: int, loss_cfg: DictConfig) -> dict[str, float]:
    """Linearly-warmed-up scalar weights for the loss terms."""

    def warm(weight, warmup_epochs) -> float:
        return linear_warmup_weight(float(weight), float(warmup_epochs), float(epoch))

    return {
        "recon": warm(loss_cfg.recon.weight, loss_cfg.recon.warmup_epochs),
        "latent": warm(loss_cfg.latent.weight, loss_cfg.latent.warmup_epochs),
        "sigreg": warm(loss_cfg.sigreg.weight, loss_cfg.sigreg.warmup_epochs),
    }


# --------------------------------------------------------------------------- #
# Epoch loop
# --------------------------------------------------------------------------- #


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: Any,
    epoch_len: int,
    recon_loss_fn: torch.nn.Module,
    sigreg_loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: Any,
    ema: ExponentialMovingAverage | None,
    device: torch.device,
    precision: str,
    grad_clip_norm: float,
    loss_cfg: DictConfig,
    use_normals: bool,
    decode_from: str,
    epoch: int,
    max_batches: int | None,
    scaler: torch.amp.GradScaler | None = None,
    spike_guard_state: dict | None = None,
    writer: SummaryWriter | None = None,
    log_every: int = 50,
    world_size: int = 1,
    is_main: bool = True,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    term_weights = _compute_term_weights(epoch, loss_cfg)
    # The target encoder only runs when something consumes its output.
    run_target = (
        term_weights["latent"] > 0.0
        or term_weights["sigreg"] > 0.0
        or decode_from == "target"
    )
    normalize_target = bool(loss_cfg.latent.get("normalize_target", True))

    # z_std: mean per-channel std of the RAW teacher latent across all
    # tokens in the batch — the collapse diagnostic (SIGReg must keep slot
    # variance non-degenerate; z_std -> 0 is collapse).
    # Reported as 0 when the target encoder does not run.
    totals = {
        k: torch.zeros((), device=device, dtype=torch.float64)
        for k in ("loss", "recon", "latent", "sigreg", "recon_tf",
                  "z_std", "z_xstd")
    }
    n_batches = 0
    phase_tag = "train" if is_train else "val"
    step_time = time.time()

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with (
            torch.set_grad_enabled(is_train),
            get_autocast_context(device, precision),
        ):
            outputs = _forward_batch(
                model,
                batch,
                use_normals=use_normals,
                run_target=run_target,
                decode_from=decode_from,
                normalize_target=normalize_target,
                teacher_forced_decode=(
                    run_target
                    and float(
                        loss_cfg.recon.get("teacher_forced_weight", 0.0)
                    )
                    > 0.0
                ),
            )
            loss, parts = _compute_total_loss(
                outputs,
                batch,
                recon_loss_fn=recon_loss_fn,
                sigreg_loss_fn=sigreg_loss_fn,
                loss_cfg=loss_cfg,
                term_weights=term_weights,
            )

        skip_step = False
        if is_train and spike_guard_state is not None:
            # Loss-spike guard (empirically, one spike tipped
            # the joint teacher/student system into the latent-collapse
            # basin permanently). Decision uses the GLOBALLY-averaged loss
            # so every rank skips identically (optimizer states stay in
            # lockstep).
            loss_val = loss.detach().float()
            if world_size > 1:
                dist.all_reduce(loss_val, op=dist.ReduceOp.AVG)
            ema_val = spike_guard_state["ema"]
            factor = spike_guard_state["factor"]
            if (
                spike_guard_state["steps"] >= spike_guard_state["warmup"]
                and ema_val is not None
                and float(loss_val) > factor * ema_val
            ):
                skip_step = True
                spike_guard_state["skipped"] += 1
                if is_main and spike_guard_state["skipped"] <= 20:
                    log.warning(
                        "spike guard: loss %.4f > %.1f x EMA %.4f — "
                        "skipping step (total skipped: %d)",
                        float(loss_val),
                        factor,
                        ema_val,
                        spike_guard_state["skipped"],
                    )
            # The EMA tracks the loss UNCONDITIONALLY (including skipped
            # steps): a legitimate regime change (e.g. the latent/SIGReg
            # term warmup ramping the total loss) re-normalises the
            # threshold within ~1/(1-beta) steps and skipping
            # self-terminates. An accepted-steps-only EMA deadlocks: the
            # ramp looks like a spike, every step is skipped, the EMA
            # never catches up (observed: r6lf_t/ab_muon frozen for 50+
            # epochs). True one-step spikes are still rejected — one
            # outlier moves the EMA by only (1-beta) x overshoot.
            beta = spike_guard_state["beta"]
            spike_guard_state["ema"] = (
                float(loss_val)
                if ema_val is None
                else beta * ema_val + (1.0 - beta) * float(loss_val)
            )
            spike_guard_state["steps"] += 1

        if is_train and not skip_step:
            scaler.scale(loss).backward()
            if world_size > 1 or grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
            if world_size > 1:
                _all_reduce_grads(model, world_size)
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(grad_clip_norm)
                )
            scaler.step(optimizer)
            scaler.update()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if ema is not None:
                ema.update(model)

        totals["loss"] += loss.detach().double()
        for k in ("recon", "latent", "sigreg", "recon_tf"):
            totals[k] += parts[k].double()
        z_tgt = outputs["z_tgt"]
        if z_tgt is not None:
            z = z_tgt.detach()
            totals["z_std"] += (
                z.reshape(-1, z.shape[-1]).float().std(dim=0).mean().double()
            )
            # Cross-SAMPLE std per (slot, channel) — the collapse
            # mode z_std cannot see (per-slot constants keep z_std healthy
            # while carrying zero per-sample information).
            if z.shape[0] > 1:
                totals["z_xstd"] += z.float().std(dim=0).mean().double()
        n_batches += 1

        now = time.time()
        step_dur = now - step_time
        step_time = now
        if is_main and batch_idx % max(1, int(log_every)) == 0:
            mem_gb = (
                torch.cuda.memory_reserved() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            )
            log.info(
                "Epoch %03d %s [%d/%d] Loss: %.6f recon: %.4f "
                "Duration: %.2fs Mem: %.2fGB",
                epoch,
                phase_tag,
                batch_idx,
                epoch_len,
                float(loss.detach()),
                float(parts["recon"]),
                step_dur,
                mem_gb,
            )
            if writer is not None:
                gstep = batch_idx + epoch_len * epoch
                writer.add_scalar(
                    f"batch/{phase_tag}_loss", float(loss.detach()), gstep
                )
                writer.add_scalar(
                    f"batch/{phase_tag}_recon", float(parts["recon"]), gstep
                )
                if is_train and lr_scheduler is not None:
                    writer.add_scalar(
                        "batch/learning_rate",
                        optimizer.param_groups[0]["lr"],
                        gstep,
                    )

    keys = list(totals.keys())
    count = torch.tensor(float(n_batches), device=device, dtype=torch.float64)
    packed = torch.stack([totals[k] for k in keys] + [count])
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    reduced = packed.tolist()
    n_total = reduced[-1]
    if n_total == 0:
        return {k: float("nan") for k in keys}
    return {k: reduced[i] / n_total for i, k in enumerate(keys)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


@hydra.main(config_path="conf", config_name="config_flarejepa", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point — train a FlareJEPA model on SuperWing."""
    DistributedManager.initialize()
    dm = DistributedManager()
    world_size = dm.world_size
    is_main = dm.rank == 0

    set_seed(int(cfg.seed))
    device = dm.device
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = output_dir / cfg.output_dir / "checkpoints"
    tb_dir = output_dir / cfg.output_dir / "tensorboard"
    if is_main:
        log.info("Output dir: %s  (world_size=%d)", output_dir, world_size)

    # Cross-config consistency: the model's normals expectation and the
    # datapipe's emission must agree (fail fast, not at feature assembly).
    use_normals = bool(cfg.model.use_normals)
    if use_normals and not bool(cfg.data.get("include_normals", False)):
        raise ValueError(
            "model.use_normals=true but data.include_normals=false — the "
            "datapipe would not emit the normals the model expects."
        )
    if bool(cfg.model.use_sdf):
        raise ValueError("SuperWing has no SDF; set model.use_sdf=false.")
    if str(cfg.training.precision).lower() == "fp16" and world_size > 1:
        # The GradScaler records overflow per-rank BEFORE _all_reduce_grads
        # shares gradients, so a local inf contaminates every rank while only
        # the originating rank skips the step — silent weight divergence.
        raise ValueError(
            "precision=fp16 with world_size>1 is not supported by the manual "
            "gradient all-reduce (per-rank overflow detection would "
            "desynchronise ranks). Use bf16."
        )

    if is_main:
        split_path, stats_path = _ensure_superwing_artifacts(cfg.data)
    if world_size > 1:
        dist.barrier()
    if not is_main:
        split_path, stats_path = _ensure_superwing_artifacts(cfg.data)

    train_loader, train_sampler = _build_loader(
        cfg.data,
        split="train",
        split_manifest_path=split_path,
        normalization_stats_path=stats_path,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        world_size=world_size,
        rank=dm.rank,
    )
    val_loader, _ = _build_loader(
        cfg.data,
        split="val",
        split_manifest_path=split_path,
        normalization_stats_path=stats_path,
        batch_size=int(cfg.training.eval_batch_size),
        shuffle=False,
        world_size=world_size,
        rank=dm.rank,
    )
    if is_main:
        log.info(
            "Train / val samples: %d / %d",
            len(train_loader.dataset),
            len(val_loader.dataset),
        )

    model = hydra.utils.instantiate(cfg.model).to(device)
    if is_main:
        log.info(
            "Model parameters: %.2f M",
            sum(p.numel() for p in model.parameters()) / 1e6,
        )

    # Optional one-time FLOP count (Phase 4: "Mean TFLOPs reported").
    # Counts the model forward on one real batch with FlopCounterMode
    # (eager, before compile); backward is approximated as 2x forward, the
    # standard convention. Per-epoch throughput is then
    # 3 * flops_fwd * steps / epoch_seconds.
    flops_fwd_per_step = None
    if bool(cfg.training.get("report_tflops", False)):
        from torch.utils.flop_counter import FlopCounterMode

        probe = move_batch_to_device(next(iter(train_loader)), device)
        probe_run_target = (
            float(cfg.training.loss.latent.weight) > 0.0
            or float(cfg.training.loss.sigreg.weight) > 0.0
            or str(cfg.training.decode_from) == "target"
        )
        counter = FlopCounterMode(display=False)
        # NOT under torch.no_grad(): FlopCounterMode's module tracker
        # registers grad hooks and asserts if tensors lack grad_fn.
        # No backward runs, so no gradients materialize regardless.
        with counter:
            _forward_batch(
                model,
                probe,
                use_normals=use_normals,
                run_target=probe_run_target,
                decode_from=str(cfg.training.decode_from),
                normalize_target=bool(
                    cfg.training.loss.latent.get("normalize_target", True)
                ),
            )
        flops_fwd_per_step = counter.get_total_flops()
        if is_main:
            log.info(
                "Forward FLOPs/step: %.3f T (train step approx %.3f T with "
                "backward)",
                flops_fwd_per_step / 1e12,
                3 * flops_fwd_per_step / 1e12,
            )

    # Optional torch.compile (Phase 4). Fixed input shapes are a design
    # invariant, so no dynamic-shape guards are needed. In-place
    # nn.Module.compile() (NOT the torch.compile wrapper) keeps state_dict
    # keys un-prefixed, so checkpoints/EMA/resume stay interchangeable
    # between compiled and eager runs.
    if bool(cfg.training.get("compile", False)):
        for name in (
            "geometry_encoder", "target_encoder", "predictor", "decoder"
        ):
            getattr(model, name).compile()
        if is_main:
            log.info("torch.compile enabled on all submodules (in-place)")

    freeze_except = cfg.training.get("freeze_except", None)
    if freeze_except:
        # Few-shot transfer (latent-value protocol): freeze everything
        # except parameters whose names start with a listed prefix. The
        # optimizer still sees all params; frozen ones never get grads
        # (the None-skipping all-reduce and AdamW handle that cleanly).
        # Substring match (not just prefix) so scattered adapter params
        # like AdaLN modulators (".ada.") can be targeted as one group.
        prefixes = tuple(str(x) for x in freeze_except)
        n_train = 0
        for name, p_ in model.named_parameters():
            trainable = any(k in name for k in prefixes)
            p_.requires_grad_(trainable)
            n_train += int(trainable) * p_.numel()
        if is_main:
            log.info(
                "freeze_except=%s: %.3fM trainable of %.3fM params",
                list(prefixes),
                n_train / 1e6,
                sum(p_.numel() for p_ in model.parameters()) / 1e6,
            )

    recon_loss_fn = build_recon_loss_from_config(cfg.training.loss.recon).to(device)
    sigreg_loss_fn = build_sigreg_from_config(cfg.training.loss.sigreg).to(device)
    optimizer = build_optimizer(model, cfg.training.optimizer)
    # The scheduler steps once per ACTUAL batch, so its horizon must use the
    # effective epoch length: max_train_batches caps a normal epoch, and
    # overfit mode repeats one batch max_train_batches (or len(loader)) times.
    _mtb_sched = cfg.training.get("max_train_batches", None)
    if bool(cfg.training.get("overfit_one_batch", False)):
        steps_per_epoch = int(_mtb_sched) if _mtb_sched else len(train_loader)
    elif _mtb_sched is not None:
        steps_per_epoch = min(int(_mtb_sched), len(train_loader))
    else:
        steps_per_epoch = len(train_loader)
    lr_scheduler = build_lr_scheduler(
        optimizer,
        name=str(cfg.training.scheduler.name),
        epochs=int(cfg.training.epochs),
        steps_per_epoch=max(1, steps_per_epoch),
        warmup_epochs=float(cfg.training.scheduler.warmup_epochs),
        step_size_epochs=int(
            cfg.training.scheduler.get("step_size_epochs", 100)
        ),
        gamma=float(cfg.training.scheduler.get("gamma", 0.5)),
    )
    ema: ExponentialMovingAverage | None = None
    if bool(cfg.training.ema.enabled):
        ema = ExponentialMovingAverage(model, decay=float(cfg.training.ema.decay))

    writer = SummaryWriter(log_dir=str(tb_dir)) if is_main else None

    grad_clip_norm = float(cfg.training.grad_clip_norm)
    save_every = int(cfg.training.save_every_epochs)
    max_eval_batches = int(cfg.training.max_eval_batches)
    log_every = int(cfg.training.get("log_every", 50))
    _mtb = cfg.training.get("max_train_batches", None)
    max_train_batches = int(_mtb) if _mtb is not None else None
    decode_from = str(cfg.training.decode_from)
    overfit_one_batch = bool(cfg.training.get("overfit_one_batch", False))

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            str(cfg.training.precision).lower() == "fp16" and device.type == "cuda"
        ),
    )

    start_epoch, best_val_loss = _load_initial_state(
        cfg,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema=ema,
        device=device,
    )

    # Overfit-one-batch gate: freeze the first train batch and repeat it.
    overfit_batch: dict | None = None
    if overfit_one_batch:
        overfit_batch = next(iter(train_loader))
        if is_main:
            log.info(
                "overfit_one_batch=true — repeating one batch of %d sample(s); "
                "validation is skipped.",
                int(overfit_batch["context_pos"].shape[0]),
            )

    sg_cfg = cfg.training.get("spike_guard", None)
    spike_guard_state = None
    if sg_cfg is not None and float(sg_cfg.get("factor", 0.0)) > 0.0:
        spike_guard_state = {
            "factor": float(sg_cfg.get("factor", 3.0)),
            "beta": float(sg_cfg.get("ema_beta", 0.98)),
            "warmup": int(sg_cfg.get("warmup_steps", 200)),
            "ema": None,
            "steps": 0,
            "skipped": 0,
        }
        if is_main:
            log.info(
                "spike guard enabled: skip steps with loss > %.1fx EMA "
                "(warmup %d steps)",
                spike_guard_state["factor"],
                spike_guard_state["warmup"],
            )

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if overfit_batch is not None:
            steps = max_train_batches or len(train_loader)
            epoch_loader: Any = itertools.repeat(overfit_batch, steps)
            epoch_len = steps
        else:
            epoch_loader = train_loader
            epoch_len = len(train_loader)
        train_metrics = _run_epoch(
            model=model,
            loader=epoch_loader,
            epoch_len=epoch_len,
            recon_loss_fn=recon_loss_fn,
            sigreg_loss_fn=sigreg_loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            device=device,
            precision=str(cfg.training.precision),
            grad_clip_norm=grad_clip_norm,
            loss_cfg=cfg.training.loss,
            use_normals=use_normals,
            decode_from=decode_from,
            epoch=epoch,
            max_batches=max_train_batches,
            scaler=scaler,
            spike_guard_state=spike_guard_state,
            writer=writer,
            log_every=log_every,
            world_size=world_size,
            is_main=is_main,
        )
        train_time = time.time() - t0

        if overfit_batch is not None:
            val_metrics = dict(train_metrics)
        else:
            if ema is not None:
                ema.apply_to(model)
            try:
                val_metrics = _run_epoch(
                    model=model,
                    loader=val_loader,
                    epoch_len=len(val_loader),
                    recon_loss_fn=recon_loss_fn,
                    sigreg_loss_fn=sigreg_loss_fn,
                    optimizer=None,
                    lr_scheduler=None,
                    ema=None,
                    device=device,
                    precision=str(cfg.training.precision),
                    grad_clip_norm=0.0,
                    loss_cfg=cfg.training.loss,
                    use_normals=use_normals,
                    decode_from=decode_from,
                    epoch=epoch,
                    max_batches=max_eval_batches,
                    writer=writer,
                    log_every=log_every,
                    world_size=world_size,
                    is_main=is_main,
                )
            finally:
                if ema is not None:
                    ema.restore(model)

        if is_main:
            log.info(
                "epoch=%03d  train_loss=%.6f  val_loss=%.6f  "
                "train_recon=%.6f val_recon=%.6f  "
                "train_latent=%.6f  train_sigreg=%.6f  z_std=%.4f  "
                "z_xstd=%.4f  lr=%.2e  time=%.1fs",
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                train_metrics["recon"],
                val_metrics["recon"],
                train_metrics["latent"],
                train_metrics["sigreg"],
                train_metrics["z_std"],
                train_metrics["z_xstd"],
                optimizer.param_groups[0]["lr"],
                train_time,
            )
            if flops_fwd_per_step is not None and train_time > 0:
                tflops = 3 * flops_fwd_per_step * epoch_len / train_time / 1e12
                log.info("Mean throughput: %.2f TFLOP/s", tflops)
                writer.add_scalar("train/tflops", tflops, epoch)
            for split_name, m in (("train", train_metrics), ("val", val_metrics)):
                for k, v in m.items():
                    writer.add_scalar(f"{split_name}/{k}", v, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
            if (epoch + 1) % save_every == 0 or epoch + 1 == int(cfg.training.epochs):
                _save_checkpoint(
                    path=ckpt_dir / f"epoch_{epoch + 1:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ema=ema,
                    epoch=epoch + 1,
                    best_val=best_val_loss,
                    cfg=cfg,
                )

        # The warmup-weighted total val loss changes meaning
        # every epoch during warmup; select_on=recon selects on the pure
        # (inference-relevant) reconstruction metric instead.
        select_key = str(cfg.training.get("select_on", "loss"))
        if val_metrics[select_key] < best_val_loss:
            best_val_loss = val_metrics[select_key]
            if is_main:
                _save_checkpoint(
                    path=ckpt_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ema=ema,
                    epoch=epoch + 1,
                    best_val=best_val_loss,
                    cfg=cfg,
                )

        # latest.pt every epoch (after best_val update so a resume restores
        # the exact selection state): walltime-chained slots resume from it.
        if is_main:
            _save_checkpoint(
                path=ckpt_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema=ema,
                epoch=epoch + 1,
                best_val=best_val_loss,
                cfg=cfg,
            )

    if writer is not None:
        writer.close()
    if is_main:
        log.info("Training done. Best val_loss=%.6f", best_val_loss)
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
