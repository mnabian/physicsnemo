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

"""Standalone DrivAerML validation for a trained PhysicsJEPA checkpoint.

Prints the **same** per-channel surface metrics GeoTransolver reports at
validation (un-normalised relative L2 / L1 for pressure + 3 wall-shear), so a
PhysicsJEPA checkpoint can be compared head-to-head against the GTS run's
"Validation Average Metrics" table. Reuses the exact metric helper from
``train_flarejepa`` (``_surface_field_metrics``) and the exact adapter
(``build_drivaer_flarejepa_loader``) — identical field normalisation and
formula on both sides.

The checkpoint carries its own resolved config (``payload["config"]``), so the
model is rebuilt exactly as trained; only the val data path can be overridden.

Usage (single GPU, inside the 26.06 container):

    python eval_drivaer_flarejepa.py \
        --checkpoint /code/mnabian/flarejepa/runs/drivaer_jepa/outputs/checkpoints/best.pt

The GeoTransolver counterpart numbers come straight from its training log's
"Epoch N Validation Average Metrics" table (l2_pressure_surf, l2_shear_*).
"""

from __future__ import annotations

import argparse

import hydra
import torch
from omegaconf import OmegaConf

from physicsnemo.distributed import DistributedManager
from src.datapipes.drivaer_flarejepa import build_drivaer_flarejepa_loader
from src.training.runtime import get_autocast_context, move_batch_to_device
from train_flarejepa import (
    _SURFACE_FIELD_METRIC_KEYS,
    _forward_batch,
    _surface_field_metrics,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint", required=True, help="Path to a FlareJEPA .pt checkpoint."
    )
    ap.add_argument(
        "--val-path",
        default=None,
        help="Override data.val_path (defaults to the checkpoint's config).",
    )
    ap.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap validation batches (default: full val set).",
    )
    ap.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        help="Evaluate the live weights instead of the EMA shadow "
        "(validation during training uses EMA, so EMA is the default).",
    )
    ap.set_defaults(use_ema=True)
    return ap.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    DistributedManager.initialize()
    dm = DistributedManager()
    device = dm.device

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.create(payload["config"])
    if str(cfg.data.get("source", "")) != "drivaer":
        raise ValueError(
            "checkpoint config is not a DrivAer run "
            f"(data.source={cfg.data.get('source')!r}); this eval is DrivAer-only."
        )
    if args.val_path is not None:
        cfg.data.val_path = args.val_path

    # Rebuild the model exactly as trained; weights are not DDP-wrapped.
    model = hydra.utils.instantiate(cfg.model).to(device).eval()
    model.load_state_dict(payload["model"])
    used_ema = False
    if args.use_ema and payload.get("ema_shadow") is not None:
        # Validation during training runs under the EMA weights; match that.
        model.load_state_dict(payload["ema_shadow"], strict=False)
        used_ema = True

    loader, _ = build_drivaer_flarejepa_loader(
        cfg.data,
        split="val",
        batch_size=int(cfg.training.eval_batch_size),
        shuffle=False,
        world_size=1,
        rank=0,
    )
    factors = loader.dataset.surface_factors
    mean, std = factors["mean"], factors["std"]
    use_normals = bool(cfg.model.use_normals)
    precision = str(cfg.training.precision)

    totals = {k: 0.0 for k in _SURFACE_FIELD_METRIC_KEYS}
    n = 0
    for i, batch in enumerate(loader):
        if args.max_batches is not None and i >= args.max_batches:
            break
        batch = move_batch_to_device(batch, device)
        with get_autocast_context(device, precision):
            out = _forward_batch(
                model,
                batch,
                use_normals=use_normals,
                run_target=False,  # inference path: teacher not needed
                decode_from="predictor",
                normalize_target=True,
            )
        fp_phys = out["field_pred"].float() * std + mean
        tgt_phys = batch["query_target"].float() * std + mean
        fm = _surface_field_metrics(fp_phys, tgt_phys)
        for k in _SURFACE_FIELD_METRIC_KEYS:
            totals[k] += float(fm[k])
        n += 1

    avg = {k: totals[k] / max(1, n) for k in _SURFACE_FIELD_METRIC_KEYS}
    print(
        f"\nPhysicsJEPA DrivAerML validation — {n} cases, "
        f"{'EMA' if used_ema else 'live'} weights, un-normalised "
        f"(GeoTransolver-comparable):\n"
        f"  rel-L2   pressure={avg['l2_pressure_surf']:.4f}  "
        f"tau_x={avg['l2_shear_x']:.4f}  tau_y={avg['l2_shear_y']:.4f}  "
        f"tau_z={avg['l2_shear_z']:.4f}\n"
        f"  rel-L1   pressure={avg['l1_pressure_surf']:.4f}  "
        f"tau_x={avg['l1_shear_x']:.4f}  tau_y={avg['l1_shear_y']:.4f}  "
        f"tau_z={avg['l1_shear_z']:.4f}\n"
    )


if __name__ == "__main__":
    main()
