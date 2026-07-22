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

r"""FlareJEPA conditioning-sweep diagnostic.

Fixes one test geometry and sweeps (aoa, mach) over the dataset's observed
range, decoding the full surface field at each condition. Reports, per
swept variable:

* **Sensitivity** — mean |dfield/dcond| between adjacent sweep points
  (must be clearly non-zero: conditioning reaches the output).
* **Smoothness** — max adjacent-step field change / mean change (a smooth
  response has a small ratio; an erratic one spikes).
* **Ground-truth cross-check** — where the dataset contains the same
  geometry at other conditions, Rel L2 of the swept prediction against
  that case's ground truth (the sweep should track the true field family).

Pass criterion (qualitative): predicted fields respond smoothly and
strongly — Cp peak scales with aoa, field pattern shifts with mach.

Usage::

    python cond_sweep_flarejepa.py \
        checkpoint=/path/to/best.pt \
        data.path=/path/to/SuperWing_Dataset \
        +sweep_output_dir=/path/to/out
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from inference_flarejepa import _build_model_from_payload
from src.datapipes import SuperWingDataset, superwing_collate
from train import _ensure_superwing_artifacts

log = logging.getLogger(__name__)

N_SWEEP = 9


def _decode(model, geo_pos, geo_feat, q_pos, cond, device):
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        return model.predict(geo_pos, q_pos, cond, geometry_features=geo_feat)[
            0
        ].float()


@hydra.main(config_path="conf", config_name="config_flarejepa", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point — conditioning sweep on one fixed geometry."""
    ckpt_path = cfg.get("checkpoint")
    if not ckpt_path:
        raise ValueError("Pass checkpoint=/path/to/best.pt")
    out_dir = Path(str(cfg.get("sweep_output_dir", "cond_sweep_flarejepa")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = _build_model_from_payload(payload, device, use_ema=True).eval()
    use_normals = bool(payload["config"]["model"]["use_normals"])

    split_path, stats_path = _ensure_superwing_artifacts(cfg.data)
    dataset = SuperWingDataset(
        root_dir=str(cfg.data.path),
        split="test",
        split_manifest_path=split_path,
        normalization_stats_path=stats_path,
        surface_points=int(cfg.data.surface_points),
        target_encoder_points=int(cfg.data.target_encoder_points),
        query_points=int(cfg.data.query_points),
        eval_full_grid_query=True,
        deterministic_sampling=True,
        normalize_xyz=bool(cfg.data.normalize_xyz),
        include_normals=use_normals,
    )

    # Fixed geometry: the first test case. Gather every test case sharing
    # its geom_idx for the ground-truth cross-check.
    base = superwing_collate([dataset[0]])
    geom_idx = int(base["geom_idx"][0])
    siblings = [
        i for i in range(len(dataset)) if int(dataset[i]["geom_idx"]) == geom_idx
    ]
    log.info(
        "Fixed geometry geom_idx=%d (case %s); %d sibling test cases",
        geom_idx,
        base["case_id"][0],
        len(siblings),
    )

    geo_pos = base["context_pos"].to(device)
    geo_feat = (
        torch.cat([geo_pos, base["context_normals"].to(device)], dim=-1)
        if use_normals
        else None
    )
    q_pos = base["query_pos"].to(device)

    # Sweep ranges: the dataset's standardised gen_params span ~[-2, 2];
    # sweep each variable over the observed +/-2-sigma range with the other
    # held at the base case's value.
    gp_names = list(dataset.gen_param_names)
    base_gp = base["gen_params"][0].clone()
    report: dict = {"geom_idx": geom_idx, "base_case": base["case_id"][0]}

    for vi, vname in enumerate(gp_names):
        sweep_vals = np.linspace(-2.0, 2.0, N_SWEEP, dtype=np.float32)
        fields = []
        for v in sweep_vals:
            gp = base_gp.clone()
            gp[vi] = float(v)
            f = _decode(model, geo_pos, geo_feat, q_pos, gp.unsqueeze(0).to(device), device)
            fields.append(f.cpu())
        fields_t = torch.stack(fields)  # (N_SWEEP, N_q, 3)

        step = (
            4.0 / (N_SWEEP - 1) * dataset.gen_param_std[vi]
        )  # physical units per step
        diffs = (fields_t[1:] - fields_t[:-1]).abs().mean(dim=(1, 2))  # per step
        sens = diffs / max(step, 1e-12)
        field_scale = fields_t.abs().mean()
        report[vname] = {
            "sweep_std_units": [-2.0, 2.0],
            "mean_abs_dfield_dcond": float(sens.mean()),
            "response_vs_fieldscale": float(
                (fields_t[-1] - fields_t[0]).abs().mean() / field_scale
            ),
            "smoothness_max_over_mean_step": float(diffs.max() / diffs.mean()),
            "per_channel_range_travel": [
                float((fields_t[-1, :, c] - fields_t[0, :, c]).abs().mean())
                for c in range(3)
            ],
        }
        log.info(
            "%s sweep: |dF/dc|=%.4f  full-range response / field scale=%.3f  "
            "smoothness(max/mean step)=%.2f",
            vname,
            report[vname]["mean_abs_dfield_dcond"],
            report[vname]["response_vs_fieldscale"],
            report[vname]["smoothness_max_over_mean_step"],
        )

    # Ground-truth cross-check on sibling conditions of the same geometry.
    # Compare in PHYSICAL units (denormalised): standardised targets are
    # zero-mean, which inflates relative errors and is not comparable to
    # the superwing_metrics numbers.
    t_mean = torch.from_numpy(dataset.target_mean)
    t_std = torch.from_numpy(dataset.target_std)
    xchecks = []
    for i in siblings[:8]:
        s = superwing_collate([dataset[i]])
        pred = _decode(
            model, geo_pos, geo_feat, q_pos, s["gen_params"].to(device), device
        ).cpu()
        pred = pred * t_std + t_mean
        tgt = s["query_target"][0] * t_std + t_mean
        rel = (pred - tgt).norm() / tgt.norm().clamp_min(1e-12)
        xchecks.append(
            {"case": s["case_id"][0], "aoa": float(s["aoa_deg"][0]),
             "mach": float(s["mach"][0]), "rel_l2_all": float(rel)}
        )
        log.info(
            "cross-check %s (aoa=%.2f mach=%.3f): rel_l2=%.4f",
            s["case_id"][0], float(s["aoa_deg"][0]), float(s["mach"][0]), float(rel),
        )
    report["gt_cross_checks"] = xchecks

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cond_sweep_report.json").write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", out_dir / "cond_sweep_report.json")


if __name__ == "__main__":
    main()
