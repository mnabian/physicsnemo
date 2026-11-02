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
"""Summarise a ``dump_yaris_predictions.py`` case directory: per-frame vector
RMSE of every model trajectory and target-relative shell-crossing statistics
(the fraction of kept nodes whose predicted mid-surface crossing depth exceeds
a tolerance where the ground truth's does not: "spurious" crossings), so the
numbers behind an exported viewer bundle are stated from the same arrays the
bundle was built from.

    python -m visualization.ovrtx.dump_summary <case dir> [--tolerance-mm 0.5] [--frames 0-24]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .scene import parse_frame_spec


def summarize_case(
    case_dir, tolerance_mm: float = 0.5, frame_spec: str | None = "all"
) -> dict:
    case_dir = Path(case_dir)
    truth_dir = case_dir / "ground_truth"
    truth = np.load(truth_dir / "positions.npy", mmap_mode="r")
    truth_depth = np.load(truth_dir / "depth_crossing.npy", mmap_mode="r")
    frames = parse_frame_spec(frame_spec, int(truth.shape[0]))
    meta = json.loads((truth_dir / "meta.json").read_text())
    time_ms = [float(meta["time_ms"][f]) for f in frames]
    report = {
        "case": meta.get("case", case_dir.name),
        "frames": frames,
        "time_ms": time_ms,
        "kept_nodes": int(truth.shape[1]),
        "tolerance_mm": tolerance_mm,
        "models": {},
    }
    truth_cross = np.stack(
        [np.asarray(truth_depth[f], dtype=np.float32) > tolerance_mm for f in frames]
    )
    report["ground_truth"] = {
        "crossing_fraction_per_frame": truth_cross.mean(axis=1).tolist(),
        "max_crossing_depth_mm_per_frame": [
            float(np.asarray(truth_depth[f]).max()) for f in frames
        ],
    }
    for label_dir in sorted(
        p
        for p in case_dir.iterdir()
        if p.is_dir() and p.name != "ground_truth" and (p / "positions.npy").exists()
    ):
        positions = np.load(label_dir / "positions.npy", mmap_mode="r")
        depth = np.load(label_dir / "depth_crossing.npy", mmap_mode="r")
        rmse, spurious, missed, excess_max = [], [], [], []
        for k, f in enumerate(frames):
            error = np.asarray(positions[f], dtype=np.float64) - np.asarray(
                truth[f], dtype=np.float64
            )
            rmse.append(float(np.sqrt((error**2).sum(axis=1).mean())))
            pred_cross = np.asarray(depth[f], dtype=np.float32) > tolerance_mm
            spurious.append(float((pred_cross & ~truth_cross[k]).mean()))
            missed.append(
                float(
                    (~pred_cross & truth_cross[k]).sum() / max(1, truth_cross[k].sum())
                )
            )
            excess_max.append(
                float(
                    np.clip(
                        np.asarray(depth[f], dtype=np.float32)
                        - np.asarray(truth_depth[f], dtype=np.float32),
                        0,
                        None,
                    ).max()
                )
            )
        label_meta = (
            json.loads((label_dir / "meta.json").read_text())
            if (label_dir / "meta.json").exists()
            else {}
        )
        report["models"][label_dir.name] = {
            "vector_RMSE_mm_per_frame": rmse,
            "vector_RMSE_mm_pooled": float(np.sqrt(np.mean(np.square(rmse)))),
            "dump_vector_RMSE_mm": label_meta.get("vector_RMSE_mm"),
            "spurious_crossing_fraction_per_frame": spurious,
            "spurious_crossing_fraction_mean": float(np.mean(spurious)),
            "missed_crossing_fraction_per_frame": missed,
            "max_excess_crossing_depth_mm_per_frame": excess_max,
            "checkpoint": label_meta.get("checkpoint_dir"),
            "epoch": label_meta.get("epoch"),
        }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--tolerance-mm", type=float, default=0.5)
    parser.add_argument("--frames", default="all")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    report = summarize_case(args.case_dir, args.tolerance_mm, args.frames)
    text = json.dumps(report, indent=1)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(
        f"case {report['case']}: {report['kept_nodes']} kept nodes, frames {report['frames'][0]}..{report['frames'][-1]}"
    )
    for name, model in report["models"].items():
        print(
            f"  {name:12s} pooled RMSE {model['vector_RMSE_mm_pooled']:.3f} mm (dump {model['dump_vector_RMSE_mm']}); "
            f"spurious crossings > {report['tolerance_mm']} mm: mean {100 * model['spurious_crossing_fraction_mean']:.4f} % of nodes, "
            f"last frame {100 * model['spurious_crossing_fraction_per_frame'][-1]:.4f} %"
        )
    print(
        f"  ground truth crossing fraction, last frame: {100 * report['ground_truth']['crossing_fraction_per_frame'][-1]:.4f} %"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
