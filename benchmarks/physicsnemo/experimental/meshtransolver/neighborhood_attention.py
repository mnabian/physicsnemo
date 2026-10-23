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
"""Single-GPU benchmark for DeFormer's static-kNN neighborhood attention.

Two scopes, each for the gather ``reference`` and the ``fused`` kernel:

* ``block``: one :class:`NeighborhoodAttentionBlock` forward + backward.
* ``update``: one full BPTT-4 training update (zero_grad, bf16-autocast
  rollout of four transitions, backward, optimizer step) of the crash example's
  ``MeshGeoFLAREAutoregressive`` composed from its Hydra configs; the ``mpnn``
  variant is the validated MeshGraphNet-processor DeFormer baseline and the
  ``knn_fused`` variant selects the kernel through the config key
  ``model.local_attention_fused=true``.

The mesh is synthetic at the body-in-white crash-mesh size (default 384,862 nodes): eight
stacked quad shells with raster-ordered node IDs, Euclidean reference
neighborhoods from :func:`physicsnemo.nn.functional.neighbors.static_knn.euclidean_knn`,
and bidirectional quad edges (about 4N directed edges) for the MPNN baseline.

Timing uses CUDA events around synchronized steps after warmup; the report has
median, p95, min and the peak allocated memory per case plus hardware and
software provenance. Run from the repository root, e.g.::

    python benchmarks/physicsnemo/experimental/meshtransolver/neighborhood_attention.py \
        --mode block --k 16 32 --impl reference fused --out block.json
    python benchmarks/physicsnemo/experimental/meshtransolver/neighborhood_attention.py \
        --mode update --k 32 --variant mpnn knn_reference knn_fused --out update.json

This module is import-safe (ASV discovery); everything runs under ``main``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
CRASH_EXAMPLE = REPO_ROOT / "examples" / "structural_mechanics" / "crash"

# Node statistics of the body-in-white crash dataset (train split), used to scale the
# synthetic rollout the way the real datapipe would.
BODY_IN_WHITE_NODE_STATS = {
    "pos_mean": [-1749.498779296875, -0.2648009955883026, 577.3088989257812],
    "pos_std": [1041.564208984375, 512.9486083984375, 336.8773498535156],
    "norm_vel_mean": [2.663264751434326, -0.0853300616145134, 0.3883599042892456],
    "norm_vel_std": [3.498845100402832, 1.226487159729004, 1.8240569829940796],
    "norm_acc_mean": [-78.9486083984375, -0.004784865770488977, -0.11517786979675293],
    "norm_acc_std": [137.16204833984375, 155.91893005371094, 274.14691162109375],
}


def _crash_example_on_path() -> None:
    """Make the crash example's helper modules importable (done lazily, once)."""
    crash_example = str(CRASH_EXAMPLE)
    if crash_example not in sys.path:
        sys.path.insert(0, crash_example)


# --------------------------------------------------------------------------- #
# Provenance                                                                   #
# --------------------------------------------------------------------------- #


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def _run_tool(executable: str, *arguments: str, cwd: Path | None = None) -> str:
    """Run a resolved executable with fixed arguments and return its stdout."""
    # The executable comes from ``shutil.which`` and the arguments are literals
    # chosen below, so the command line carries no untrusted input.
    completed = subprocess.run(  # noqa: S603
        [executable, *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    return completed.stdout.strip()


def environment_info(device: torch.device) -> dict:
    """Hardware and software provenance recorded with every report."""
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": _version("triton"),
        "warp": _version("warp"),
        "torch_geometric": _version("torch_geometric"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu_name": properties.name,
                "gpu_capability": f"{properties.major}.{properties.minor}",
                "gpu_memory_gib": round(properties.total_memory / 2**30, 1),
            }
        )
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            info["nvidia_smi"] = "unavailable (nvidia-smi not found)"
        else:
            try:
                info["nvidia_smi"] = _run_tool(
                    nvidia_smi,
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                    "-i",
                    str(device.index or 0),
                )
            except Exception as error:  # pragma: no cover - diagnostics only
                info["nvidia_smi"] = f"unavailable ({type(error).__name__})"
    git = shutil.which("git")
    if git is None:
        info["source"] = "unknown (git not found)"
    else:
        try:
            sha = _run_tool(git, "rev-parse", "HEAD", cwd=REPO_ROOT)
            dirty = _run_tool(git, "status", "--short", cwd=REPO_ROOT)
            info["source"] = f"{sha} ({'dirty' if dirty else 'clean'})"
        except Exception:
            info["source"] = "unknown"
    return info


# --------------------------------------------------------------------------- #
# Synthetic body-in-white-sized shell mesh                                     #
# --------------------------------------------------------------------------- #


def build_mesh(num_nodes: int, k: int, seed: int, sheets: int = 8) -> dict:
    """Stacked quad shells with raster-ordered IDs, quad edges and Euclidean kNN."""
    from physicsnemo.nn.functional.neighbors.static_knn import euclidean_knn

    _crash_example_on_path()
    from local_neighborhoods import knn_edge_features, with_self_slot

    generator = torch.Generator().manual_seed(seed)
    per_sheet = math.ceil(num_nodes / sheets)
    rows = max(2, math.ceil(math.sqrt(per_sheet / 2)))
    cols = max(2, math.ceil(per_sheet / rows))
    spacing, gap = 1.0, 1.5
    sheet_id, row, col = torch.meshgrid(
        torch.arange(sheets), torch.arange(rows), torch.arange(cols), indexing="ij"
    )
    sheet_id, row, col = sheet_id.reshape(-1), row.reshape(-1), col.reshape(-1)
    total = sheets * rows * cols
    warp = (
        0.2
        * spacing
        * torch.sin(2 * math.pi * col / cols)
        * torch.cos(2 * math.pi * row / rows)
    )
    pos = torch.stack(
        (col * spacing, row * spacing, sheet_id * gap + warp), dim=-1
    ).float()
    pos = pos + (torch.rand(total, 3, generator=generator) - 0.5) * 0.1 * spacing
    pos = pos[:num_nodes]  # IDs are raster-ordered, so this drops a sheet tail

    ids = torch.arange(total).view(sheets, rows, cols)
    right = torch.stack((ids[:, :, :-1].reshape(-1), ids[:, :, 1:].reshape(-1)))
    down = torch.stack((ids[:, :-1, :].reshape(-1), ids[:, 1:, :].reshape(-1)))
    undirected = torch.cat((right, down), dim=1)
    keep = (undirected < num_nodes).all(dim=0)
    undirected = undirected[:, keep]
    edge_index = torch.cat((undirected, undirected.flip(0)), dim=1).contiguous()
    displacement = pos[edge_index[0]] - pos[edge_index[1]]
    edge_attr = torch.cat(
        (displacement, displacement.norm(dim=-1, keepdim=True)), dim=-1
    )
    edge_attr = (edge_attr - edge_attr.mean(0)) / (edge_attr.std(0) + 1e-8)

    knn = euclidean_knn(pos, k)
    knn_index, knn_mask = with_self_slot(knn)
    knn_edge_attr = knn_edge_features(pos, knn_index)
    valid = knn_edge_attr[knn_mask]
    knn_edge_attr = (knn_edge_attr - valid.mean(0)) / (valid.std(0) + 1e-8)
    # Normalized coordinates as the datapipe would hand them to the model.
    coords = (pos - pos.mean(0)) / (pos.std(0) + 1e-8)
    return {
        "coords": coords.contiguous(),
        "edge_index": edge_index,
        "edge_attr": edge_attr.contiguous(),
        "knn_index": knn_index.contiguous(),
        "knn_mask": knn_mask.contiguous(),
        "knn_edge_attr": knn_edge_attr.contiguous(),
        "layout": {
            "sheets": sheets,
            "rows": rows,
            "cols": cols,
            "num_edges": int(edge_index.shape[1]),
        },
    }


# --------------------------------------------------------------------------- #
# Timing                                                                       #
# --------------------------------------------------------------------------- #


def measure_steps(step, device: torch.device, warmup: int, reps: int) -> dict:
    """CUDA-event timing of ``step`` after ``warmup`` untimed calls."""
    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    seconds, peaks = [], []
    for _ in range(reps):
        torch.cuda.reset_peak_memory_stats(device)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        step()
        end.record()
        torch.cuda.synchronize(device)
        seconds.append(start.elapsed_time(end) / 1000.0)
        peaks.append(torch.cuda.max_memory_allocated(device))
    return {
        "median_s": statistics.median(seconds),
        "p95_s": float(np.percentile(seconds, 95)),
        "min_s": min(seconds),
        "max_s": max(seconds),
        "mean_s": statistics.fmean(seconds),
        "samples_s": seconds,
        "peak_allocated_gib": max(peaks) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "warmup": warmup,
        "reps": reps,
    }


# --------------------------------------------------------------------------- #
# Block benchmark                                                              #
# --------------------------------------------------------------------------- #


def bench_block(args, mesh: dict, k: int, impl: str, device: torch.device) -> dict:
    from physicsnemo.experimental.models.meshtransolver.local_attention import (
        NeighborhoodAttentionBlock,
    )

    torch.manual_seed(args.seed)
    block = NeighborhoodAttentionBlock(
        args.hidden_dim,
        args.heads,
        args.edge_dim,
        fused=(impl == "fused"),
        fused_backend=args.backend,
    ).to(device)
    num_nodes, slots = mesh["knn_index"].shape
    index = mesh["knn_index"].to(device)
    mask = mesh["knn_mask"].to(device)
    x = torch.randn(num_nodes, args.hidden_dim, device=device, requires_grad=True)
    # The processor's edge encoder ends in LayerNorm, which autocast keeps in
    # float32, so the per-slot edge latent reaching a block is float32.
    edge_latent = torch.randn(
        num_nodes, slots, args.edge_dim, device=device, requires_grad=True
    )

    def step():
        block.zero_grad(set_to_none=True)
        x.grad = None
        edge_latent.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.autocast):
            out = block(x, index, mask, edge_latent)
        out.float().square().mean().backward()

    result = measure_steps(step, device, args.warmup, args.reps)
    result.update(
        {"scope": "block", "impl": impl, "k": k, "slots": slots, "num_nodes": num_nodes}
    )
    # The block and its tensors are released when this frame returns; the
    # caller empties the CUDA cache before the next case.
    return result


# --------------------------------------------------------------------------- #
# Full BPTT-4 training update                                                  #
# --------------------------------------------------------------------------- #


def _compose_model_config(variant: str, k: int, backend: str):
    from hydra import compose, initialize_config_dir

    if variant == "mpnn":
        name, overrides = "crash_deformer_autoregressive", []
    else:
        name = "crash_deformer_knn_autoregressive"
        overrides = [f"model.local_knn_k={k}", "model.local_neighborhood=euclidean"]
        if variant == "knn_fused":
            # The supported switch: the same keys the release recipes set.
            overrides += [
                "model.local_attention_fused=true",
                f"model.local_attention_backend={backend}",
            ]
    with initialize_config_dir(
        config_dir=str(CRASH_EXAMPLE / "conf"), version_base="1.3"
    ):
        return compose(config_name=name, overrides=overrides)


def _build_optimizer(model, cfg):
    name = str(cfg.training.optimizer)
    if name == "muon":
        from utils import build_muon_optimizer  # crash example helper

        return build_muon_optimizer(model, cfg), "muon+adamw"
    if name == "adamw":
        return (
            torch.optim.AdamW(
                model.parameters(),
                lr=cfg.training.start_lr,
                weight_decay=cfg.training.optimizer_weight_decay,
                fused=True,
            ),
            "adamw",
        )
    return torch.optim.Adam(
        model.parameters(), lr=cfg.training.start_lr, fused=True
    ), "adam"


def bench_update(args, mesh: dict, k: int, variant: str, device: torch.device) -> dict:
    _crash_example_on_path()
    from datapipe import SimSample  # crash example datapipe
    from hydra.utils import instantiate
    from torch_geometric.data import Data

    cfg = _compose_model_config(variant, k, args.backend)
    torch.manual_seed(args.seed)
    model = instantiate(cfg.model).to(device).train()
    optimizer, optimizer_name = _build_optimizer(model, cfg)

    num_nodes = mesh["coords"].shape[0]
    coords = mesh["coords"].to(device)
    graph = Data(
        num_nodes=num_nodes,
        edge_index=mesh["edge_index"].to(device),
        edge_attr=mesh["edge_attr"].to(device),
    )
    if variant != "mpnn":
        graph.knn_index = mesh["knn_index"].to(device)
        graph.knn_mask = mesh["knn_mask"].to(device)
        graph.knn_edge_attr = mesh["knn_edge_attr"].to(device)
        graph.knn_metric = "euclidean"
    stats = {
        "node": {
            key: torch.tensor([value], device=device)
            for key, value in BODY_IN_WHITE_NODE_STATS.items()
        }
    }
    dt = float(cfg.model.dt)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    velocity = torch.randn(num_nodes, 3, device=device, generator=generator) * 0.05
    thickness = torch.randn(num_nodes, 1, device=device, generator=generator)
    steps = args.bptt
    drift = (
        torch.arange(1, steps + 1, device=device).view(1, steps, 1)
        * dt
        * velocity.unsqueeze(1)
    )
    target = (
        coords.unsqueeze(1)
        + drift
        + 0.01 * torch.randn(num_nodes, steps, 3, device=device, generator=generator)
    )
    sample = SimSample(
        node_features={
            "coords": coords,
            "features": thickness,
            "previous_coords": coords - velocity * dt,
        },
        node_target=target,
        graph=graph,
    )
    criterion = torch.nn.MSELoss()

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.autocast):
            prediction = model(sample=sample, data_stats=stats)
            loss = criterion(prediction, sample.node_target)
        loss.backward()
        optimizer.step()

    result = measure_steps(step, device, args.warmup, args.reps)
    result.update(
        {
            "scope": "update",
            "variant": variant,
            "k": k if variant != "mpnn" else None,
            "num_nodes": num_nodes,
            "num_edges": int(graph.edge_index.shape[1]),
            "bptt_steps": steps,
            "optimizer": optimizer_name,
            "config": "crash_deformer_autoregressive"
            if variant == "mpnn"
            else "crash_deformer_knn_autoregressive",
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    )
    # The model, optimizer and sample are released when this frame returns; the
    # caller empties the CUDA cache before the next case.
    return result


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _print_table(results: list[dict]) -> None:
    header = f"{'scope':7s} {'case':14s} {'K':>3s} {'median s':>10s} {'p95 s':>10s} {'min s':>10s} {'peak GiB':>9s}"
    print(header)
    print("-" * len(header))
    for row in results:
        case = row.get("impl") or row.get("variant")
        k = "" if row.get("k") is None else str(row["k"])
        print(
            f"{row['scope']:7s} {case:14s} {k:>3s} {row['median_s']:10.3f} {row['p95_s']:10.3f} "
            f"{row['min_s']:10.3f} {row['peak_allocated_gib']:9.1f}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["block", "update"], required=True)
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=384_862,
        help="body-in-white crash mesh size by default",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[16, 32],
        help="neighbors per node (excluding self)",
    )
    parser.add_argument(
        "--impl",
        nargs="+",
        default=["reference", "fused"],
        choices=["reference", "fused"],
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        default=["mpnn", "knn_reference", "knn_fused"],
        choices=["mpnn", "knn_reference", "knn_fused"],
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--edge-dim", type=int, default=32)
    parser.add_argument("--bptt", type=int, default=4)
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "triton", "torch"]
    )
    parser.add_argument(
        "--no-autocast",
        dest="autocast",
        action="store_false",
        help="disable bf16 autocast",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=None, help="JSON report path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("this benchmark needs a CUDA device")
    torch.cuda.set_device(device)
    info = environment_info(device)
    print(json.dumps(info, indent=2))
    results, failures = [], []
    for k in args.k:
        started = time.time()
        mesh = build_mesh(args.num_nodes, k, args.seed)
        print(
            f"mesh K={k}: {mesh['layout']} built in {time.time() - started:.1f}s",
            flush=True,
        )
        cases = args.impl if args.mode == "block" else args.variant
        for case in cases:
            if args.mode == "update" and case == "mpnn" and k != args.k[0]:
                continue  # the MPNN baseline does not depend on K
            label = f"{args.mode}:{case}:K={k}"
            print(f"running {label}", flush=True)
            try:
                if args.mode == "block":
                    result = bench_block(args, mesh, k, case, device)
                else:
                    result = bench_update(args, mesh, k, case, device)
            except torch.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                failures.append({"case": label, "error": f"OOM: {str(error)[:300]}"})
                print(f"{label}: out of memory", flush=True)
                continue
            torch.cuda.empty_cache()  # the case's tensors died with its frame
            results.append(result)
            print(
                f"{label}: median {result['median_s']:.3f}s p95 {result['p95_s']:.3f}s "
                f"peak {result['peak_allocated_gib']:.1f} GiB",
                flush=True,
            )
    _print_table(results)
    report = {
        "environment": info,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "results": results,
        "failures": failures,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 1 if failures and not results else 0


if __name__ == "__main__":
    sys.exit(main())
