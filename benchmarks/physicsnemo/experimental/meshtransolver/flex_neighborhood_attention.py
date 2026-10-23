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
"""Evaluate ``torch.nn.attention.flex_attention`` as a fused path for static-kNN attention.

The idea: the neighborhoods are static per mesh, so reorder the nodes once with
reverse Cuthill-McKee so that every node's ``K`` neighbors lie within a bandwidth
``B`` of the diagonal, build a banded ``[N, 2B+1]`` slot table (``int8``: slot
index or -1), and give ``flex_attention`` a ``mask_mod`` (slot >= 0) and a
``score_mod`` that adds the per-slot head bias looked up through that table. The
attention probabilities and the value sum are then exact kNN attention computed
by FlashAttention-class block-sparse kernels. The edge-value term
``W_e(concat_h sum_j p_j e_j)`` cannot be expressed inside ``flex_attention``
(``e_j`` is a per-pair value); it is recovered exactly from the returned
log-sum-exp with one gathered-key pass, ``p_j = exp(q.k_j * scale + b_j - lse)``.

Sub-commands::

    bandwidth  --num-nodes 384862 --k 16 32          # CPU: RCM bandwidth statistics
    check                                            # CUDA: exactness vs the gather reference
    bench      --num-nodes 384862 --k 32             # CUDA: flex vs Triton vs reference functional

Import-safe (ASV discovery); everything runs under ``main``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _neighborhood_benchmark():
    """Import the sibling ``neighborhood_attention`` script (mesh, provenance).

    The directory goes on ``sys.path`` only when a sub-command needs it, so
    importing this module (ASV discovery) has no side effects.
    """
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import neighborhood_attention

    return neighborhood_attention


# --------------------------------------------------------------------------- #
# Reordering and band statistics                                               #
# --------------------------------------------------------------------------- #


def rcm_permutation(
    index: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse Cuthill-McKee order of the symmetrized kNN graph.

    Returns ``(perm, inverse)``: ``perm[new] = old`` and ``inverse[old] = new``.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    num_nodes, slots = index.shape
    rows = torch.arange(num_nodes).repeat_interleave(slots)[mask.reshape(-1)].numpy()
    cols = index.reshape(-1)[mask.reshape(-1)].numpy()
    adjacency = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_nodes, num_nodes)
    )
    adjacency = (adjacency + adjacency.T).tocsr()
    adjacency.data[:] = 1
    perm = torch.from_numpy(
        np.asarray(
            reverse_cuthill_mckee(adjacency, symmetric_mode=True), dtype=np.int64
        )
    )
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(num_nodes)
    return perm, inverse


def band_statistics(
    index: torch.Tensor, mask: torch.Tensor, inverse: torch.Tensor, block: int = 128
) -> dict:
    """Bandwidth of the reordered neighborhoods and the resulting block sparsity."""
    num_nodes = index.shape[0]
    new_query = inverse.unsqueeze(1).expand_as(index)
    new_key = inverse[index]
    span = (new_key - new_query).abs()[mask]
    bandwidth = int(span.max())
    quantiles = np.percentile(span.numpy(), [50, 90, 99, 99.9])
    # Block-sparse cost model: query blocks of `block` rows attend to every key
    # block that any of their rows touches (what a BlockMask materializes).
    query_block = new_query[mask] // block
    key_block = new_key[mask] // block
    pairs = torch.unique(query_block * ((num_nodes + block - 1) // block) + key_block)
    num_query_blocks = (num_nodes + block - 1) // block
    exact_pairs = int(pairs.numel())
    banded_blocks_per_row = (2 * bandwidth + 1) / block + 1
    valid_per_node = mask.sum(1).float().mean().item()
    return {
        "num_nodes": num_nodes,
        "slots": int(index.shape[1]),
        "valid_slots_per_node": valid_per_node,
        "bandwidth_max": bandwidth,
        "bandwidth_p50": float(quantiles[0]),
        "bandwidth_p90": float(quantiles[1]),
        "bandwidth_p99": float(quantiles[2]),
        "bandwidth_p999": float(quantiles[3]),
        "band_width_2B+1": 2 * bandwidth + 1,
        "block_size": block,
        "key_blocks_per_query_block_exact": exact_pairs / num_query_blocks,
        "key_blocks_per_query_block_banded": banded_blocks_per_row,
        "scored_keys_per_query_banded": banded_blocks_per_row * block,
        "redundancy_vs_knn": banded_blocks_per_row * block / valid_per_node,
        "slot_table_gib_int8": num_nodes * (2 * bandwidth + 1) / 2**30,
    }


def build_slot_table(
    index: torch.Tensor, mask: torch.Tensor, bandwidth: int
) -> torch.Tensor:
    """``[N, 2B+1]`` int8 table: the slot number of key ``kv = q + offset - B`` or -1.

    ``index`` must already be in the reordered node numbering.
    """
    num_nodes, slots = index.shape
    if slots > 127:
        raise ValueError("int8 slot table supports at most 127 slots")
    offsets = (
        index - torch.arange(num_nodes, device=index.device).unsqueeze(1) + bandwidth
    )
    if bool(((offsets < 0) | (offsets > 2 * bandwidth))[mask].any()):
        raise ValueError("a valid neighbor lies outside the band")
    table = torch.full(
        (num_nodes, 2 * bandwidth + 1), -1, dtype=torch.int8, device=index.device
    )
    rows = (
        torch.arange(num_nodes, device=index.device).unsqueeze(1).expand_as(index)[mask]
    )
    table[rows, offsets[mask]] = (
        torch.arange(slots, dtype=torch.int8, device=index.device)
        .unsqueeze(0)
        .expand_as(index)[mask]
    )
    return table


# --------------------------------------------------------------------------- #
# flex_attention formulation                                                   #
# --------------------------------------------------------------------------- #


def banded_block_mask(
    num_nodes: int, bandwidth: int, mask_mod, device, block: int = 128
):
    """BlockMask for the band, built directly from the bandwidth (no N x N pass)."""
    from torch.nn.attention.flex_attention import BlockMask

    num_blocks = (num_nodes + block - 1) // block
    first = torch.clamp((torch.arange(num_blocks) * block - bandwidth) // block, min=0)
    last = torch.clamp(
        (torch.arange(num_blocks) * block + block - 1 + bandwidth) // block,
        max=num_blocks - 1,
    )
    counts = last - first + 1
    width = int(counts.max())
    # from_kv_blocks sizes its dense [q_blocks, kv_blocks] scratch from the last
    # dimension of kv_indices, so pad every row to the full block count.
    kv_indices = torch.zeros((num_blocks, num_blocks), dtype=torch.int32)
    kv_indices[:, :width] = (
        (first.unsqueeze(1) + torch.arange(width).unsqueeze(0))
        .clamp(max=num_blocks - 1)
        .to(torch.int32)
    )
    kv_indices = kv_indices.view(1, 1, num_blocks, num_blocks).to(device)
    kv_num_blocks = counts.to(torch.int32).view(1, 1, num_blocks).to(device)
    kwargs = {}
    if "seq_lengths" in inspect.signature(BlockMask.from_kv_blocks).parameters:
        kwargs["seq_lengths"] = (num_nodes, num_nodes)
    return BlockMask.from_kv_blocks(
        kv_num_blocks, kv_indices, BLOCK_SIZE=block, mask_mod=mask_mod, **kwargs
    )


class FlexNeighborhoodAttention:
    """Exact kNN attention via flex_attention on RCM-ordered nodes plus the edge pass."""

    def __init__(self, index, mask, bandwidth, device, compile_kernel=True):
        from torch.nn.attention.flex_attention import flex_attention

        self.num_nodes, self.slots = index.shape
        self.bandwidth = bandwidth
        self.index, self.mask = index, mask
        self.slot_table = build_slot_table(index, mask, bandwidth)
        table, band, last = self.slot_table, bandwidth, self.num_nodes - 1

        # flex_attention evaluates the mods on padded rows/columns beyond N when N
        # is not a multiple of the block size, so every table index is clamped.
        def mask_mod(b, h, q_idx, kv_idx):
            q = q_idx.clamp(max=last)
            offset = kv_idx.clamp(max=last) - q + band
            inside = (
                (offset >= 0)
                & (offset <= 2 * band)
                & (q_idx <= last)
                & (kv_idx <= last)
            )
            slot = table[q, offset.clamp(0, 2 * band)]
            return inside & (slot >= 0)

        self.mask_mod = mask_mod
        self.block_mask = banded_block_mask(self.num_nodes, bandwidth, mask_mod, device)
        self.flex = (
            torch.compile(flex_attention, dynamic=False)
            if compile_kernel
            else flex_attention
        )

    def __call__(self, query, key, value, bias, edge_latent, scale):
        table, band, last = self.slot_table, self.bandwidth, self.num_nodes - 1

        def score_mod(score, b, h, q_idx, kv_idx):
            q = q_idx.clamp(max=last)
            offset = (kv_idx.clamp(max=last) - q + band).clamp(0, 2 * band)
            slot = table[q, offset].to(torch.long)
            return score + bias[q, slot.clamp(min=0), h].to(score.dtype)

        def to4(t):  # [N, H, dh] -> [1, H, N, dh]
            return t.permute(1, 0, 2).unsqueeze(0).contiguous()

        out, lse = self.flex(
            to4(query),
            to4(key),
            to4(value),
            score_mod=score_mod,
            block_mask=self.block_mask,
            scale=scale,
            return_lse=True,
        )
        out = out[0].permute(1, 0, 2)  # [N, H, dh]
        lse = lse[0].permute(1, 0)  # [N, H] float32
        # Edge-value term: per-pair values are outside flex_attention, so recover
        # the K real probabilities from the log-sum-exp with one gathered-key pass.
        num_nodes, heads, head_dim = query.shape
        gathered = key.index_select(0, self.index.reshape(-1)).view(
            num_nodes, self.slots, heads, head_dim
        )
        logits = torch.einsum("nhd,nkhd->nhk", query.float(), gathered.float()) * scale
        logits = logits + bias.float().permute(0, 2, 1)
        logits = logits.masked_fill(~self.mask.unsqueeze(1), float("-inf"))
        prob = torch.exp(logits - lse.unsqueeze(-1)).to(value.dtype)
        edge_out = torch.einsum("nhk,nke->nhe", prob, edge_latent.to(value.dtype))
        return out, edge_out


# --------------------------------------------------------------------------- #
# Reference functional (the gather math of NeighborhoodAttentionBlock)         #
# --------------------------------------------------------------------------- #


def reference_functional(query, key, value, bias, edge_latent, index, mask, scale):
    num_nodes, heads, head_dim = query.shape
    slots = index.shape[1]
    flat = index.reshape(-1)
    k = key.index_select(0, flat).view(num_nodes, slots, heads, head_dim)
    v = value.index_select(0, flat).view(num_nodes, slots, heads, head_dim)
    logits = torch.einsum("nhd,nkhd->nhk", query.float(), k.float()) * scale
    logits = logits + bias.float().permute(0, 2, 1)
    logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
    weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0).to(v.dtype)
    out = torch.einsum("nhk,nkhd->nhd", weights, v)
    edge_out = torch.einsum("nhk,nke->nhe", weights, edge_latent.to(weights.dtype))
    return out, edge_out


# --------------------------------------------------------------------------- #
# Sub-commands                                                                 #
# --------------------------------------------------------------------------- #


def cmd_bandwidth(args) -> dict:
    build_mesh = _neighborhood_benchmark().build_mesh

    report = {}
    for k in args.k:
        started = time.time()
        mesh = build_mesh(args.num_nodes, k, args.seed)
        index, mask = mesh["knn_index"], mesh["knn_mask"]
        built = time.time()
        perm, inverse = rcm_permutation(index, mask)
        ordered = time.time()
        stats = band_statistics(index, mask, inverse)
        # Structural-adjacency ordering, as an alternative to ordering the kNN graph.
        _, struct_inverse = rcm_permutation_edges(mesh["edge_index"], args.num_nodes)
        struct_stats = band_statistics(index, mask, struct_inverse)
        identity = band_statistics(index, mask, torch.arange(args.num_nodes))
        report[f"K={k}"] = {
            "mesh_layout": mesh["layout"],
            "seconds_mesh": built - started,
            "seconds_rcm_knn_graph": ordered - built,
            "rcm_on_knn_graph": stats,
            "rcm_on_structural_edges": struct_stats,
            "raster_order_no_reordering": identity,
        }
        print(
            f"K={k}: RCM(kNN) bandwidth max {stats['bandwidth_max']} p99 {stats['bandwidth_p99']:.0f}; "
            f"RCM(structural) max {struct_stats['bandwidth_max']}; raster max {identity['bandwidth_max']}; "
            f"banded redundancy x{stats['redundancy_vs_knn']:.1f}; slot table {stats['slot_table_gib_int8']:.2f} GiB",
            flush=True,
        )
    return report


def rcm_permutation_edges(edge_index: torch.Tensor, num_nodes: int):
    """RCM on an explicit (directed, will be symmetrized) edge list."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    rows, cols = edge_index[0].numpy(), edge_index[1].numpy()
    adjacency = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_nodes, num_nodes)
    )
    adjacency = (adjacency + adjacency.T).tocsr()
    adjacency.data[:] = 1
    perm = torch.from_numpy(
        np.asarray(
            reverse_cuthill_mckee(adjacency, symmetric_mode=True), dtype=np.int64
        )
    )
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(num_nodes)
    return perm, inverse


def _reordered_inputs(mesh, device, heads, head_dim, edge_dim, dtype, seed):
    """RCM-reorder the mesh and draw random projections in the new numbering."""
    index, mask = mesh["knn_index"], mesh["knn_mask"]
    perm, inverse = rcm_permutation(index, mask)
    index = inverse[index[perm]].to(device)
    mask = mask[perm].to(device)
    bandwidth = int(
        (index - torch.arange(index.shape[0], device=device).unsqueeze(1))
        .abs()[mask]
        .max()
    )
    num_nodes, slots = index.shape
    generator = torch.Generator(device=device).manual_seed(seed)

    def draw(*shape):
        return torch.randn(
            *shape, device=device, generator=generator, dtype=torch.float32
        ).to(dtype)

    query, key, value = (
        draw(num_nodes, heads, head_dim),
        draw(num_nodes, heads, head_dim),
        draw(num_nodes, heads, head_dim),
    )
    bias = draw(num_nodes, slots, heads)
    edge_latent = torch.randn(
        num_nodes, slots, edge_dim, device=device, generator=generator
    )  # float32 like the LN output
    return index, mask, bandwidth, query, key, value, bias, edge_latent


def strict_fp32() -> None:
    """Disable TF32 (NGC images enable it) so float32 checks compare true float32."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def cmd_check(args) -> dict:
    """fp32 exactness of the banded flex formulation against the gather reference."""
    build_mesh = _neighborhood_benchmark().build_mesh

    strict_fp32()
    device = torch.device(args.device)
    heads, head_dim, edge_dim = 4, 16, 8
    mesh = build_mesh(3000, 8, args.seed)
    index, mask, bandwidth, query, key, value, bias, edge_latent = _reordered_inputs(
        mesh, device, heads, head_dim, edge_dim, torch.float32, args.seed
    )
    scale = head_dim**-0.5
    flex = FlexNeighborhoodAttention(
        index, mask, bandwidth, device, compile_kernel=not args.no_compile
    )
    inputs = [
        t.clone().requires_grad_() for t in (query, key, value, bias, edge_latent)
    ]
    reference_inputs = [
        t.clone().requires_grad_() for t in (query, key, value, bias, edge_latent)
    ]
    weight_out = torch.randn(query.shape, device=device)
    weight_edge = torch.randn(query.shape[0], heads, edge_dim, device=device)
    out, edge_out = flex(*inputs, scale)
    (
        (out.float() * weight_out).sum() + (edge_out.float() * weight_edge).sum()
    ).backward()
    ref_out, ref_edge = reference_functional(*reference_inputs, index, mask, scale)
    (
        (ref_out.float() * weight_out).sum() + (ref_edge.float() * weight_edge).sum()
    ).backward()
    report = {
        "bandwidth": bandwidth,
        "num_nodes": int(index.shape[0]),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    for name, a, b in (("out", out, ref_out), ("edge_out", edge_out, ref_edge)):
        report[f"max_abs_diff_{name}"] = float(
            (a.detach().float() - b.detach().float()).abs().max()
        )
        report[f"max_abs_{name}"] = float(b.detach().float().abs().max())
    for name, a, b in zip(
        ("query", "key", "value", "bias", "edge_latent"), inputs, reference_inputs
    ):
        report[f"max_abs_diff_grad_{name}"] = (
            None
            if a.grad is None
            else float((a.grad.float() - b.grad.float()).abs().max())
        )
        report[f"max_abs_grad_{name}"] = (
            None if b.grad is None else float(b.grad.float().abs().max())
        )
    print(json.dumps(report, indent=2))
    return report


def _time(step, device, warmup, reps):
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
        "peak_allocated_gib": max(peaks) / 2**30,
        "warmup": warmup,
        "reps": reps,
    }


def _bench_k(args, k: int, device: torch.device) -> tuple[list, list]:
    """Time every available case at one ``K``; the tensors die with this frame."""
    from physicsnemo.experimental.models.meshtransolver.fused_neighborhood_attention import (
        neighborhood_attention,
        triton_available,
    )

    build_mesh = _neighborhood_benchmark().build_mesh
    heads, head_dim, edge_dim = args.heads, args.hidden_dim // args.heads, args.edge_dim
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    results, failures = [], []
    mesh = build_mesh(args.num_nodes, k, args.seed)
    index, mask, bandwidth, query, key, value, bias, edge_latent = _reordered_inputs(
        mesh, device, heads, head_dim, edge_dim, dtype, args.seed
    )
    stats = band_statistics(index.cpu(), mask.cpu(), torch.arange(index.shape[0]))
    print(
        f"K={k}: bandwidth after RCM {bandwidth}, banded redundancy x{stats['redundancy_vs_knn']:.1f}",
        flush=True,
    )
    scale = head_dim**-0.5
    leaves = [
        t.clone().requires_grad_() for t in (query, key, value, bias, edge_latent)
    ]
    weight_out = torch.randn(query.shape, device=device, dtype=torch.float32)
    weight_edge = torch.randn(
        query.shape[0], heads, edge_dim, device=device, dtype=torch.float32
    )

    def run(fn):
        def step():
            for leaf in leaves:
                leaf.grad = None
            out, edge_out = fn(*leaves)
            (
                (out.float() * weight_out).sum()
                + (edge_out.float() * weight_edge).sum()
            ).backward()

        return step

    cases = {
        "reference": lambda q, kk, v, b, e: reference_functional(
            q, kk, v, b, e, index, mask, scale
        ),
    }
    if triton_available(device):
        cases["triton"] = lambda q, kk, v, b, e: neighborhood_attention(
            q, kk, v, b, e, index, mask, scale, "triton"
        )
    try:
        flex = FlexNeighborhoodAttention(
            index, mask, bandwidth, device, compile_kernel=not args.no_compile
        )
        cases["flex+edge_pass"] = lambda q, kk, v, b, e: flex(q, kk, v, b, e, scale)
    except Exception as error:
        failures.append(
            {
                "case": f"flex:K={k}",
                "error": f"{type(error).__name__}: {str(error)[:400]}",
            }
        )
    for name, fn in cases.items():
        label = f"{name}:K={k}"
        try:
            started = time.time()
            result = _time(run(fn), device, args.warmup, args.reps)
            result.update(
                {
                    "case": name,
                    "k": k,
                    "bandwidth": bandwidth,
                    "dtype": str(dtype),
                    "wall_incl_compile_s": time.time() - started,
                }
            )
            results.append(result)
            print(
                f"{label}: median {result['median_s'] * 1e3:.1f} ms p95 {result['p95_s'] * 1e3:.1f} ms peak {result['peak_allocated_gib']:.2f} GiB",
                flush=True,
            )
        except Exception as error:
            torch.cuda.empty_cache()
            failures.append(
                {
                    "case": label,
                    "error": f"{type(error).__name__}: {str(error)[:400]}",
                }
            )
            print(
                f"{label}: FAILED {type(error).__name__}: {str(error)[:200]}",
                flush=True,
            )
    return results, failures


def cmd_bench(args) -> dict:
    """Functional-level forward+backward: flex (+edge pass) vs Triton vs reference."""
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    info = _neighborhood_benchmark().environment_info(device)
    print(json.dumps(info, indent=2))
    results, failures = [], []
    for k in args.k:
        k_results, k_failures = _bench_k(args, k, device)
        results.extend(k_results)
        failures.extend(k_failures)
        torch.cuda.empty_cache()  # the leaves and the flex module died with _bench_k
    return {"environment": info, "results": results, "failures": failures}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["bandwidth", "check", "bench"])
    parser.add_argument("--num-nodes", type=int, default=384_862)
    parser.add_argument("--k", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--edge-dim", type=int, default=32)
    parser.add_argument(
        "--bf16", action="store_true", help="bf16 projections (default float32)"
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = {"bandwidth": cmd_bandwidth, "check": cmd_check, "bench": cmd_bench}[
        args.command
    ](args)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
