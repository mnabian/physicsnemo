# Fused neighborhood attention benchmark

Benchmark note, measured 2026-10-20 and 2026-10-21 (UTC).

DeFormer's static-kNN processors (`NeighborhoodAttentionBlock` in
`physicsnemo/experimental/models/meshtransolver/local_attention.py`) can evaluate their
attention in two ways: the gather reference, which materializes `key[index]` and `value[index]`
as `[N, K, H, dh]` tensors (and float32 copies of both), or the fused kernel in
`physicsnemo/experimental/models/meshtransolver/fused_neighborhood_attention.py`, a
`torch.autograd.Function` with the same math that reads keys and values on the fly, keeps an
online softmax per node and head, and saves only the per-(node, head) log-sum-exp and the two
outputs for backward (Triton kernels on CUDA, a chunked pure-Torch backend on any device;
`backend="auto"` picks Triton when it imports). The switch is the block flag `fused`
(`fused_backend`), exposed as `MeshGeoFLARE(local_attention_fused=..., local_attention_backend=...)`
and as the Hydra keys `model.local_attention_fused` / `model.local_attention_backend`. The release
recipes (`examples/structural_mechanics/crash/conf/model/deformer_time_conditional.yaml`, used by
`yaris_deformer`) set `local_attention_fused: true`, so they train with the fused kernel by
default; the reference path remains selectable with `model.local_attention_fused=false`. The flag
adds no parameters, so checkpoints are interchangeable between the two paths. This note records
how much time and memory the fused kernel saves at the crash-mesh size, and why a
`torch.nn.attention.flex_attention` formulation was evaluated and not adopted.

## Benchmark setup

**Hardware.** One NVIDIA GB200 (189,471 MiB) of a four-GPU node, driver 580.126.20, CUDA 13.0
driver / 13.2 runtime. The other three GPUs were idle during the timed runs.

**Software.** NGC PyTorch container (26.04): Python 3.12.3, Torch `2.12.0a0+0291f960b6.nv26.04`,
Triton 3.6.0, Warp 1.14.0, torch-geometric 2.7.0.

**Mesh.** Synthetic, at the body-in-white crash-mesh size: N = 384,862 nodes as eight stacked quad shells
(156 x 309 each, raster-ordered node IDs, 1.5 spacing gap between shells), Euclidean
neighborhoods from `physicsnemo.nn.functional.neighbors.static_knn.euclidean_knn` with K in
{16, 32} neighbors plus the self slot, and 1,532,012 directed quad edges for the MPNN baseline.
The synthetic mesh matches the real mesh in N, K, D and H and has comparable neighbor locality
(IDs ordered along panels), not its part structure. Both scripts build it with `build_mesh` in
`neighborhood_attention.py` (deterministic for a given seed).

**Model dimensions.** Hidden width D = 256, H = 8 heads (dh = 32), edge latent E = 32, bf16
autocast with a float32 edge latent (the processor's edge encoder ends in LayerNorm, which
autocast keeps in float32).

**Timing.** Warm-up 2, then 5 timed repetitions for the block and update scopes; the functional
flex_attention benchmark used warm-up 1 and 3 repetitions. Each repetition is bracketed by CUDA
events with a device synchronization. The tables report the median and p95 (and min where
given) over repetitions and the maximum over repetitions of `torch.cuda.max_memory_allocated`,
reset before each repetition.

**Update scope.** One training update is `zero_grad`, a bf16-autocast rollout of four
transitions (BPTT-4) with MSE loss, `backward` and an optimizer step of
`rollout.MeshGeoFLAREAutoregressive` composed from the crash example's Hydra configs:
`crash_deformer_autoregressive` for the MPNN baseline (the validated MeshGraphNet-processor
DeFormer) and `crash_deformer_knn_autoregressive` with `model.local_knn_k=K` and
`model.local_neighborhood=euclidean` for the kNN variants, plus `model.local_attention_fused=true`
for the fused variant. FLARE backbone n_hidden 256, 6 layers, slice_num 128; 1 pre + 2 post
processor layers with per-layer checkpoint segments; `checkpoint_rollout=true`; no contact. The
optimizer is Muon + AdamW, which the composed recipe selects (`training.optimizer: muon`); the
JSON report records it in its `optimizer` field.

**Reproduction.** From the repository root on a CUDA machine (the update scope additionally
needs the crash example's dependencies, hydra and torch-geometric):

    python benchmarks/physicsnemo/experimental/meshtransolver/neighborhood_attention.py \
        --mode block --k 16 32 --impl reference fused --out block.json
    python benchmarks/physicsnemo/experimental/meshtransolver/neighborhood_attention.py \
        --mode update --k 16 32 --variant mpnn knn_reference knn_fused --out update.json
    python benchmarks/physicsnemo/experimental/meshtransolver/flex_neighborhood_attention.py \
        bandwidth --num-nodes 384862 --k 16 32
    python benchmarks/physicsnemo/experimental/meshtransolver/flex_neighborhood_attention.py check
    python benchmarks/physicsnemo/experimental/meshtransolver/flex_neighborhood_attention.py \
        bench --num-nodes 384862 --k 16 32 --bf16 --warmup 1 --reps 3

Each run prints, and with `--out` writes, a JSON report that includes the hardware and software
versions it ran with.

## Results

### Block level: one `NeighborhoodAttentionBlock` forward + backward

`neighborhood_attention.py --mode block`, GB200, bf16 autocast.

| K | Implementation | median s | p95 s | peak allocated GiB |
| --- | --- | --- | --- | --- |
| 16 | reference (gather) | 0.113 | 0.113 | 21.6 |
| 16 | fused (Triton) | 0.021 | 0.021 | 6.7 |
| 32 | reference (gather) | 0.216 | 0.216 | 40.1 |
| 32 | fused (Triton) | 0.030 | 0.030 | 9.2 |

Speedup 5.4x (K=16) and 7.2x (K=32); peak memory 3.2x and 4.4x lower. The fused peak is
dominated by the inputs the block receives (the float32 `[N, K, 32]` edge latent and its
gradient are 1.6 GiB each at K=32; bias, dlogits and P are 0.4 GiB each) rather than by the
attention itself.

### Update level: one BPTT-4 training update on the synthetic mesh

`neighborhood_attention.py --mode update`, GB200, bf16 autocast, Muon + AdamW.

| Variant | K | median s | p95 s | min s | peak allocated GiB |
| --- | --- | --- | --- | --- | --- |
| MPNN baseline (validated DeFormer) | - | 2.903 | 2.905 | 2.902 | 64.9 |
| kNN attention, gather reference | 16 | 4.802 | 5.052 | 4.801 | 77.5 |
| kNN attention, fused Triton | 16 | **2.860** | 2.861 | 2.858 | **61.7** |
| kNN attention, gather reference | 32 | 6.871 | 6.874 | 6.869 | 100.6 |
| kNN attention, fused Triton | 32 | **3.215** | 3.216 | 3.214 | **68.8** |

The fused path removes 1.94 s (K=16) and 3.66 s (K=32) from the reference update, 2.1x faster
end to end at K=32 with 31.8 GiB less peak memory. Against the MPNN baseline measured in the
same allocation, fused K=16 is 1.5 percent faster and uses 3.2 GiB less; fused K=32 is
11 percent (0.31 s) slower and uses 3.9 GiB more. The attention itself costs about 30 ms per
block forward + backward (table above), so the rest of the fused update at K=32 is the per-block
edge-bias projection and edge-latent gradient (two `[N, K, 32]` float32 tensors per block), the
edge encoder on `[N, K, 4]`, and the FFN and linear layers. On the real crash mesh the synthetic
reference costs are reproduced within about 15 percent (see Supplementary).

### Functional level: banded flex_attention vs the fused kernel

`flex_neighborhood_attention.py bench`, GB200, attention functional only (no linear layers or
FFN), forward + backward, bf16 projections, float32 edge latent, warm-up 1, 3 timed repetitions.

Formulation: reorder the nodes once with reverse Cuthill-McKee (RCM) so each node's neighbors
fall within a bandwidth B, build an `int8` slot table `[N, 2B+1]` (slot number of key
`kv = q + offset - B`, or -1), and hand `flex_attention` a `BlockMask` built directly from the
band with `mask_mod = slot >= 0` and a `score_mod` that adds `bias[q, slot, h]`. That is exact kNN
attention for the value term with autograd for q, k, v and the bias. The edge-value term
`W_e(concat_h sum_j p_j e_j)` cannot be expressed inside flex_attention (e_j is a per-pair value,
not a per-key value), so the script recovers the K real probabilities from the returned
log-sum-exp, `p_j = exp(q.k_j scale + b_j - lse)`, with one gathered-key pass (`[N, K, H, dh]`
keys, bf16) and aggregates `e_j` with them; the key gather the fused kernel avoids therefore
comes back.

| K | case | median ms | p95 ms | peak allocated GiB |
| --- | --- | --- | --- | --- |
| 16 | gather reference functional | 97.5 | 97.5 | 24.9 |
| 16 | fused Triton functional | 10.6 | 10.7 | 8.9 |
| 16 | flex_attention (banded, B=2,536) + lse edge pass | 515.0 | 699.0 | 23.6 |
| 32 | gather reference functional | 194.6 | 194.8 | 45.3 |
| 32 | fused Triton functional | 17.6 | 17.7 | 12.7 |
| 32 | flex_attention (banded, B=3,791) + lse edge pass | failed | | |

At K=32 the compiled flex kernel raised `CUDA error: an illegal memory access` inside Inductor's
autotuning for the 7,583-wide band at N = 384,862 (Torch 2.12.0a0, Triton 3.6.0), with all table
indices clamped and after the same formulation ran at K=16 and in the 3,000-node exactness
check; the failure was not investigated further.

Assessment: the banded flex_attention route is correct but not competitive for this operator.
With K+1 = 17 or 33 real neighbors out of a 5k-8k band it scores 230-300x more pairs than needed
and evaluates a table lookup per pair in `mask_mod` / `score_mod`; at K=16 it is 49x slower than
the fused Triton kernel and 5x slower than the gather reference, with no memory advantage over
the reference once the edge-value pass (which needs the gathered keys anyway) is included. It
would also require reordering every per-node tensor of the datapipe by the RCM permutation.
NATTEN does not apply (regular grids only). The fused Triton kernel does exactly K+1 reads per
node and needs no reordering. The script is kept so that the bandwidth of real neighborhoods can
be measured with the `bandwidth` sub-command on a datapipe's `knn_index` / `knn_mask` should the
banded approach be revisited (for example with a per-pair-value extension of flex_attention).

## Supplementary

### Band structure of the reordered neighborhoods

`flex_neighborhood_attention.py bandwidth`, CPU, synthetic mesh as above.

| K | ordering | B max | B p50 | B p99 | key blocks per query block (banded) | scored keys per query | redundancy vs K+1 | slot table |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | RCM on the kNN graph | 2,536 | 786 | 2,512 | 40.6 | 5,200 | 306x | 1.82 GiB |
| 32 | RCM on the kNN graph | 3,791 | 1,251 | 3,760 | 60.2 | 7,710 | 234x | 2.72 GiB |
| 16 | RCM on structural edges only | 48,517 | | | | | | |
| 32 | RCM on structural edges only | 96,720 | | | | | | |
| any | raster order (no reordering) | 48,823 / 97,026 | | | | | | |

B is a few thousand on this mesh, and only when the kNN graph itself is reordered (ordering by
structural edges does not help because Euclidean neighbors cross stacked panels). The exact
block-sparse pattern touches 19.0 / 22.6 key blocks per 128-row query block for K = 16 / 32
versus 40.6 / 60.2 for the full band. RCM of the K=32 kNN graph takes well under a second on a
laptop CPU (scipy). A real mesh has more parts and more complex connectivity than eight regular
sheets, so its B should be measured with the same sub-command before relying on a banded
approach.

### Exactness of the banded formulation

`flex_neighborhood_attention.py check`, 3,000 nodes, strict float32 (TF32 disabled). Uncompiled
`flex_attention` (math path) vs the gather reference: max abs diff 7e-7 (`out`, magnitude 4.0),
1e-6 (`edge_out`), and 1.4e-6 / 2.1e-6 / 1.4e-6 / 2.5e-6 / 1e-6 for the `query` / `key` / `value`
/ `bias` / `edge_latent` gradients (magnitudes 3-6), i.e. float32 round-off, including the
gradient of the bias captured by `score_mod`. The `torch.compile`d flex kernel differs from the
reference by 2.4e-3 (`out`) and 2.5e-3 / 3.1e-3 / 4.3e-3 (`query` / `key` / `bias` gradients)
while `edge_out` and the `value` / `edge_latent` gradients stay at 1e-6: the compiled template
runs its float32 dots in TF32 even with matmul precision `highest`, so a strict-float32 flex path
is not available in this build (irrelevant under the bf16 training recipe).

### Correctness checks of the fused kernel

The fused kernel is covered by `test/models/meshtransolver/test_local_attention.py`
(`pytest -k fused`): the Torch backend against the gather reference in float32 on CPU (forward,
input gradients and every parameter gradient), a float64 `gradcheck` of the manual backward with
non-power-of-two sizes and padded slots, fully masked rows, neighbor-order and padding invariance
of the fused processor, bitwise identical checkpoint segments, and on CUDA the Triton kernel
against the reference in strict float32 and under bf16 autocast plus a peak-memory ratio test.
Two measurement conditions matter for these comparisons. NGC images enable TF32 for float32
matmuls (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`), so the reference's float32 einsums run in TF32
while the Triton kernel computes exact float32; the CUDA float32 tests and the `check`
sub-command disable TF32 on both sides. Key and value gradients are accumulated with float32
atomics, so their last bits vary between runs; the strict-float32 CUDA tests use 2e-4
tolerances for that reason.

### Real crash mesh, full training update (profiling harness not included in the repository)

The numbers in this subsection were produced with a profiling harness that is not part of the
repository. They are not reproducible from the shipped scripts and are given only to calibrate
the synthetic-mesh results above. Same GPU type (one GB200 of a four-GPU node, otherwise idle),
driver 580.96; CPython 3.12, Torch 2.12.0+cu130, Triton 3.7.0, Warp 1.14.0, torch-geometric
2.7.0, torch_scatter 2.1.2. Data: the real body-in-white training split (127 cases, warm VTP and
geodesic-kNN caches), `datapipe.window_seed=42`, `training.reproducible_sampling=true`. Config
`crash_deformer_knn_autoregressive` with `model.local_neighborhood=geodesic`, plus
`model.local_attention_fused=true` for the fused rows; BPTT-4, bf16 autocast, Muon + AdamW; the
harness builds the real `Trainer` and times 6 updates after 2 warm-up updates with CUDA events
and `max_memory_allocated`.

| Variant (single GB200, N = 384,862) | K | median s / update | min - max s | peak allocated GB |
| --- | --- | --- | --- | --- |
| MPNN baseline | - | 3.409 | 3.389 - 3.414 | 79.2 |
| kNN reference | 16 | 4.880 | 4.873 - 4.890 | 83.9 |
| kNN fused (`model.local_attention_fused=true`) | 16 | **3.071** | 3.055 - 3.091 | **67.0** |
| kNN reference | 32 | 6.917 | 6.909 - 6.935 | 109.1 |
| kNN fused (`model.local_attention_fused=true`) | 32 | **3.443** | 3.432 - 3.459 | **74.9** |

On the real mesh the fused path removes 1.81 s (K=16, 1.59x) and 3.47 s (K=32, 2.01x) per
update and 17-34 GB of peak memory. Fused K=16 is 10 percent faster than the MPNN baseline;
fused K=32 is 34 ms (1 percent) slower, a gap outside the run-to-run spread of either
measurement (3.389-3.414 s vs 3.432-3.459 s). The synthetic mesh reproduces the reference costs
of this table within about 15 percent.

Equivalence on the same data (same weights, the fused flag toggled on one of two model copies,
first training window, 4 transitions, residual gates set to 0.3 / 0.2 because the recipe
initializes them at zero, which would hide the processors):

| K | comparison | max abs diff | relative L2 | reference magnitude |
| --- | --- | --- | --- | --- |
| 32 | trajectory, strict float32 (TF32 off) | 2.7e-5 | 5.3e-8 | 2.74 |
| 32 | trajectory, bf16 autocast (training path) | 6.4e-4 | 6.0e-5 | 2.74 |
| 32 | training-loss gradients, bf16, all 282 parameters | 1.9e-6 | 1.0e-3 | |
| 16 | trajectory, strict float32 | 2.1e-5 | 5.1e-8 | 2.74 |
| 16 | trajectory, bf16 autocast | 7.2e-4 | 6.5e-5 | 2.74 |
| 16 | training-loss gradients, bf16, all 282 parameters | 1.9e-6 | 1.5e-3 | |

Per parameter, the only gradients with a relative L2 above 0.4 are the three `edge_bias.bias`
vectors (8-17x), whose true gradient is exactly zero because the softmax is shift invariant per
head; both paths return bf16 noise there (absolute differences below 1.9e-6). The next-worst are
small bias vectors of the FLARE backbone at 0.2-0.4 relative, again with absolute differences at
the 1e-6 level. Under strict float32 the outputs agree to float32 round-off; under the bf16
recipe they differ at the bf16 noise floor, as expected from the kernel keeping the attention
weights in float32 where the reference rounds them to bf16. The losses on that window were
2.75917e-4 (reference) vs 2.75934e-4 (fused) at K=32 and 2.31156e-4 vs 2.31152e-4 at K=16, i.e.
6e-5 and 1.4e-5 relative.
