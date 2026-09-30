# Surface-contact implementation

## Material-surface correction (2026-09-22)

The predictive recipe now explicitly opts into `contact_surface_material_fan`.
The earlier AGA smokes passed all 175 unit tests and short TBPTT/replay, but
their 24-step rollouts produced forecast quads failing the **projected simple
polygon** policy. That policy is unsuitable as a mandatory condition on every
untrained model prediction. The strict endpoint/functional defaults remain
available and retain their errors; this is a new surface contract, not a claim
that the failed predicted polygons were valid.

- Validate every facet of the supplied, observed **start-of-window** mesh;
  invalid reference quads/triangles, nonfinite coordinates, missing thickness,
  bad connectivity and candidate-budget overflow still fail explicitly.
- Preserve the same arithmetic-center material fan and original-vertex weights
  throughout the rollout. Its four triangles are transported with the nodal
  coordinates. The deformed surface is the union of these triangles, including
  folds/overlap, rather than a newly inferred simple polygon in its area-normal
  projection. No adaptive diagonal swap or dropped facet is involved.
- A numerically collapsed material triangle projects onto its closed edges;
  coincident edges become points. This is exact at collapse and an explicit
  floating-point near-degeneracy approximation. Its returned normal is zero;
  the contact model still uses its regularized separation direction, not this
  triangle normal. Geometry selection remains discrete and selected values/
  barycentrics retain gradients. Nondifferentiable ties are not smoothed away.
- Current and forecast-end projections use this same representation, with the
  existing two-anchor causal activation. The reference check reads only observed
  input coordinates, never a future target. Strict and material projections are
  identical on supported nondegenerate center fans, including their gradients.

This extension is **not** a solver inversion repair, nonpenetration constraint,
friction law, general bilinear-quad projector or full CCD. Highly distorted
learned trajectories remain a model-quality problem; smoke diagnostics report
non-simple projected fans instead of concealing them. No FLARE/MLP parameters,
time step, optimizer or TBPTT horizon changed. New GPU/full-rollout validation
results must be reported separately from the historical results below.

This branch starts at `9266d9fd26e5eef6e175a844132ea8aaea2d25df` and does not
modify any existing run snapshot. No training submission is part of this change.

## Scope

Implement a GPU-capable, two-sided **node–facet** contact detector, using
conservative expanded surface bounds, exact closest-point geometry, supplied
thickness, incidence exclusions, and surface-interpolated latent messages.
Triangles and quads are supported. Quads use four triangles around the arithmetic
center, matching the pinned TYPE7 geometric decomposition. GM's `[a,b,c,c]`
triangle encoding is preserved, including facet IDs and PyG batch offsets.
Keep the FLARE backbone, contact MLP parameters, optimizer, time step and TBPTT
window unchanged. Keep the existing nearest-k implementation available.

OpenRadioss reference: commit
`546129947c7569ce2f6c61ffb3435bf9e7db0a2a`, TYPE7 candidate and penetration
routines. The production implementation is independent; upstream AGPL source
must stay outside the Apache-licensed package and is only an optional external
validation oracle. This is not a full OpenRadioss/LS-DYNA interface port.

## Acceptance gates

- Conservative broad-phase recall, including large faces with distant vertices.
- Correct face, edge, corner, two-sided, thickness, incidence and batch behavior.
- Explicit errors for invalid/degenerate geometry and exhausted candidate budgets.
- Live barycentric/closest-point gradients, empty-contact behavior, checkpointed
  BPTT equivalence and unchanged legacy outputs.
- Between-frame crossing diagnostic under an explicitly linear motion model;
  unresolved temporal intervals must never be reported as certified no-contact.
- CPU reference and GPU agreement; pinned OpenRadioss triangle/quad oracle comparison.
- Real-data geometry/memory inspection before authorizing a future training run.

## Boundaries

No hard force law, friction, nonpenetration projection, edge–edge contact,
erosion, or bilinear/curved-shell equivalence is claimed. Original solver decks
are needed to match surface sets, facet thickness, exclusions and initial-gap
handling. Nodal-thickness interpolation is a documented dataset adapter, not
evidence of equivalence to the original deck. Conservative swept bounds alone
do not prevent tunneling; the temporal diagnostic is distinct from the learned
time integrator. No automatic adjustment to the 5 ms prediction interval.

## Implementation and assumptions

Select the opt-in crash config
`gm_crash_deformer_surface_contact_autoregressive_tbptt`. Nothing in the existing
nearest-k recipe or submitted runs selects this backend automatically.

1. `node_triangle_candidates` in `physicsnemo/nn/functional/neighbors/surface_contact.py`
   supports three/four-node facets and enumerates every overlapping node/facet
   bound using a GPU Warp BVH. It excludes incidence, cross-batch pairs, and
   explicitly supplied node/facet exclusions. No top-k cap, component-wide or
   blanket hop exclusion. Bounds cover whole facets, with outward rounding.
2. The expanded boxes use half the node thickness + 5 mm node padding and half
   the maximum facet nodal thickness. That conservatively encloses the narrow
   phase's interpolated thickness. Broad-phase budget overflow raises before
   allocating pair messages; no silent truncation. The GM config budget is 4M
   pairs, informed by the Run100 diagnostic, not a guarantee for every case.
3. Narrow phase uses the closest point `q`, original-vertex weights `b`, distance
   `d = |q-x_i|`, and shell clearance
   `g = d - (t_i + sum(b_j*t_j))/2`. This is an unsigned mid-surface distance
   minus thickness, not a history-dependent penetration depth. Degenerate
   candidate facets raise explicitly. Quads must have a simple projection along
   their area normal, with the arithmetic center strictly inside the polygon's
   visibility kernel: all four projected fan triangles must have positive area.
   This admits mildly concave centroid-star-shaped quads without overlapping or
   filling a notch. Every unique quad is checked on each contact build, including
   distant facets; unsupported shapes raise with facet IDs. Warped supported
   quads retain the same center-fan representation.
4. Broad phase and narrow-phase membership are detached. Surviving geometric
   quantities are recomputed live to avoid retaining autograd state for rejected
   candidates. Surface source latents and velocities are interpolated with the
   live weights before the unchanged contact MLP. Destination is the query node;
   there is no equal/opposite force law or conserved-momentum guarantee.
5. The 12 channels remain `(q-x_i)/10, d/10, g/10, n, dv/1000, dot(dv,n)/1000`.
   Here `n=(q-x_i)/sqrt(|q-x_i|^2 + 0.1^2)` is a regularized separation direction,
   including edge/corner regions—not an oriented finite-element shell normal.
   Weights are `(1-clamp(max(g,0)/5,0,1)^2)^2`. All lengths are dataset mm.
   Negative gap is in the thickness band; positive gap under 5 mm is proximity.
6. The recipe requires supplied nodal thickness and element connectivity. Without
   the original deck we assume all exported facets are eligible, incidence-only
   exclusions unless explicitly provided, no initial-gap correction, and no
   special tied/eroding/frictional interfaces. These cannot be inferred reliably
   from this VTP export. A new implementation is not a claim of accuracy gain.
7. `swept_node_triangle_check` is a separate, triangle-only diagnostic for linear
   motion between frames. It returns witnessed hits, separated intervals, or
   **unresolved** intervals when its budget is exhausted. It does not run inside
   the training recipe, supply missing labels, resolve tunneling, or alter the
   integration time step. Edge–edge and fully coupled CCD are future work.

## Validation (2026-09-21)

Dedicated branch/worktree; GPU checks ran on `mnabian-dev`, L40 GPU 0
`GPU-c611b5d0-c900-2226-afd9-9edee9a39f0e`, Torch 2.11.0+cu128, Warp 1.14.0.
The GPU-host skill guided read-only preflight and an isolated test environment;
no shared environment, training job, or existing source snapshot was changed.

- **125 tests passed** on the L40 host (CPU and CUDA cases, no skips). Local Mac:
  81 passed, 44 CUDA-only skips. Tests cover geometric regions, two-sidedness, barycentric gradcheck,
  topology/cache/batching, quad encoding, dense contacts beyond 16 neighbors,
  budget failures, empty graphs, incidence/exclusions, whole-facet recall,
  triangle/quad checkpointed multi-step BPTT, BF16 autocast backward, non-default
  CUDA streams, and legacy contact regressions. New/core changed code passes Ruff;
  pre-existing datapipe/rollout lint findings were not included in this change.
- External compiled OpenRadioss TYPE7 oracle: **98,304 unique cases**, each checked
  on CPU and CUDA, including triangles, warped convex quads, and mixed batches.
  Maximum FP64 discrepancy in its squared zone-of-influence score: **9.77e-14**.
  This is geometric validation, not solver force/contact-setting equivalence.
- Full GM **Run100**: 384,862 nodes; 377,286 facets (27,910 triangles, 349,376 quads).
  Input SHA256: `d27cbed5fc1a9422146708efd012e82d4e4a664a0c4c497063ede23d4ac97400`.
  At frames 1/6/12/25, broad pairs were 2,150,990 / 2,213,712 / 2,242,348 /
  2,264,055. Narrow-band pairs were 1,006,866 / 998,810 / 996,646 / 993,020.
  Only 1,307 / 1,906 / 2,087 / 1,875 had nonpositive shell clearance; do not
  describe all proximity messages as actual physical contact.
- Full-case geometry-only backward at frame 25 passed with finite, nonzero
  position/velocity gradients. Observed Torch peak was about 3.82 GiB, **not**
  end-to-end model/TBPTT memory. These single-case timings are diagnostics, not
  a controlled throughput benchmark.

Evidence and the external oracle harness are in the sibling task artifact folder
`../artifacts/surface-contact-20260921/`; solver source remains outside this repo.
Before any future training launch: audit more cases, perform full-width model
forward/backward memory smoke at the configured TBPTT horizon, and review the
explicit contact assumptions. No training submission is part of this change.

## Independent-review follow-up (2026-09-21)

- Surface data now rejects missing thickness even when it is not a requested
  static input feature. The surface rollout preserves supplied zero thickness
  and never substitutes the legacy fallback. Non-surface behavior is unchanged.
- Source interpolation accumulates one vertex at a time, avoiding the unused
  placeholder gather and the `E x 4 x H` forward intermediates. Half/BF16 products
  accumulate in FP32. Values and gradients are tested against the dense reference;
  low-precision scatter summation order can change rounding, so bitwise backward
  equivalence to the old implementation is not claimed.
- Triangle projection uses scale-normalized local coordinates. Regression tests
  cover FP32 values and finite gradients at scales `1e-10`, `1`, and `1e10`.
- Fresh local contact regressions: **91 passed, 52 CUDA-only skipped**. The new
  GPU/full-width smoke results are separate; prior GPU results above apply to
  the pre-review commit, not automatically to these fixes.
- Run100 passes the quad-quality policy at all 26 exported frames. A temporal
  audit found a node–quad pair outside the 5 mm band at both 10 and 15 ms, but
  with negative clearance at the linearly interpolated midpoint. This is a
  witnessed limitation of the geometric/time-sampling surrogate, not evidence
  of the original solver's continuous trajectory. Current-position detection
  and a separate swept diagnostic do not close this gap. Swept broad-phase
  bounds alone would not fix current-gap membership/cutoff either.
- The first GPU review job passed all 141 then-current regressions but stopped
  before the model smoke: Run106 facet 39147 becomes mildly concave in projection
  at 20 ms. Its center-fan remains consistently oriented and nondegenerate. The
  initially proposed strict-convexity guard was therefore too restrictive and
  was refined to the actual nonoverlapping, centroid-star-shaped fan condition.
  A synthetic concave-notch regression verifies that this does not fill missing
  polygon regions; the original pathological concave/bow-tie cases still raise.

Long training remains gated on resource and temporal-method review. A bounded
single-GPU smoke may test implementation viability without claiming temporal
coverage or authorizing a training campaign. Evidence/scripts are in
`../artifacts/surface-contact-20260921/review-followup/`.

## Causal predictive activation (2026-09-21)

New opt-in configuration:
`gm_crash_deformer_predictive_surface_contact_autoregressive_tbptt`.
The endpoint-only configuration and nearest-k backend remain unchanged.
The user selected AGA GB300 for subsequent validation and training; H100-specific
memory optimization is no longer a readiness requirement. This section describes
the method, not an assertion that AGA execution or long training has passed.

At time `t`, use only current coordinates and the physical velocity derived from
the current and previous observed/predicted frame. The horizon `T` is exactly
the existing integration step (5 ms in GM). Future ground truth is never read.

1. Forecast each node by `x(T) = x(0) + T*v`. Enumerate conservative swept
   node/facet AABBs, including the same thickness and 5 mm padding. Subtract a
   common constant velocity for broad-phase construction to remove rigid car
   translation without changing relative trajectories. Incidence, batch and
   supplied exclusions are unchanged. The new broad-pair budget is 8M and must
   still be validated against full-case AGA execution; overflow raises.
2. Find the closest facet point at both current and forecast-end geometry.
   These give two original-vertex barycentric vectors `b0` and `bT`. Current and
   forecast-end quads both pass the nonoverlapping center-fan quality guard.
3. Each vector identifies a material point with linear trajectory. For each
   anchor, set `r = sum(b*x_j)-x_i`, `u = sum(b*v_j)-v_i`, and
   `s = clamp(-dot(r,u)/dot(u,u), 0, T)` (zero speed selects zero). Its predicted
   clearance is `norm(r+s*u) - (thickness_i + sum(b*thickness_j))/2`.
   Take the minimum of both anchor clearances and the current clearance for
   membership and smooth cutoff. Analytic minimization can detect a crossing
   between the two time samples, not just at a sampled endpoint.
4. The 12 message features, source latent interpolation, and reported physical
   gap still describe CURRENT geometry. A positive current gap is not relabeled
   as physical penetration. Selected forecast projections, barycentrics, velocities
   and cutoff retain gradients; detached broad/narrow membership is still discrete.
   No trainable parameter, backbone, optimizer, integration step, or TBPTT window
   is changed. Setting the prediction horizon to zero retains endpoint behavior.

The second anchor matters: forecasting only the current closest material point
misses sliding along a facet edge. On the recorded Run100 pair, the causal
5–10 ms velocity gives a 5.142 mm minimum for that single anchor (still outside
the 5 mm band). With the forecast-end anchor the pair is activated at 10 ms,
weight approximately 0.0987, while its current clearance remains +5.918 mm.
This uses no 15 ms ground truth and does not predict actual shell overlap.

This is a two-anchor constant-velocity approximation, NOT full moving-facet CCD,
an accurate acceleration forecast, a force law, or a nonpenetration guarantee.
A closest point attained only at an intermediate feature, rapid acceleration,
edge–edge contact, or later evolving geometry can still be missed. Changing
interpolated thickness can also mean another material point has lower shell
clearance than the two nearest midsurface anchors. Forecast quality guards may
reject severe extrapolated distortion; such failures remain explicit and require
review, never silently skipping facets. Empirical rollout accuracy remains to
be measured separately from these geometry/regression checks.
