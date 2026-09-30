# Mesh-Attention Crash Architecture Reproduction

This document records the implementation contract for reproducing the model family
described in arXiv:2605.11784, *Crash Assessment via Mesh-Based Graph Neural
Networks and Physics-Aware Attention*, in PhysicsNeMo's structural-mechanics crash
recipe.

The paper does not release source code or enough implementation detail for a bitwise
reproduction. Public arXiv source and GitHub repository searches performed on
2026-08-31 found no author implementation. Every inferred choice below is therefore
kept configurable and is called out explicitly.

## Project decisions

- Implement all three reusable hybrid architectures:
  `MeshTransolver`, `MeshGeoTransolver`, and `MeshGeoFLARE`.
- Put reusable processors under `physicsnemo.experimental.models`; keep dataset,
  transient-scheme, and output-shaping logic in the crash recipe.
- Phase 1 uses the recipe's one-shot scheme: one forward pass predicts all 50 future
  frames and all configured output fields.
- Phase 2 uses closed-loop autoregressive acceleration rollout with a generic sparse
  contact graph rebuilt from predicted geometry.
- Train and evaluate on user-provided bumper-beam data, preserving the selected
  train/validation/test splits.
- When changing training hosts, retain the same repository revision,
  configuration, and data split.

## Dataset contract

The current bumper-beam dataset contains:

| Split | Simulations | Notes |
| --- | ---: | --- |
| train | 129 | `run*.vtp` |
| validation | 5 | `run1`, `run10`, `run100`, `run120`, `run130` |
| test | 5 | Byte-identical copies of the validation files |

The project will preserve these splits as requested. Results on `test` must be labeled
as repeated-validation results, not as independent generalization estimates.

Each one-shot sample contains normalized initial coordinates, optional node features,
three global parameters (`velocity_x`, `thickness_scale`, and `rwall_origin_y`), mesh
connectivity, and 50 target frames. With the current bumper configuration, each frame
has five target channels: position `(x, y, z)`, effective plastic strain, and von Mises
stress. The model output is therefore `[N, 50, 5]`, flattened to 250 decoder channels
inside the core model and reshaped by the recipe wrapper.

Global parameters are standardized with statistics computed from the 129 training
simulations and reused unchanged for validation and test. This is necessary for the
bumper data because the selected raw parameters have substantially different scales
(`rwall_origin_y` spans 0--240, while `thickness_scale` spans 0.7--1.3).

## Phase 1 architecture contract

The MeshTransolver and MeshGeoTransolver variants use the following high-level
sequence:

```text
node/edge encoders
  -> pre-MPNN on structural mesh
  -> global processor
  -> post-MPNN on structural mesh
  -> node decoder
```

Paper defaults are retained unless a crash configuration overrides them:

| Model | Pre-MPNN | Global processor | Global blocks | Post-MPNN | Hidden | Tokens |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| MeshTransolver | 1 | PhysicsAttention++ | 6 | 2 | 128 | 128 |
| MeshGeoTransolver | 1 | GALE | 4 | 2 | 128 | 128 |
| MeshGeoFLARE++ | 1 | full GeoFLARE++ (GALE_FPP) | 6 | 2 | 256 | 128 |

Implementation mapping:

- Mesh processing reuses PhysicsNeMo `MeshGraphNetProcessor` building blocks.
- MeshTransolver reuses `TransolverBlock` with `plus=True`.
- MeshGeoTransolver reuses `GlobalContextBuilder` and `GALE_block` with
  `attention_type="GALE"`.
- MeshGeoFLARE uses the complete GeoFLARE++ backbone with
  `attention_type="GALE_FPP"`: the same input projection, 256-channel hidden
  width, six FLARE-family blocks, local ball-query features, context builder,
  and output projection. Each block retains the fixed-query GeoFLARE parameters
  and adds two full-width projections that synthesize input-conditioned routing
  queries. A residual structural-mesh MPNN acts on the native 256-channel node
  embedding after the input projection and before local-feature concatenation;
  it never decodes through the raw functional input dimension. A second residual
  MPNN refines the physical output channels. No fixed-query GeoFLARE layer is
  removed or reduced, so the `GALE_FA` parameter set remains a strict subset of
  MeshGeoFLARE's state. Zero mesh gates recover the exact GeoFLARE function.
  This latent-space topology changes the mesh-branch parameter names and shapes;
  checkpoints from the earlier input-space residual adapter require a fresh run
  and must not be resumed into this model.
- Geometry-aware variants receive normalized coordinates as geometry context and the
  selected run parameters as global context.
- Structural mesh edges are the only edges used by both MPNN stages in Phase 1.
- The pre- and post-MPNN stages have independent parameters and share the encoded
  structural edge representation. This is an inferred choice: the paper does not say
  whether an updated hidden edge state crosses the global processor, and the reusable
  PhysicsNeMo processor returns only its updated node state.
- Variable-size PyG batches are processed without cross-sample global attention. The
  mesh stages operate on the flattened batch; global attention is applied separately
  to each graph and concatenated back in node order.

## Phase 1 recipe variants

The crash recipe provides one-shot Hydra model and experiment configurations for:

- bumper MeshTransolver
- bumper MeshGeoTransolver
- bumper MeshGeoFLARE++

All three use `CrashGraphDataset`, the same targets, normalized global parameters,
MSE loss, and the same split. The GeoFLARE comparison configuration intentionally
makes MeshGeoFLARE's optimization recipe identical to the existing bumper GeoFLARE
baseline: 121 training samples, Muon, float16 mixed precision, cosine schedule,
validation and checkpointing every 10 epochs, 10,000 epochs, and no early stopping.
Relative to fixed-query GeoFLARE, the model additions are FLARE++ query synthesis,
the structural-edge encoder, and the pre/post MPNNs.

## Phase 2 contact and rollout contract

The generic contact block is inserted after pre-MPNN and before global attention:

```text
H_contact = H_pre + alpha * DeltaH_contact
```

The implementation:

1. Build candidate pairs from current predicted coordinates using radius search.
2. Exclude self pairs, structural mesh edges, and optionally same-component pairs.
3. Apply configurable shell/thickness filtering.
4. Retain at most `k` nearest pairs per destination node.
5. Encode relative displacement, distance, and available surface information.
6. Aggregate contact messages into a latent residual.
7. Use a learnable scalar gate initialized to zero.
8. Rebuild candidate pairs after every autoregressive integration step.

The reusable contact graph uses eight edge channels: relative displacement (3),
distance (1), signed shell-surface gap (1), and contact normal (3). The contact
block substitutes a learned obstacle embedding for analytic rigid-surface edges,
aggregates messages by receiving node, bounds the latent correction with `tanh`,
and applies a scalar gate initialized to zero. Analytic cylinder gaps are measured
from the shell surface rather than its midsurface by subtracting half of the local
shell thickness.

The bumper VTP files contain five disconnected structural components but no rigid-wall
nodes, and their exported point-level `thickness` array is identically zero. The
OpenRadioss generation deck resolves those missing inputs: the collider is a cylinder
parallel to the z axis with center `(-170, rwall_origin_y, 0)`, radius 127 mm, and
search distance 200 mm; the two shell properties are 1.8 and 2.2 mm and are multiplied
by `thickness_scale`. The recipe therefore adds analytic cylinder edges and uses a
documented 2.0 mm nominal shell-thickness fallback. If a future reader supplies a
non-zero physical thickness field, that field takes precedence.

The autoregressive recipe keeps the finite-element connectivity and its structural
edge attributes fixed as reference-mesh material data. When contact is enabled, both
contact graph types are rebuilt from the predicted geometry at every step. The model
predicts normalized acceleration plus the configured extra fields (strain and stress
for the bumper), integrates position and velocity with the paper's semi-implicit
Euler update, and feeds only position and velocity back into the next step. The
`velocity_x` metadata is converted from its source numeric unit (mm/ms, numerically
equal to m/s) to mm/s before normalized-coordinate integration.

Paper defaults are `k=32` for MeshTransolver and `k=16` for the geometry-aware
hybrids. Search radius, shell filtering, normal construction, and component filtering
are unspecified by the paper and must be selected using training data only. Candidate
graphs are validated against analytic bumper-to-cylinder gap. The supplied VTP files
contain no FE contact labels, so precision/recall cannot be reported without an
additional contact-truth export; candidate-count and invariant checks are reported
instead and no contact-accuracy claim is made.

The initial node-to-node search radius is a configurable 10 mm engineering starting
point, not a paper-reported hyperparameter. Same-component filtering defaults off so
folding self-contact remains representable; structural mesh edges are always removed
from the contact candidates.

For matched contact ablations, set `model.enable_contact=false` while leaving
`model.use_contact=true`. This retains the same contact-block parameters and random
initialization but supplies an empty contact graph. Setting both options to `false`
removes the contact block entirely and is intended for non-contact deployment rather
than a controlled architecture ablation.

## Required validation gates

1. Constructor, shape, validation-error, forward, backward, AMP, checkpoint, and
   variable-graph-batch tests for each reusable model.
2. Crash-wrapper tests verifying `[N, 250] -> [N, 50, 5]` reshaping and position-only
   residual addition.
3. CPU and single-GPU smoke tests.
4. Single-simulation overfit for all three one-shot variants.
5. Equal-budget comparison against existing GeoTransolver, GeoFLARE, Transolver, and
   MeshGraphNet one-shot baselines.
6. Autoregressive no-contact stability test.
7. Contact graph precision/recall, candidate count, and memory/runtime measurements.
8. Matched-seed contact/no-contact evaluation.

Every remote experiment must record repository SHA, working-tree state, job ID,
compute node, GPU model, library versions, exact command, Hydra configuration, seed,
precision, batch size, checkpoint, and exit status.

## Validation record

The measurements below predate the FLARE++ switch and are retained only as
historical execution checks. They are not a valid fixed-query
GeoFLARE-versus-MeshGeoFLARE++ architecture comparison. The FLARE++ architecture
requires a fresh 10,000-epoch run.

The architecture and rollout revisions through `650aab8` have passed the following
gates on the supplied bumper mesh (`N=13,675`, 50 predicted frames):

| Check | Result |
| --- | --- |
| Core and crash targeted tests, H100 | 42 passed, 6 expected legacy xfails |
| One-shot real-sample forward/backward | all three variants passed |
| Full 50-step contact forward/backward | all three variants passed |
| MeshTransolver contact rollout | 5.45 s, 2.54 GB reserved |
| MeshGeoTransolver contact rollout | 5.65 s, 2.05 GB reserved |
| MeshGeoFLARE contact rollout | 5.71 s, 1.56 GB reserved |
| One-simulation, 1,000-epoch one-shot overfit | `6.86e-4`, `5.89e-4`, and `5.29e-4` final MSE |
| Full-rollout contact learning probes | all three losses decreased; all three gates moved away from zero |
| Matched-seed MeshTransolver contact/no-contact probe | initial `12.474199`/`12.474208`; epoch 10 `5.095238`/`5.097621` |
| Saved-checkpoint inference | 50 predicted and 50 exact VTPs; positions, strain, and stress finite |
| Recipe-level early-stop integration | stopped at epoch 2 of 3 after the configured stale evaluation |

Candidate reconstruction over the ground-truth geometry in all 51 frames of
`train/run2.vtp` produced
5,600--5,794 node-contact edges per frame, at most 14 total contact edges per
receiving node after adding the analytic cylinder, and zero overlap with structural
mesh edges. The paper cap is therefore respected without saturation for this sample
for both `k=16` and `k=32`. Cylinder gaps were also finite and consistent with the
analytic 127 mm radius and 2 mm shell-thickness fallback; the minimum shell-surface
gap was approximately `-1.0 mm`. These are graph-construction checks, not
contact-label precision or surrogate-accuracy results.

Longer full-dataset optimization, a statistically meaningful contact/no-contact
accuracy ablation, equal-budget legacy baselines, and an independent test split remain
experiment-level gates rather than constructor or execution gates.

## Resolved prerequisite repairs

The implementation includes focused repairs for the following pre-existing recipe
issues:

- graph edge statistics weight the first sample incorrectly;
- the bumper GeoFLARE configuration uses the unsupported name `GALE_FE` instead of
  `GALE_FA`;
- autoregressive velocity metadata require an explicit m/s-to-mm/s conversion;
- copied seed frames must not be included in autoregressive evaluation metrics;
- validation and standalone evaluation must use the same numerical precision.
- run-level global parameters must be normalized using training-only statistics.
