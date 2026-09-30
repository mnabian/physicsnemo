# DeFormer contact integration

This implementation adds **learned point-contact messages** to the corrected
latent-space DeFormer with standard FLARE (`GALE_FA`), not FLARE++.
Existing contact-disabled experiment configurations remain unchanged.

## Architecture and rollout

```text
node inputs -> native GeoFLARE latent projection -> structural pre-MPNN
                                                     |
predicted physical positions -> nearest-k contact -> gated contact delta
                                                     |
              structural latent residual + contact delta
                                                     |
                  complete FLARE backbone -> mesh-only post-MPNNs
                                                     |
                        acceleration -> integrate -> next predicted state
```

The structural and contact residuals have independent gates. Contact is not
multiplied by the structural pre-MPNN gate (which starts at zero). The contact
gate starts at `1e-3`; setting it to zero recovers the contact-disabled forward
result. Both message-passing paths use latent features, without projecting the
pre-MPNN to three position channels. Final refinement retains only the original
finite-element edges.

At every predicted step, the reusable `FunctionalContactGraphBuilder`:

1. Uses our `contact_search` functional to find the nearest **16 eligible nodes
   inside a physical center-to-center radius**. Self-pairs and structural
   neighbors in either direction are excluded **before** selection. Coincident
   but distinct nonstructural nodes remain eligible. Batch graphs never mix.
2. Reuses per-rollout static exclusion CSR, but rebuilds discovery from the
   current predicted positions. Optional `graph.contact_exclusion_edges` extend
   exclusions, for example with precomputed geodesic neighborhoods. Entire
   connected components are not excluded, allowing folded-shell self-contact.
3. Recomputes relative displacement, distance, normal, and thickness-adjusted
   clearance from live PyTorch tensors. The eight features are
   `[dx/r (3), d/r, (d-(t_i+t_j)/2)/r, normal (3)]`.
4. Optionally appends relative velocity and normal relative speed, yielding 12
   features. Physical velocity is `(x_norm_current-x_norm_previous)/dt * pos_std`;
   normalization means must not be added to velocities.
5. Optionally weights contact by `w=(1-clamp(d/r,0,1)^2)^2`. Both messages and the
   full node correction are tapered; MLP biases cannot create residual contact
   when every weight is zero. Node activation is `1-exp(-sum(w))`. The taper
   and its first derivative vanish at the radius boundary. This does **not**
   smooth nearest-k swaps at equal distances.

For bumper beam, a separate analytic stationary-cylinder encoder supplies rigid
obstacle edges with a learned obstacle embedding, including optional velocities
and a signed-gap-based taper. GM full-car configs deliberately disable this
bumper-specific cylinder. Nodal shell thickness is used without requiring any
bumper global parameters; missing GM thickness defaults to zero.

## Differentiability

Discovery and discrete neighbor IDs are intentionally detached. Once selected,
edge geometry, velocities, smooth weights, messages, acceleration, and position
integration retain gradients across the rollout. Activation checkpointing takes
contact features and weights as explicit tensor inputs, preserving this path.
Empty/no-obstacle contact produces zero rather than missing parameter gradients,
so ranks with different contact counts retain matching reduction layouts.

Thus this is compatible with closed-loop BPTT and the recipe's existing random
time-window TBPTT. It is **not differentiable through edge insertion/removal or
nearest-k identity changes**. No future ground-truth trajectory is used to build
evaluation contact graphs.

## Configurations

| Dataset | Geometric contact | Motion-aware smooth contact (recommended experiment) |
| --- | --- | --- |
| GM full car | `gm_crash_deformer_contact_autoregressive_tbptt` | `gm_crash_deformer_contact_kinematic_autoregressive_tbptt` |
| Bumper | `bumper_deformer_contact_autoregressive` | `bumper_deformer_contact_kinematic_autoregressive` |

These inherit the corresponding existing baseline's dataset/split settings,
optimization schedule, precision, and rollout scheme. GM keeps 500 epochs and
four-step training windows; bumper keeps 10,000 epochs and its full closed-loop
rollout. There is no early stopping. Data paths still need supplying as in the
baseline. Start fresh model/optimizer states: contact adds trainable parameters.
Legacy `SparseContactGraphBuilder` remains available for old configurations;
the new configurations explicitly select `contact_graph_backend=nearest_k` and
`contact_search_implementation=warp`.

For memory-constrained full-car runs, the optional
`gm_crash_deformer_contact_checkpointed_autoregressive_tbptt` configuration
adds nested contact activation checkpointing and splits each structural edge
and node update into its own checkpoint segment. The `..._offloaded_...`
variant additionally offloads structural checkpoint inputs to host memory.
These are execution options only: parameter names/shapes, contact radius,
nearest-k, attention, and the four-transition TBPTT objective are unchanged.
`checkpoint_contact` defaults to false for existing configurations. Contact
features and smooth weights are explicit checkpoint inputs; live geometry and
velocity gradients are retained. CPU offload trades transfer time and host RAM
for device memory; full-car CUDA smoke tests are required before a main run.

Example (not submitted automatically):

```bash
python train.py --config-name=gm_crash_deformer_contact_kinematic_autoregressive_tbptt \
  training.raw_data_dir=/path/to/train \
  training.raw_data_dir_validation=/path/to/validation
```

The default radius `10.0` is an **initial setting in dataset length units**, not
a calibrated full-car value or a gap threshold. Validate units, mesh spacing,
neighbor-count distributions, and known contact events before expensive runs.
Velocity scale is `1000.0` dataset length units per second. The Warp path has
bounded nearest-k output storage and no dense pairwise distance matrix; request
`torch` only for small reference tests. Candidate work can still be large in
dense regions. Search topology is reusable only while node ordering, batch,
connectivity, and device remain unchanged.

## Scope and limitations

The thickness clearance is a **point-pair proxy**, not a true signed shell
surface gap. We deliberately do not claim an exact reproduction of an
underspecified thickness filter, or apply a post-top-k filter that can silently
discard eligible neighbors. Direction is the node-to-node vector; coincident
pairs have zero normal. There are no vertex-face/edge-edge narrow-phase tests,
continuous collision detection, friction laws, force balance, or guaranteed
nonpenetration. This is a contact-aware learned dynamics model, not a contact
solver. Accuracy gains require new controlled experiments.

The functional under `physicsnemo/nn/functional/neighbors/contact_search/` was
ported from the Functionals project's tested nearest-k implementation without
altering its source. Model/recipe tests cover exclusion ordering, batches,
coincident points, smooth cutoff, physical units, checkpointed multi-step
gradients, zero-gate parity, fixed structural post-edges, and config parity.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 python -m pytest \
  test/nn/functional/neighbors/test_contact_search.py \
  test/models/meshtransolver \
  examples/structural_mechanics/crash/tests/test_deformer_contact.py \
  examples/structural_mechanics/crash/tests/test_rollout.py \
  examples/structural_mechanics/crash/tests/test_contact_graph.py \
  examples/structural_mechanics/crash/tests/test_meshgeoflare_config_parity.py -q
```

Use a PhysicsNeMo environment with matching PyTorch/torch-scatter, PyG, Warp,
Hydra, pytest, and the crash reader dependencies. CPU tests do not establish
CUDA or distributed-training correctness; see the accompanying validation
report for what was actually exercised.
