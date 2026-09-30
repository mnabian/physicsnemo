# DeFormer contact v2

Versioned configuration: `gm_crash_deformer_contact_v2_autoregressive_tbptt`.
FLARE (`GALE_FA`), 256-wide latent contact residual, 398,593 contact parameters.
The existing 500-epoch Muon/cosine recipe, 4-transition closed-loop TBPTT,
24-transition validation, and 127/8/15 case splits are unchanged. No early stopping.

## Geometry and material topology

VTP polygon connectivity is retained in an optional tensor-only cache extension.
The contact datapipe requires that connectivity; old caches lacking it are rebuilt,
not silently accepted. Exclude self, symmetric one- and two-hop mesh neighborhoods,
and all vertex pairs in a common element **before** nearest-neighbor selection.
Sparse CPU topology is shared across matching ensemble meshes and copied without
mutating the CPU dataset cache. Connected components are never wholly excluded;
distant regions of a folded part remain eligible for self-contact.

This is a mesh-resolution-dependent local-neighborhood policy. It can exclude
genuine interactions between very nearby material neighborhoods; it is not a
universal geometric contact criterion. A node-to-node shell-clearance proxy cannot
detect every vertex-face or edge-edge intersection on a coarse shell mesh.

## Continuous bounded sparsification

Discover the nearest **K+1** eligible candidates inside the physical radius R=10 mm;
retain at most K=16 messages. Set support s to the live (K+1)th distance, or R if
there is no buffer neighbor. For a retained pair at distance d, use

`w_rank = [max(s²-d², 0) / (s²+epsilon²)]²`, epsilon=0.1 mm.

The message disappearing at a K/K+1 exchange has zero contribution. This removes
the finite response jump of hard nearest-k selection. Buffer coordinates remain
in autograd. The response is continuous and piecewise differentiable, **not
globally C1**: order-statistic changes, ReLU layers, and aggregation introduce
gradient kinks. Exact dense boundary ties have zero weight, including an all-
coincident over-capacity cluster. This conservative degeneracy policy avoids
arbitrary node-ID-dependent forces but cannot resolve an already collapsed mesh.

Direction is `relative / sqrt(sum(relative²)+epsilon²)`. Its Jacobian is bounded
by 1/epsilon=10/mm, including coincidence. This regularized direction is not an
exact unit normal at short distances. Distance and clearance feature channels
retain their physical definitions. Relative speed uses the same live direction.

Let g=d-(thickness_i+thickness_j)/2. A separate proximity activation is
`w_gap = [1-min(max(g,0)/5 mm, 1)²]²`. The final weight is `w_rank*w_gap`.
This 5 mm proxy band is an explicit modeling choice to test, not a measured
contact tolerance or proof of surface contact. Exact zero-weight messages may be
removed, preserving outputs and first-order derivatives; no small-weight cutoff
is used. Weighted aggregation and node activation taper the entire residual,
including MLP biases, to zero when the contact mass vanishes.

## Reproducibility

Contact construction restores the CPU RNG stream before creating subsequent
shared context layers (`contact_isolate_rng=True`). Shared no-contact parameters
and the outer RNG state are bitwise identical under the same initialization seed.
Random training windows use a stateless seed/epoch/sample-slot key, independent
of model size, rank, worker assignment, access order and prior draws. Case shuffle
retains the original distributed sampler seed 0. DataLoader generators are separate.

Each checkpoint captures Python, NumPy, Torch CPU, current-rank CUDA and loader
RNG state on **every rank**, after validation. Restart restores after initialization
and rejects a world-size mismatch or missing v2 RNG state. Epoch-boundary replay
is tested; mid-epoch progress is intentionally replayed from the last checkpoint.
Workers are non-persistent. The original v2 full-car replay failed despite exact
RNG restoration because identical same-process updates were themselves numerically
non-repeatable. The revised recipe also enables strict PyTorch deterministic
algorithms, deterministic cuDNN, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA
initialization. Unsupported nondeterministic kernels fail rather than warn.
The controlled H100 test then reproduced trajectories, gradients, contact graphs
and parameter updates bitwise across three identical updates. It cost about 36%
more steady-state time in that single-case diagnostic. The full eight-GPU replay
gate remains required; cross-hardware/framework reproducibility is not promised.
Resuming a nondeterministic checkpoint into this policy is rejected. Historical
recipes do not opt into this execution policy automatically.

The subsequent eight-GPU diagnostic exposed an additional custom-kernel gap:
the geometry encoder's Warp radius-search backward used floating-point atomic
adds, which are outside PyTorch's deterministic-algorithm enforcement. A repeated
update first diverged in one rank's local backward despite identical inputs,
contact graphs, output and loss. In deterministic mode, selected-point gradients
now use PyTorch's deterministic indexed reduction, with independent batch offsets
and explicit exclusion of unused neighbor slots. The legacy atomic path remains
unchanged when deterministic mode is disabled. This changes summation order, not
the architecture, parameters, objective, selected edges, or derivative formula.
Full-car repeated-update and fresh-process restart gates must still pass for this
revision; small-case tests alone do not authorize the main run.

Existing baseline/no-contact runs use the old RNG recipe. This new run is not a
strict same-initialization/same-window paired comparison against those runs.
Checkpoint weights from v1 remain loadable with v1 settings; do not resume them
under v2 and describe the result as an unchanged experiment.

## Literature and limits

- [Crash Assessment via Mesh-Based Graph Neural Networks and Physics-Aware Attention](https://arxiv.org/html/2605.11784): mesh/global hybrid and gated proximity contact.
- [MeshGraphNets](https://arxiv.org/abs/2010.03409): separate mesh and spatial interaction edges.
- [Incremental Potential Contact](https://ipc-sim.github.io/): separates candidate generation from contact treatment and addresses degeneracies in geometric contact.

The buffered rank envelope, regularized node direction and proxy activation above
are our design choices, not claims of implementing IPC. This is a learned latent
interaction model, not a constrained mechanics solver. It does not enforce
non-penetration, Newton's third law, frictional dissipation, continuous collision
detection, or rotational equivariance of the full neural network. Geometry/velocity
features are translation/Galilean invariant and rotate consistently; that alone
does not confer invariance on the MLP.
