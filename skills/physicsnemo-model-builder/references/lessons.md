# Hard-won lessons

Traps distilled from real model integrations. Surface the relevant one *inline*
as you scaffold — they're cheap to avoid up front and expensive to discover in
review or after a checkpoint won't load.

## 1. `Module` serialization breaks on raw `nn.Module` args (the #1 trap)

Injecting a `torch.nn.Module` as a constructor argument passes `forward` but
makes `save()`/`from_checkpoint()` raise `TypeError`. Use constructor-from-config
or convert submodules with `Module.from_torch`. Always prove it with
`validate_checkpoint`. Full detail: `references/serialization.md`.

## 2. The TE-aware `LayerNorm` is CUDA-only when Transformer Engine is present

`physicsnemo/nn/module/layer_norm.py::get_layer_norm_class()` decides **once at
import** whether `LayerNorm` is Transformer Engine's (when TE + CUDA are
available) or `torch.nn.LayerNorm`. TE's LayerNorm **cannot run on CPU tensors**.
Consequences:
- A model using it runs fine on a CPU-only box (falls back to torch) and on
  GPU (uses TE) — but on a TE-enabled GPU box it **cannot** run on CPU.
- Tests must **skip the `device="cpu"` case when TE is the active backend** (the
  repo's own `test/nn/module/test_layer_norm.py` does this). A simple guard:
  detect `issubclass(LayerNorm, torch.nn.LayerNorm)` is False ⇒ TE active ⇒
  skip cpu. `PHYSICSNEMO_FORCE_TE=0` forces the torch path for CPU work.

## 3. `experimental/` skips lint, not runtime contracts

`physicsnemo/experimental/` is excluded from ruff and interrogate (incomplete
docstrings/missing-tests are tolerated), but it is **not** exempt from behavior:
imports must work, `forward` must run, and a `physicsnemo.Module` must still
serialize. Don't let "it's experimental" excuse a broken `from_checkpoint`.

## 4. Promote a layer to `physicsnemo.nn` only on the second consumer

A layer used by exactly one model is not yet a reusable primitive — keep it
local to the model directory. Promote it to `physicsnemo/nn/module/` when a
**second** model actually needs it. Premature generalization freezes an API for
a single user and is its own anti-pattern.

## 5. jaxtyping single-token shape strings trip `F821`

`Float[Tensor, "n dim"]` (multi-token) is fine, but a single-token annotation
like `Int[Tensor, "n"]` makes ruff flag `F821` (undefined name). Add
`# noqa: F821` on that line — it's the established convention. Annotate **all**
tensor args including optional ones (`cond`, `mask`, `batch_ids`) per `MOD-006`.

## 6. Use `test.common`, don't hand-roll test machinery

`test.common` provides `validate_forward_accuracy` (auto-managed reference
`.pth` for non-regression — `MOD-008b`) and `validate_checkpoint` (save/load
round-trip — `MOD-008c`). Hand-rolling committed `.mdlus` golden files +
ad-hoc comparisons is non-idiomatic and brittle; these helpers are the house
pattern (see `test/nn/module/test_mlp_layers.py`).

## 7. For kNN/backend correctness tests, compare distances, not indices

cuML (CUDA) and torch/scipy order equal- and near-equal-distance neighbors
differently, so asserting exact neighbor **index** equality across backends
fails spuriously in GPU CI. Compare the selected neighbors' **distances**
(what `KNN.compare_forward` does). For attention this is doubly safe — the
aggregation is permutation-invariant over neighbors anyway.

## 8. Behavior-preserving refactors must be *proven* behavior-preserving

If you dedup/restructure existing model code, verify zero behavior change:
keep `state_dict` keys stable (so committed checkpoints still load), and run a
bitwise/`assert_close` equivalence check (same seed, same weights, before vs
after). A green test suite alone doesn't prove equivalence.

## 9. Losses and metrics go in `metrics/`, not a `losses/` folder

There is no `losses/` convention. Regression/SSL losses belong in
`physicsnemo/metrics/<domain>/` (or `physicsnemo/experimental/metrics/`),
reusing existing primitives (e.g. `metrics/general/mse.py`) rather than
re-defining them. Out of scope for this skill — redirect.

## 10. The `device` fixture parametrizes cpu *and* cuda

In `test/`, the `device` fixture runs both cpu and cuda when a GPU is present
(it auto-skips cuda when absent). So a layer that can't run on cpu (e.g. under
TE, lesson #2) will fail the cpu parametrization unless guarded. Don't assume
"GPU CI = cuda-only."
