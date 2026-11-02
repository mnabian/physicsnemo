# DeFormer predictions in the car-crash OVRTX viewer

Export tooling that turns a crash surrogate's predicted trajectory, the ground
truth and derived per-node fields (position error, shell penetration depth)
into the data layout of the **car-crash OVRTX demo** (NVIDIA's OVRTX crash
viewer), with the part structure of the mesh preserved. It covers the
full-mesh Yaris pipeline (Zarr stores, time-conditional models) and the body-in-white VTP
pipeline (`inference.py`). The viewer itself and its shader templates are not
part of this repository; `templates/README.md` names the viewer release the
exporter follows and where its files come from.

Contents of this directory:

| File | Purpose |
|---|---|
| `scene.py` | `CrashScene` (positions, faces, parts, fields), `MergedLayout` (the viewer's per-part merged mesh), loaders for the Yaris dumps, VTP frame sequences and raw body-in-white VTPs, a synthetic test scene |
| `fields.py` | derived fields (error magnitude, displacement, face reductions) and `SlotAssignment`: which field goes into which of the viewer's three display slots, with the field's label and suggested colour range |
| `atlas.py` | the viewer's 16-bit per-face and per-vertex UDIM atlases and packed texture coordinates (port of the demo's generators) |
| `usd_writer.py` | the source stage (`main_project.usd`) and the static USD schemas, mirroring the release's prim structure (needs `usd-core`) |
| `materials.py`, `templates/` | OmniSurface part materials, the contour colour map, and the copy of the viewer release's MDL shader templates into a bundle (`templates/README.md` lists the files; the directory ships without them) |
| `bundle.py` | writes a complete bundle (`car_crash_assets/` + `runtime/`); checks its prerequisites before writing anything and removes what it created when writing fails |
| `validate.py` | re-runs the viewer's start-up validation and the demo's atlas round-trip checks on a bundle, without OVRTX (the USD checks need `usd-core`, the others do not) |
| `export_ovrtx.py` | command-line entry point |
| `dump_yaris_predictions.py` | training environment, one GPU: predicted trajectories, ground truth and penetration depths of a Yaris checkpoint, the input of `--yaris-dump` |
| `dump_summary.py` | per-frame vector RMSE and spurious-crossing statistics of a dump case directory, from the same arrays a bundle is built from |
| `render_headless.py` | renders a bundle to PNG frames through the viewer's own server class, without the WebRTC stream (needs the viewer's Python stack and a GPU, section 4.1) |
| `montage.py` | tiles `render_headless.py` frames of several trajectories into a labelled comparison grid with the contour colour bar |
| `../../tests/test_ovrtx_export.py` | synthetic-mesh tests (atlas numerics, layout, slot assignment, VTP and dump input, bundle writing and validation; the USD-dependent ones need `usd-core`) |

## 1. What the demo consumes

The viewer (its `src/car_crash_ovrtx_ovstream.py`) is an OVRTX + OVSTAGE
renderer streamed over WebRTC (OVSTREAM) from a Linux NVIDIA GPU host in
Docker; the browser only sends controls. It mounts two directories read-only:

**`car_crash_assets/`** — the *source stage*. `main_project.usd` is a Y-up
USD stage (24 time codes per second, `metersPerUnit = 0.01` in the release
files) whose crash geometry lives under `/World/d3plot`, an Xform rotated
`-90°` about x. The meshes themselves keep their raw LS-DYNA **z-up
millimetre** coordinates (the release car spans x −5828..50, y ±1054,
z 0..1928 mm, a frame the Yaris stores share: x −4390..230, y ±1185,
z 0..1760 mm). One `/World/d3plot/<part>/Mesh` prim per part holds static
`faceVertexCounts` / `faceVertexIndices` (triangles and quads only),
time-sampled `points` (one sample per frame at integer time codes), a
`PhysicsField` variant set whose variants author the time-sampled
`primvars:<field>` arrays, and an OmniSurface material binding
(`/World/Looks/...`). A `DomeLight` (HDR, rotated 270° about x) lights the
scene. `cae_materials.mdl` is the scalar material whose
`scene::data_lookup_float("<field>", 0.0f)` call the runtime rewrites per
field, and `colormaps/contour.png` is an 8-texel blue-to-red ramp.

**`runtime/`** — pre-baked caches the fast path plays back without touching
the source meshes (fixed topology is required):

| Path | Content |
|---|---|
| `ovrtx_merged/merged_geometry.json` | frames, counts, material groups, file names (`version` 5) |
| `ovrtx_merged/merged_positions.npy` | `float32 [frames, points, 3]`: all visible parts concatenated in part order, nodes shared by two parts duplicated ("merge without welding") |
| `ovrtx_merged/merged_face_vertex_indices.npy` | `int32 [face_vertex_count]` |
| `ovrtx_merged/vertex_fields/<field>.npy` | `float32 [frames, points]` per-vertex field for the direct path |
| `ovrtx_merged/merged_scene.usdc`, `merged_vertex_scene.usdc`, `merged_original_material_scene.usdc` | static merged mesh `/World/d3plot/OVRTXMergedTransient/Mesh` (source meshes deactivated), with face-varying atlas `primvars:st`, vertex `st`, and `materialBind` GeomSubsets per original material |
| `ovrtx_atlas/` | per-face 16-bit PNG atlas (one texel per face, one tile per frame), `von_mises_atlas.json`, packed-coordinate schemas, `ovrtx_atlas_scalar.mdl` |
| `ovrtx_vertex_atlas/`, `ovrtx_vertex_atlas_v/` | per-vertex UDIM atlases (2×2 texels per face, 4 frames per 8192² page for the release mesh), `u_vertex_atlas.json` / `v_vertex_atlas.json`, schema, `ovrtx_vertex_atlas_scalar.mdl` |
| `ovrtx_primvars/von_mises_manifest.json`, `von_mises_schema.usdc` | mesh list (path, face counts) and the primvar-fallback schema |
| `variant_overlays/<field>.usda` | PhysicsField variant selections per field |
| `field_materials/ovrtx_original_material_section.mdl`, `ovrtx_st_scalar.mdl` | runtime material templates (required by the container entrypoint) |
| `deformer/` | the exporter's sidecars: `deformer_scene.json` (scene metadata, slot mapping, labels and ranges, manifest), raw face-slot values, merged face counts and the merged-to-scene index maps; read by `validate.py` and `render_headless.py`, not by the viewer |

Everything is validated at start-up (`load_merged_geometry_metadata`,
`load_atlas_metadata`, `load_vertex_atlas_metadata`, ...): frame indexes must
lie in **0..24**, counts must agree across files, material groups must cover
every face, names and layouts are fixed. Fields are stored as linear 16-bit
values over `[encoding_min, encoding_max]`; the UI's colour range only
changes the MDL mapping.

The display modes are hard-coded in the runtime and the web UI:
`Solid_Color_Default` (original materials, per-material-group colour),
`Von_Mises_Von_Mises_layer_Outer_ip_Centroid` (per-face atlas),
`U_magnitude_layer_NULL_ip_NULL` and `V_magnitude_layer_NULL_ip_NULL`
(per-vertex atlases). Their labels ("Von Mises Stress", "U Displacement
Magnitude", "V Velocity Magnitude") and default colour ranges are not
data-driven.

## 2. Mapping DeFormer output onto it

| Demo concept | Yaris pipeline | body-in-white pipeline |
|---|---|---|
| frames 0..24 at 24 fps | 31 (or fewer) frames at 5 ms: by default frames 0..24 (0–120 ms) are exported; `--frames fit` spreads 25 frames over the whole trajectory; `--frames all` is rejected for trajectories longer than 25 frames | `displacement_t*` frames (0–125 ms at 5 ms, 26 frames) or the `frame_XXX_pred.vtp` sequence of `inference.py` (frame interval `--vtp-dt-ms`, default 5 ms) |
| coordinates | kept nodes (1,480,410 of 1,480,590; outlier mask applied), raw z-up mm | VTP points, raw z-up mm |
| parts | `nodes/part_id` / `shell/part_id` (852 parts); each part prim gets the nodes its faces reference | no part ids in the VTPs: one part, or connected components (`--vtp-parts components`) |
| faces | `shell/connectivity` quads/triangles on kept nodes (degenerate shells and faces touching excluded nodes dropped) plus the exterior faces of `solid/connectivity` hexahedra (`--no-solids` skips them) | the VTP polygons |
| per-face slot (labelled "Von Mises") | `penetration_crossing_face_mm`: max over the face's nodes of the shell-to-shell mid-surface crossing depth (from `penetration_metrics.PenetrationMetrics`, 5 mm band, 10 mm reference clearance) | `error_mm`, max over the face's nodes (the VTP pipeline computes no penetration fields) |
| "U" vertex slot | models: `error_mm` = \|x̂ − x\|; ground truth: `displacement_mm` | `error_mm` when exact frames are given, else `displacement_mm` |
| "V" vertex slot | models: `displacement_mm`; ground truth: `penetration_crossing_mm` per node | `displacement_mm` |
| original materials | 12 OmniSurface colour groups, parts assigned round-robin | same |

Other available fields (Yaris): `penetration_envelope_mm` (thickness-envelope
depth), `wall_crossing_mm` (rigid wall at x = 100 mm),
`truth_penetration_crossing_mm` on model scenes. Any of them can be put in a
slot with `--slot-face`, `--slot-u`, `--slot-v`. The actual mapping, labels
and suggested colour ranges are written to
`runtime/deformer/deformer_scene.json` (and as `deformer_json` in the stage's
`customLayerData`), because the viewer cannot display them. When a slot shows
another field than the label's default (a VTP scene has no penetration
fields, for instance), its label and range follow the field actually shown
(`fields.FIELD_LABELS`, `fields.FIELD_RANGES`; a field not listed there gets
the range of its values).

One bundle holds one trajectory. Comparing ground truth, teacher and corrector
therefore means three bundles of the same case, switched by pointing
`CAR_CRASH_ASSET_DIR` / `CAR_CRASH_RUNTIME_DIR` (`docker/local.env`) at the
bundle and restarting the service.

## 3. Running the export

Requirements: Python with `usd-core` (`pxr`; the viewer pins
`usd-core==24.11`, which has no wheel for Python 3.13, and `usd-core==26.11`
also works with the exporter), `numpy`, `pillow`, and `pyvista` for VTP input.
Without `usd-core` the export refuses to start and writes nothing; the tests'
atlas, layout and slot-assignment checks and the validator's non-USD checks
still run. One way to set up such an environment next to the crash example:

```bash
cd examples/structural_mechanics/crash
uv venv /tmp/ovrtx-venv --python 3.12
uv pip install --python /tmp/ovrtx-venv/bin/python usd-core==24.11 "numpy<2.3" pillow pyvista pytest
```

Shader templates: the viewer's runtime needs five MDL files from the viewer
release next to the data (`cae_materials.mdl`, the two atlas samplers and
the two runtime materials; `templates/README.md` lists them with their
locations in the release). They are not part of this repository. Copy them
into `visualization/ovrtx/templates/` (ignored by git) or pass the directory
that holds them with `--templates-dir` (or `OVRTX_TEMPLATES_DIR`). A
`--templates-dir` that does not exist is an error and nothing is written; a
directory lacking some of the files is not: the export writes everything
else, lists the missing files under `missing_templates` in
`runtime/deformer/deformer_scene.json`, and the validator reports the bundle
as incomplete (every one of the five files is required at its destination).

### 3.1 Yaris: predictions in the training environment

`dump_yaris_predictions.py` runs the model through the same trainer, datapipe
and checkpoint loading as `evaluate_yaris_tc.py` (one GPU; `datapipe.contact_surface=true`
gives the penetration module its shell facets) and writes, per case,
`mesh.npz` (reference geometry, part ids, kept-node ids, thickness, shell
faces, solid hexahedra, target frames), `ground_truth/` and `<label>/` with
`positions.npy` (`float32 [F, N, 3]` mm, frame 0 = reference),
`depth_envelope.npy`, `depth_crossing.npy`, `wall_crossing.npy`
(`float16 [F, N]` mm) and `meta.json` (checkpoint, per-case vector RMSE,
frame times). It is a Hydra script launched with `torchrun` from the crash
example directory, once per trajectory label, with the checkpoint's recipe
and the paths of the Zarr stores, the split manifest, the outlier mask, the
work directory (statistics and neighbourhood caches) and the checkpoint:

```bash
cd examples/structural_mechanics/crash
torchrun --standalone --nproc_per_node=1 visualization/ovrtx/dump_yaris_predictions.py \
    --config-name yaris_deformer \
    yaris.data_root=<zarr stores> yaris.split_manifest=<splits.csv> yaris.outlier_mask=<outlier_nodes.npz> \
    yaris.work_dir=<work dir> training.ckpt_path=<network run>/checkpoints +training.load_epoch=60 \
    datapipe.contact_surface=true training.num_dataloader_workers=0 hydra.run.dir=<out>/hydra_teacher \
    +ovrtx_dump.output_dir=<out> +ovrtx_dump.label=teacher +ovrtx_dump.wall_position_mm=100 \
    '+ovrtx_dump.cases=[<case name>,...]'
```

The corrector run uses `--config-name yaris_deformer_corrector` with
`model.teacher_weights=<network .mdlus file>`, its own `training.ckpt_path`
and `+training.load_epoch`, and `+ovrtx_dump.label=corrector`; every run also
writes `ground_truth/` for its cases. The dump's per-case RMSE must reproduce
the validation record of the checkpoint (the `per_case` entries of the
evaluation suite); `dump_summary.py <case dir>` prints it next to the
crossing statistics.

### 3.2 Bundles

```bash
V=/tmp/ovrtx-venv/bin/python
HDR=<release>/car_crash_assets/ZetoCGcom_ExhibitionHall_Interior1.hdr   # optional dome texture
T=<directory holding the viewer's MDL templates>                         # templates/README.md
for label in ground_truth teacher corrector; do
  $V -m visualization.ovrtx.export_ovrtx --yaris-dump <dumps>/<case> --label $label \
      --out <bundles>/<case>/$label --frames 0-24 --environment-hdr $HDR --templates-dir $T --validate
done
```

the body-in-white data, from an `inference.py` run (or set `inference.ovrtx_export_dir` so
`inference.py` writes the bundle itself after each VTP sequence; the hook
skips with a warning when `usd-core` is not installed):

```bash
$V -m visualization.ovrtx.export_ovrtx --vtp-pred-dir predicted_vtps/rank0/<run> \
    --vtp-exact-dir exact_vtps/rank0/<run> --out <bundles>/<run> --vtp-parts components --validate
$V -m visualization.ovrtx.export_ovrtx --biw-vtp <Run100.vtp> --out <bundles>/Run100_truth --validate
```

Options: `--frames` (`demo`, `fit`, `a-b`, `a,b,c`, `stride:k`; `all` only
for trajectories of at most 25 frames), `--slot-*`, `--vtp-dt-ms` (frame
interval of a VTP sequence, default 5 ms), `--source-frames reference` (a
small source stage with frame 0 only; the runtime's fast path never reads the
source frames, but the demo's own cache generators and its USD fallback do),
`--transform z-up-to-y-up` (not needed: the stage rotates `/World/d3plot`
like the release), `--max-texture-side`, `--material-groups`,
`--single-sided`, `--overwrite`. `--help` describes each.

A bundle records the exporting machine's absolute paths (the sources, the
templates directory, the bundle's own files) and the command line in
`runtime/deformer/deformer_scene.json` and the atlas metadata; nothing reads
them back, but they travel with a bundle that is shared.

Validation: `python -m visualization.ovrtx.validate <bundle>` prints a JSON
report; it fails when any of the runtime's start-up checks would, when a
shader template is missing, when an atlas texel decodes further than one
quantisation step from the cached field, or when a packed texture coordinate
deviates from the layout. Without `usd-core` the USD stage and schema checks
are listed under `not_verified` and the other checks run.

## 4. Loading a bundle in the demo

On the GPU host, from a source checkout of the demo:

```bash
cp docker/local.env.example docker/local.env
# point the service at the bundle (absolute paths work)
echo "CAR_CRASH_ASSET_DIR=/data/bundles/<case>/corrector/car_crash_assets" >> docker/local.env
echo "CAR_CRASH_RUNTIME_DIR=/data/bundles/<case>/corrector/runtime" >> docker/local.env
./start.sh --source           # or ./restart.sh --source after switching bundles
```

Then open `http://<host>:8089/`, connect, and use the display modes with the
colour ranges from `runtime/deformer/deformer_scene.json` (position error
0–30 mm, crossing depth 0–5 mm, displacement 0–1000 mm are the defaults).
The runtime's `field_materials` volume is writable; the two MDL templates it
copies from the bundle are the release's.

## 4.1 Headless rendering without the WebRTC service

`render_headless.py` drives the demo's own `CarCrashServer` class (loaded
from `src/car_crash_ovrtx_ovstream.py`) and writes the `LdrColor` render
variable to PNG, so each image comes from the viewer's code path minus the
stream. Requirements:

* the demo's Python stack at the versions it pins (`ovrtx`, `ovstage`,
  `ovstream`, `warp-lang`; aarch64 wheels exist for them) and an NVIDIA GPU
  with the Vulkan ICD;
* the real-time render mode needs DLSS, i.e. an RTX workstation GPU. On GPUs
  without it set `CAR_CRASH_RENDER_MODE=PathTracing` (the demo's reference
  path tracer): about 90 s of shader compilation on the first run and seconds
  per frame afterwards. `Minimal` renders near-black;
* `libOpenGL.so.0` is only needed by OVSTAGE's `omniverse://` resolver; local
  files render without it.

```bash
export CAR_CRASH_GEOMETRY_BACKEND=merged CAR_CRASH_RENDER_MODE=PathTracing CAR_CRASH_ACTIVE_CUDA_GPUS=0
python visualization/ovrtx/render_headless.py --server-source <demo>/src/car_crash_ovrtx_ovstream.py \
    --bundle <bundle> --out <dir> --frames 0,8,16,24 --warmup 48 [--wireframe] [--section Y:500:1] \
    [--eye 3600 2300 4300 --target -1700 650 0]      # close-up of the Yaris; default = the demo's overview
```

The colour range of each display mode comes from the bundle's
`deformer_scene.json` (`--ranges` overrides it) and is recorded with the
field's label in `render_manifest.json`, which `montage.py` reads for its
captions.

Time handling: the viewer selects the merged geometry explicitly (position
sync per frame), which was correct in every mode. The U/V vertex atlases pick
their tile from `animation_start_frame + animation_time * fps` in MDL, and
OVRTX's `state::animation_time()` accumulates across every `renderer.step`
that passes a non-zero `delta_time`, so the headless renderer steps with
`delta_time = 0` throughout and recreates the renderer per display mode
(known limitation 8).

## 5. Full-scale check

At full scale (one validation case of the Yaris dataset, frames 0-24, the
ground truth, the network and the corrector as three bundles) each bundle has
854 part meshes, about 1.3M merged points and 1.36M faces (quads and
triangles, 114k of them exterior faces of solid elements), a per-face atlas of
5840 texels per side and vertex atlases of three UDIM pages of 7008 texels per
side per field; the export takes about 18 s and validates in a few seconds.
Rendered through the demo's server class in PathTracing mode (section 4.1),
the geometry plays back correctly in every display mode, the crossing-depth
field is zero outside the crush zone, and the models' position error grows
with time and concentrates on the wheel rims and the crush zone.

## 6. What was verified, and what was not

* Unit tests on a synthetic three-part mesh (quads and triangles): atlas
  encode/decode within one quantisation step, packed coordinates against the
  demo's layout rules, merged layout, slot assignment, VTP and dump input,
  bundle writing and validation, source-stage structure.
* The demo's own tools, run on a bundle written by this exporter
  (`tools/generate_merged_geometry_cache.py`, `generate_von_mises_atlas.py`,
  `generate_vertex_field_atlas.py`, `verify_von_mises_atlas.py`,
  `verify_vertex_field_atlas.py`), reproduce the exporter's merged positions,
  face indexes, vertex fields and both atlases bit for bit and pass their
  verification (synthetic mesh and the full-scale bundles of section 5).
* The viewer's start-up validation functions were ported (not merely
  imitated) in `validate.py` and pass on the three exported Yaris bundles; at
  full scale the demo's `generate_merged_geometry_cache.py` reproduced the
  ground-truth bundle's merged positions, face indexes, vertex fields and
  material groups exactly, and both `verify_*` scripts passed on it.
* Rendering through the demo's own `CarCrashServer` (OVRTX + OVSTAGE, the
  release's MDL templates compiled) on a Linux NVIDIA GPU host without DLSS,
  sections 4.1 and 5, in the reference PathTracing mode, not the real-time
  mode, and without the WebRTC stream or the browser UI.
* **Not verified**: the WebRTC service itself (`./start.sh --source`, the
  browser client, field switching and timeline playback from the UI) and the
  real-time render mode; those need an RTX workstation GPU host.

## 7. Known limitations

Properties of the viewer release the exporter targets. The exporter works
around them where it can and records what it cannot change in
`runtime/deformer/deformer_scene.json`.

1. **Field labels and ranges are hard-coded** in the runtime (`FIELDS`,
   `FIELD_DEFAULT_RANGES`) and the web UI (`FIELD_CONFIG`). DeFormer fields
   ride in the "Von Mises" / "U" / "V" slots, so the UI shows those names and
   its own default ranges; the field actually shown, its label and the
   suggested range are in `deformer_scene.json` (`slots`) and the range has
   to be set in the UI by hand.
2. **The timeline is fixed to 25 frames** (`START_FRAME`/`END_FRAME`, the
   slider, `frame_count` in the MDL defaults). Longer trajectories (Yaris: up
   to 31 frames) are subsampled (`--frames fit`) or cut (`--frames demo`,
   `a-b`); `--frames all` is rejected for them.
3. **The camera is fixed** to the release car (`EYE`/`TARGET` in the
   runtime). The Yaris sits in the same z-up mm frame (x −4390..230 vs
   x −5828..50) and renders within the default view; `render_headless.py
   --eye/--target` reframes headless renders, the live viewer cannot be
   reframed from the bundle. `deformer_scene.json` carries the source-frame
   bounds.
4. **One trajectory per service instance.** Comparing ground truth, teacher
   and corrector takes several bundles and a restart per bundle. Passthrough
   meshes (`passthrough_meshes` in `merged_geometry.json`) could carry a
   second trajectory as a ghost but have no field atlas; the exporter does
   not use them.
5. **Generator ordering** in the demo's tools: `generate_merged_geometry_cache.py`
   reads `runtime/ovrtx_atlas/von_mises_atlas.json` while
   `generate_von_mises_atlas.py` reads `runtime/ovrtx_merged/merged_geometry.json`,
   so a fresh runtime cannot be built with them in the documented order
   (merged first). The exporter writes both caches itself; the demo's tools
   then reproduce them from the source stage.
6. **Units**: the release declares `metersPerUnit = 0.01` while the data are
   millimetres (the stage is Y-up with the data rotated under `/World/d3plot`).
   The exporter reproduces the declaration because the viewer's camera, lights
   and section planes are tuned to it (`usd_writer.DEMO_METERS_PER_UNIT`).
7. **Pillow ≥ 12 breaks the demo's atlas generators**: `Image.fromarray(array, mode="I;16")`
   in `generate_von_mises_atlas.py` / `generate_vertex_field_atlas.py` raises
   `ValueError: Invalid mode for data type` (the scripts' `uv` headers do not
   pin Pillow). This exporter omits the `mode` argument (Pillow infers `I;16`
   for `uint16`); the demo's generators need Pillow < 12 or the same change.
8. **`animation_time` is not reset by `load_scene()`** (observed in the
   headless runs). The U/V atlases select their frame tile from it, so a
   renderer that kept stepping with a non-zero `delta_time` in another display
   mode shows U/V tiles offset by the elapsed time until the next
   `go_to_frame`. `render_headless.py` steps with `delta_time = 0` and
   recreates the renderer per display mode; whether the live server drifts
   after a period in a non-time-driven mode was not verified.
9. **Back faces**: the release meshes author `doubleSided = false`; crash
   shells have arbitrary winding, so the exporter authors `true` by default
   (`--single-sided` restores the release's value). Whether the RTX Real-Time
   2.0 path culls back faces was not verified.
