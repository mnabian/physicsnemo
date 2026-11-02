# Shader templates of the car-crash OVRTX viewer

The viewer's runtime expects a handful of MDL shader files next to the data.
They belong to the viewer release, not to this repository, so this directory
ships without them. Copy the files below from the release into this directory
(git ignores them here) or into any directory you pass to the exporter with
`--templates-dir` (or the `OVRTX_TEMPLATES_DIR` environment variable;
`inference.ovrtx_templates_dir` for the inference hook). `export_bundle`
copies whatever it finds and lists the rest under `missing_templates` in
`runtime/deformer/deformer_scene.json`; `validate.py` requires each of the
five MDL files at its destination below and reports a bundle lacking any of
them as incomplete, because the viewer's runtime and container entrypoint
need them (`contour.png` is regenerated, so it is optional). A templates
directory that does not exist at all is an error and nothing is written.

| File | Where the viewer release keeps it | Where the bundle places it |
|---|---|---|
| `cae_materials.mdl` | `car_crash_assets/cae_materials.mdl` of the release data: the scalar material whose `scene::data_lookup_float("<field>", 0.0f)` call the runtime rewrites per field; it must export `material ScalarColor` | `car_crash_assets/cae_materials.mdl` |
| `ovrtx_atlas_scalar.mdl` | `tools/ovrtx_atlas_scalar.mdl` of the viewer repository | `runtime/ovrtx_atlas/ovrtx_atlas_scalar.mdl` |
| `ovrtx_vertex_atlas_scalar.mdl` | `tools/ovrtx_vertex_atlas_scalar.mdl` of the viewer repository | `runtime/ovrtx_vertex_atlas/` and `runtime/ovrtx_vertex_atlas_v/` |
| `ovrtx_original_material_section.mdl` | `runtime/field_materials/` of the release data: the section-cutout wrappers of the original OmniSurface materials | `runtime/field_materials/` |
| `ovrtx_st_scalar.mdl` | `runtime/field_materials/` of the release data: the primvar-fallback scalar material reading the field from `primvars:st` | `runtime/field_materials/` |
| `contour.png` (optional) | `car_crash_assets/colormaps/contour.png` of the release data: the 8-texel blue-to-red "Kit contour" ramp | `car_crash_assets/colormaps/contour.png`; regenerated from `materials.CONTOUR_RGB8` when absent |

The files ship with the viewer release; copy them from the viewer repository
at its `main` branch commit `44993d0d` (the `tools/` files) and from the
release data `assets-2026.09.23` (the `car_crash_assets/` and
`runtime/field_materials/` files), the versions the exporter follows.
