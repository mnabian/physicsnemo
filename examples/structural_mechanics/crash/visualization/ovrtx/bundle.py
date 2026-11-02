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
"""Write a complete car-crash OVRTX viewer bundle from a :class:`CrashScene`.

A bundle is the pair of directories the demo mounts (``car_crash_assets/`` and
``runtime/``) for one trajectory::

    <bundle>/
      car_crash_assets/main_project.usd        source stage (parts, variants, materials, light)
      car_crash_assets/cae_materials.mdl        scalar material template the runtime rewrites
      car_crash_assets/colormaps/contour.png    8-texel contour ramp
      car_crash_assets/<environment>.hdr        optional dome texture
      runtime/variant_overlays/<field>.usda     PhysicsField variant selections
      runtime/ovrtx_primvars/von_mises_manifest.json, von_mises_schema.usdc
      runtime/ovrtx_atlas/                      per-face 16-bit atlas + schemas + MDL
      runtime/ovrtx_merged/                     merged positions/fields caches + merged schemas
      runtime/ovrtx_vertex_atlas/, ovrtx_vertex_atlas_v/   per-vertex UDIM atlases
      runtime/field_materials/                  runtime MDL templates
      runtime/deformer/                         sidecars: scene metadata, slot mapping, raw field values,
                                                merged face counts and source indexes

Everything the demo's ``src/car_crash_ovrtx_ovstream.py`` validates at start-up
(``load_merged_geometry_metadata``, ``load_atlas_metadata``,
``load_vertex_atlas_metadata``, ``original_material_groups``, the entrypoint's
file checks) is written here; :mod:`validate` re-checks it. The MDL shader
files come from the viewer release through ``templates_dir`` (see
:mod:`materials`); a bundle lists the ones it could not copy.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import atlas as atlas_mod
from . import materials, usd_writer
from .fields import SlotAssignment, resolve_slots
from .scene import (
    FACE_SLOT,
    ORIGINAL_MATERIALS_FIELD,
    U_SLOT,
    V_SLOT,
    CrashScene,
    MergedLayout,
)
from .usd_writer import (
    MERGED_MESH_PATH,
    part_mesh_path,
    write_merged_scene,
    write_per_mesh_st_schema,
    write_source_stage,
    write_variant_overlay,
)

MERGED_RUNTIME_DIR = "ovrtx_merged"
ATLAS_RUNTIME_DIR = "ovrtx_atlas"
VERTEX_ATLAS_DIRS = {U_SLOT: "ovrtx_vertex_atlas", V_SLOT: "ovrtx_vertex_atlas_v"}
VERTEX_ATLAS_STEMS = {U_SLOT: "u_vertex_atlas", V_SLOT: "v_vertex_atlas"}
VERTEX_ATLAS_LABELS = {U_SLOT: "U", V_SLOT: "V"}
VERTEX_ATLAS_METADATA = {U_SLOT: "u_vertex_atlas.json", V_SLOT: "v_vertex_atlas.json"}
VERTEX_ATLAS_SCHEMA = {
    U_SLOT: "u_vertex_atlas_schema.usdc",
    V_SLOT: "v_vertex_atlas_schema.usdc",
}
BUNDLE_FORMAT_VERSION = 1


def _json_dump(path, payload) -> None:
    Path(path).write_text(
        json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8"
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _relative_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)


def check_prerequisites(environment_hdr=None, templates_dir=None) -> None:
    """Raise before anything is written: ``ImportError`` without ``usd-core``,
    ``FileNotFoundError`` for a missing environment HDR or a ``templates_dir``
    that is not a directory (a templates directory lacking some files is not an
    error; the bundle lists them under ``missing_templates``)."""
    usd_writer._pxr()
    if environment_hdr and not Path(environment_hdr).is_file():
        raise FileNotFoundError(f"environment HDR not found: {environment_hdr}")
    if templates_dir is not None and not Path(templates_dir).expanduser().is_dir():
        raise FileNotFoundError(f"templates directory not found: {templates_dir}")


def _first_missing_ancestor(path: Path) -> Path | None:
    """The outermost directory ``mkdir(parents=True)`` would create for ``path``."""
    missing = None
    while not path.exists():
        missing = path
        if path.parent == path:
            break
        path = path.parent
    return missing


def export_bundle(
    scene: CrashScene,
    out_dir,
    *,
    assignment: SlotAssignment | None = None,
    max_texture_side: int = atlas_mod.DEFAULT_MAX_TEXTURE_SIDE,
    material_group_count: int = 12,
    environment_hdr: str | None = None,
    source_frames: str = "all",
    overwrite: bool = False,
    double_sided: bool = True,
    extra_metadata: dict | None = None,
    templates_dir=None,
    log=print,
) -> dict:
    """Write the bundle and return its manifest (also saved as
    ``runtime/deformer/deformer_scene.json``). ``templates_dir`` is where the
    viewer release's shader templates are copied from (:mod:`materials`); the
    files it lacks are listed under ``missing_templates``.

    Nothing is written until the prerequisites (:func:`check_prerequisites`)
    and the scene's frame range have been checked; when writing fails, the
    directories this call created are removed so the call can be retried."""
    out = Path(out_dir)
    assets = out / "car_crash_assets"
    runtime = out / "runtime"
    check_prerequisites(environment_hdr, templates_dir)
    if scene.num_frames > 25 or scene.frames != list(range(scene.num_frames)):
        raise ValueError(
            "the viewer's timeline is frames 0..24; select at most 25 consecutive demo "
            "frames (scene.select_frames) before exporting"
        )
    if (assets.exists() or runtime.exists()) and not overwrite:
        raise FileExistsError(f"{out} already holds a bundle; pass overwrite=True")
    created_root = _first_missing_ancestor(out)
    for directory in (assets, runtime):
        if directory.exists():
            shutil.rmtree(directory)
    try:
        return _write_bundle(
            scene,
            out,
            assets,
            runtime,
            assignment=assignment,
            max_texture_side=max_texture_side,
            material_group_count=material_group_count,
            environment_hdr=environment_hdr,
            source_frames=source_frames,
            double_sided=double_sided,
            extra_metadata=extra_metadata,
            templates_dir=templates_dir,
            log=log,
        )
    except BaseException:
        for directory in (assets, runtime):
            shutil.rmtree(directory, ignore_errors=True)
        if created_root is not None:
            shutil.rmtree(created_root, ignore_errors=True)
        raise


def _write_bundle(
    scene: CrashScene,
    out: Path,
    assets: Path,
    runtime: Path,
    *,
    assignment,
    max_texture_side,
    material_group_count,
    environment_hdr,
    source_frames,
    double_sided,
    extra_metadata,
    templates_dir,
    log,
) -> dict:
    started = time.perf_counter()
    assets.mkdir(parents=True)
    runtime.mkdir(parents=True)
    assignment = assignment or SlotAssignment()
    layout = MergedLayout.build(scene)
    slots = resolve_slots(scene, assignment)
    face_values = layout.gather_faces(slots[FACE_SLOT])  # [F, M] merged face order
    vertex_values = {
        slot: layout.gather_nodes(slots[slot]) for slot in (U_SLOT, V_SLOT)
    }  # [F, P]
    frames = list(range(scene.num_frames))
    groups, group_of_part = materials.part_material_groups(
        [p.part_id for p in layout.parts], material_group_count
    )
    log(f"[bundle] {scene.name}: {layout.summary()} frames={len(frames)}")
    missing_templates: list[str] = []

    def copy_template(name, destination):
        copied = materials.copy_template(name, destination, templates_dir)
        if copied is None and name not in missing_templates:
            missing_templates.append(name)
        return copied

    # ----- assets --------------------------------------------------------------
    hdr_name = None
    if environment_hdr:
        hdr_name = Path(environment_hdr).name
        shutil.copyfile(environment_hdr, assets / hdr_name)
    if copy_template(materials.CAE_MATERIALS_MDL, assets / "cae_materials.mdl"):
        materials.check_cae_materials(
            (assets / "cae_materials.mdl").read_text(encoding="utf-8")
        )
    (assets / "colormaps").mkdir()
    if (
        materials.copy_template(
            materials.CONTOUR_PNG, assets / "colormaps" / "contour.png", templates_dir
        )
        is None
    ):
        materials.write_contour_png(assets / "colormaps" / "contour.png")
    scene_meta = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "scene": scene.name,
        "units": scene.units,
        "frames": frames,
        "time_ms": [float(t) for t in scene.time_ms],
        "slots": assignment.as_dict(),
        "scene_meta": scene.meta,
    }
    stage_info = write_source_stage(
        assets / "main_project.usd",
        scene,
        layout,
        {FACE_SLOT: face_values, **vertex_values},
        groups,
        group_of_part,
        environment_hdr=environment_hdr,
        source_frames=source_frames,
        double_sided=double_sided,
        custom_layer_data=scene_meta,
    )
    mesh_paths = stage_info["mesh_paths"]
    log(
        f"[bundle] source stage written ({source_frames} frames) in {time.perf_counter() - started:.0f}s"
    )

    # ----- variant overlays and primvar manifest -------------------------------
    for field_name in (ORIGINAL_MATERIALS_FIELD, FACE_SLOT, U_SLOT, V_SLOT):
        write_variant_overlay(
            runtime / "variant_overlays" / f"{field_name}.usda", mesh_paths, field_name
        )
    primvars_dir = runtime / "ovrtx_primvars"
    primvars_dir.mkdir()
    manifest = {
        "field": FACE_SLOT,
        "mesh_count": len(layout.parts),
        "source_usd": str(assets / "main_project.usd"),
        "meshes": [
            {
                "path": part_mesh_path(part.prim_name),
                "face_count": part.face_count,
                "face_vertex_count": int(layout.part_face_counts(part).sum()),
                "point_count": part.point_count,
            }
            for part in layout.parts
        ],
    }
    _json_dump(primvars_dir / "von_mises_manifest.json", manifest)

    # ----- per-face atlas ------------------------------------------------------
    face_layout = atlas_mod.FaceAtlasLayout.build(layout.face_count, len(frames))
    atlas_dir = runtime / ATLAS_RUNTIME_DIR
    atlas_dir.mkdir()
    atlas_started = time.perf_counter()
    face_atlas, face_min, face_max, face_nonfinite = atlas_mod.encode_face_atlas(
        face_values, face_layout
    )
    atlas_mod.write_png16(atlas_dir / "von_mises_atlas_16bit.png", face_atlas)
    atlas_seconds = time.perf_counter() - atlas_started
    per_mesh_packed = [
        atlas_mod.packed_face_coordinates(
            layout.part_face_counts(part), part.face_offset, face_layout.tile_side
        )
        for part in layout.parts
    ]
    write_per_mesh_st_schema(
        atlas_dir / "von_mises_atlas_schema.usdc", layout, per_mesh_packed, FACE_SLOT
    )
    # The primvar fallback schema: the same per-mesh st declaration (zeros), rewritten by the runtime.
    write_per_mesh_st_schema(
        primvars_dir / "von_mises_schema.usdc",
        layout,
        [np.zeros_like(st) for st in per_mesh_packed],
        FACE_SLOT,
    )
    cell_st = atlas_mod.face_cell_coordinates(
        layout.face_counts, 0, face_layout.tile_side
    )
    merged_packed_st = atlas_mod.packed_from_cell_coordinates(
        layout.face_counts, cell_st, face_layout.tile_side
    )
    merged_positions = layout.gather_positions(scene.positions)
    points0 = merged_positions[0]
    write_merged_scene(
        atlas_dir / "von_mises_merged_schema.usdc",
        layout,
        points0,
        merged_packed_st,
        "faceVarying",
    )
    copy_template(materials.ATLAS_MDL, atlas_dir / materials.ATLAS_MDL)
    _relative_symlink(atlas_dir / "colormaps", assets / "colormaps")
    face_atlas_meta = {
        "version": 4,
        "field": FACE_SLOT,
        "layout": atlas_mod.LAYOUT,
        "frames": frames,
        "frame_count": len(frames),
        "tile_side": face_layout.tile_side,
        "tile_grid": face_layout.tile_grid,
        "atlas_width": face_layout.atlas_side,
        "atlas_height": face_layout.atlas_side,
        "encoding": atlas_mod.ENCODING,
        "encoding_min": face_min,
        "encoding_max": face_max,
        "nonfinite_value_count": face_nonfinite,
        "mesh_count": len(layout.parts),
        "face_count": layout.face_count,
        "face_vertex_count": layout.face_vertex_count,
        "source_usd": str(assets / "main_project.usd"),
        "atlas_file": "von_mises_atlas_16bit.png",
        "schema_file": "von_mises_atlas_schema.usdc",
        "atlas_seconds": atlas_seconds,
        "schema_seconds": time.perf_counter() - atlas_started - atlas_seconds,
        "total_seconds": time.perf_counter() - atlas_started,
        "merged_schema_file": "von_mises_merged_schema.usdc",
    }
    _json_dump(atlas_dir / "von_mises_atlas.json", face_atlas_meta)
    log(
        f"[bundle] face atlas {face_layout.atlas_side}^2 written in {time.perf_counter() - atlas_started:.0f}s"
    )

    # ----- merged geometry cache ------------------------------------------------
    merged_dir = runtime / MERGED_RUNTIME_DIR
    merged_dir.mkdir()
    (merged_dir / "passthrough_positions").mkdir()
    positions_path = merged_dir / "merged_positions.npy"
    cache = np.lib.format.open_memmap(
        positions_path, mode="w+", dtype=np.float32, shape=merged_positions.shape
    )
    cache[:] = merged_positions
    cache.flush()
    del cache
    np.save(
        merged_dir / "merged_face_vertex_indices.npy",
        np.ascontiguousarray(layout.face_vertex_indices, dtype=np.int32),
    )
    vertex_field_dir = merged_dir / "vertex_fields"
    vertex_field_dir.mkdir()
    vertex_field_meta = {}
    for slot in (U_SLOT, V_SLOT):
        values_path = vertex_field_dir / f"{slot}.npy"
        np.save(
            values_path, np.ascontiguousarray(vertex_values[slot], dtype=np.float32)
        )
        vertex_field_meta[slot] = {
            "values_file": f"vertex_fields/{slot}.npy",
            "interpolation": "vertex",
            "point_count": layout.point_count,
        }
    write_merged_scene(
        merged_dir / "merged_scene.usdc", layout, points0, cell_st, "faceVarying"
    )
    write_merged_scene(
        merged_dir / "merged_vertex_scene.usdc",
        layout,
        points0,
        np.zeros((layout.point_count, 3), dtype=np.float32),
        "vertex",
    )
    write_merged_scene(
        merged_dir / "merged_original_material_scene.usdc",
        layout,
        points0,
        cell_st,
        "faceVarying",
        material_groups=groups,
        part_material_group=group_of_part,
    )
    group_meta = []
    for index, group in enumerate(groups):
        parts = [p for p, g in zip(layout.parts, group_of_part) if g == index]
        group_meta.append(
            {
                "source_material_path": f"/World/Looks/{group['name']}",
                "section_material_name": group["section_material_name"],
                "parameters": group["parameters"],
                "face_count": int(sum(p.face_count for p in parts)),
                "mesh_count": len(parts),
            }
        )
    # The runtime requires every group's name to follow its index and a positive mesh count.
    group_meta = [g for g in group_meta if g["mesh_count"] > 0]
    for index, group in enumerate(group_meta):
        group["section_material_name"] = (
            f"{materials.SECTION_MATERIAL_PREFIX}_{index:02d}"
        )
    merged_meta = {
        "version": 5,
        "source_usd": str(assets / "main_project.usd"),
        "frames": frames,
        "merged_mesh_path": MERGED_MESH_PATH,
        "schema_file": "merged_scene.usdc",
        "vertex_schema_file": "merged_vertex_scene.usdc",
        "original_material_schema_file": "merged_original_material_scene.usdc",
        "positions_file": "merged_positions.npy",
        "face_vertex_indices_file": "merged_face_vertex_indices.npy",
        "mesh_count": len(layout.parts),
        "source_mesh_count": len(layout.parts),
        "excluded_invisible_meshes": [],
        "point_count": layout.point_count,
        "face_count": layout.face_count,
        "face_vertex_count": layout.face_vertex_count,
        "atlas_tile_side": face_layout.tile_side,
        "atlas_field": FACE_SLOT,
        "vertex_fields": vertex_field_meta,
        "original_material_groups": group_meta,
        "passthrough_meshes": [],
    }
    _json_dump(merged_dir / "merged_geometry.json", merged_meta)
    log(f"[bundle] merged cache written in {time.perf_counter() - started:.0f}s")

    # ----- per-vertex UDIM atlases --------------------------------------------
    vertex_layout = atlas_mod.VertexAtlasLayout.build(
        layout.face_count, len(frames), max_texture_side
    )
    corners = atlas_mod.face_corner_indices(
        layout.face_counts, layout.face_vertex_indices
    )
    vertex_st = atlas_mod.vertex_texture_coordinates(
        layout.face_counts, vertex_layout.tile_cells
    )
    vertex_atlas_meta = {}
    for slot in (U_SLOT, V_SLOT):
        slot_started = time.perf_counter()
        directory = runtime / VERTEX_ATLAS_DIRS[slot]
        directory.mkdir()
        values = vertex_values[slot]
        enc_min, enc_max, nonfinite = atlas_mod.vertex_atlas_range(values, corners)
        files = []
        for bank in range(vertex_layout.bank_count):
            page = atlas_mod.encode_vertex_atlas_bank(
                values, corners, vertex_layout, bank, enc_min, enc_max
            )
            name = atlas_mod.udim_file_name(VERTEX_ATLAS_STEMS[slot], bank)
            atlas_mod.write_png16(directory / name, page)
            files.append(name)
        bank_seconds = time.perf_counter() - slot_started
        write_merged_scene(
            directory / VERTEX_ATLAS_SCHEMA[slot],
            layout,
            points0,
            vertex_st,
            "faceVarying",
        )
        copy_template(
            materials.VERTEX_ATLAS_MDL, directory / materials.VERTEX_ATLAS_MDL
        )
        _relative_symlink(directory / "colormaps", assets / "colormaps")
        span = enc_max - enc_min
        meta = {
            "version": 5,
            "field": slot,
            "frames": frames,
            "frame_count": len(frames),
            "layout": atlas_mod.LAYOUT,
            "tile_cells": vertex_layout.tile_cells,
            "tile_side": vertex_layout.tile_side,
            "tile_grid": vertex_layout.bank_tile_grid,
            "bank_tile_grid": vertex_layout.bank_tile_grid,
            "frames_per_bank": vertex_layout.frames_per_bank,
            "atlas_bank_count": vertex_layout.bank_count,
            "udim_pattern": atlas_mod.udim_pattern(VERTEX_ATLAS_STEMS[slot]),
            "udim_start": atlas_mod.UDIM_START,
            "udim_columns": atlas_mod.UDIM_COLUMNS,
            "atlas_width": vertex_layout.bank_side,
            "atlas_height": vertex_layout.bank_side,
            "encoding": atlas_mod.ENCODING,
            "encoding_min": enc_min,
            "encoding_max": enc_max,
            "max_quantization_error": span / (2.0 * 65535.0),
            "nonfinite_value_count": nonfinite,
            "missing_value_fill": 0.0,
            "point_count": layout.point_count,
            "face_count": layout.face_count,
            "face_vertex_count": layout.face_vertex_count,
            "triangle_face_count": int((layout.face_counts == 3).sum()),
            "quadrilateral_face_count": int((layout.face_counts == 4).sum()),
            "merged_mesh_path": MERGED_MESH_PATH,
            "source_merged_schema": str(merged_dir / "merged_scene.usdc"),
            "source_positions": str(positions_path),
            "source_vertex_values": str(vertex_field_dir / f"{slot}.npy"),
            "atlas_files": files,
            "schema_file": VERTEX_ATLAS_SCHEMA[slot],
            "mdl_file": materials.VERTEX_ATLAS_MDL,
            "animation_driven": True,
            "atlas_seconds": bank_seconds,
            "schema_seconds": time.perf_counter() - slot_started - bank_seconds,
            "total_seconds": time.perf_counter() - slot_started,
        }
        _json_dump(directory / VERTEX_ATLAS_METADATA[slot], meta)
        vertex_atlas_meta[slot] = meta
        log(
            f"[bundle] {VERTEX_ATLAS_LABELS[slot]} atlas: {len(files)} UDIM page(s) of {vertex_layout.bank_side}^2 in {time.perf_counter() - slot_started:.0f}s"
        )

    # ----- runtime materials ------------------------------------------------------
    field_materials = runtime / "field_materials"
    field_materials.mkdir()
    for name in materials.RUNTIME_MATERIAL_TEMPLATES:
        copy_template(name, field_materials / name)

    # ----- sidecars ----------------------------------------------------------------
    sidecar = runtime / "deformer"
    sidecar.mkdir()
    np.save(
        sidecar / "face_slot_values.npy",
        np.ascontiguousarray(face_values, dtype=np.float32),
    )
    np.save(
        sidecar / "merged_point_source.npy",
        np.ascontiguousarray(layout.merged_source_nodes, dtype=np.int64),
    )
    np.save(
        sidecar / "merged_face_source.npy",
        np.ascontiguousarray(layout.face_order, dtype=np.int64),
    )
    # Face vertex counts in merged order: lets validate.py run the vertex-atlas
    # round trip without reading the USD schema (no pxr needed).
    np.save(
        sidecar / "face_counts.npy",
        np.ascontiguousarray(layout.face_counts, dtype=np.int32),
    )
    bounds = scene.bounds()
    manifest_out = {
        **scene_meta,
        "layout": layout.summary(),
        "parts": [
            {
                "part_id": part.part_id,
                "name": part.name,
                "prim_path": part_mesh_path(part.prim_name),
                "point_count": part.point_count,
                "face_count": part.face_count,
                "material_group": groups[group_of_part[i]]["name"],
            }
            for i, part in enumerate(layout.parts)
        ],
        "bounds_source_frame": {"min": bounds[0].tolist(), "max": bounds[1].tolist()},
        "stage": stage_info,
        "environment_hdr": hdr_name,
        "face_atlas": {
            k: face_atlas_meta[k]
            for k in (
                "tile_side",
                "tile_grid",
                "encoding_min",
                "encoding_max",
                "nonfinite_value_count",
            )
        },
        "vertex_atlases": {
            VERTEX_ATLAS_LABELS[s]: {
                k: m[k]
                for k in (
                    "tile_cells",
                    "bank_tile_grid",
                    "atlas_bank_count",
                    "encoding_min",
                    "encoding_max",
                )
            }
            for s, m in vertex_atlas_meta.items()
        },
        "material_groups": group_meta,
        "templates_dir": str(materials.resolve_templates_dir(templates_dir)),
        "missing_templates": missing_templates,
        "field_statistics": {
            name: {
                "min": float(np.nanmin(v)),
                "max": float(np.nanmax(v)),
                "mean": float(np.nanmean(v)),
            }
            for name, v in {**scene.node_fields, **scene.face_fields}.items()
        },
        "extra": extra_metadata or {},
        "seconds": time.perf_counter() - started,
    }
    _json_dump(sidecar / "deformer_scene.json", manifest_out)
    log(f"[bundle] done in {time.perf_counter() - started:.0f}s -> {out}")
    if missing_templates:
        log(
            f"[bundle] WARNING: shader templates not found in {materials.resolve_templates_dir(templates_dir)}: "
            f"{missing_templates}; the viewer requires them (visualization/ovrtx/templates/README.md)"
        )
    return manifest_out
