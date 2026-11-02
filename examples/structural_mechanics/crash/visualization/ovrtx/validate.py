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
"""Validate an exported bundle the way the car-crash OVRTX runtime does.

The checks port the start-up validation of the demo's
``src/car_crash_ovrtx_ovstream.py`` (``load_merged_geometry_metadata``,
``load_atlas_metadata``, ``load_vertex_atlas_metadata``,
``original_material_groups``, ``original_material_merged_available``,
``can_use_merged_geometry``, ``prepare_runtime_materials`` and the Docker
entrypoint's file checks) without OVRTX, plus the data round trips of the
demo's ``tools/verify_von_mises_atlas.py`` and
``tools/verify_vertex_field_atlas.py`` (atlas texels decode to the cached field
values within one quantisation step; packed texture coordinates match the
layout). USD checks run when ``pxr`` is importable; otherwise they are
reported as skipped (the other checks need only ``numpy`` and ``pillow``).
The report says what was and was not verified.

CLI::

    python -m visualization.ovrtx.validate <bundle> [--frames 0,12,24] [--samples 65536]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import atlas as atlas_mod
from . import materials
from .scene import FACE_SLOT, ORIGINAL_MATERIALS_FIELD, U_SLOT, V_SLOT

START_FRAME, END_FRAME = 0, 24
DIRECT_SCALAR_FIELDS = {U_SLOT, V_SLOT}
VERTEX_ATLAS_RUNTIME_DIRS = {
    U_SLOT: "ovrtx_vertex_atlas",
    V_SLOT: "ovrtx_vertex_atlas_v",
}
VERTEX_ATLAS_METADATA_FILES = {
    U_SLOT: "u_vertex_atlas.json",
    V_SLOT: "v_vertex_atlas.json",
}
ORIGINAL_MATERIAL_COLOR_INPUTS = frozenset(
    {"diffuse_reflection_color", "specular_reflection_color"}
)
ORIGINAL_MATERIAL_FLOAT_INPUTS = frozenset(
    {
        "diffuse_reflection_weight",
        "metalness",
        "specular_reflection_weight",
        "specular_reflection_roughness",
        "specular_reflection_ior",
        "specular_transmission_weight",
        "specular_transmission_dispersion_abbe",
        "specular_transmission_scattering_depth",
    }
)
ORIGINAL_MATERIAL_BOOL_INPUTS = frozenset({"enable_specular_transmission"})
ORIGINAL_MATERIAL_SUPPORTED_INPUTS = (
    ORIGINAL_MATERIAL_COLOR_INPUTS
    | ORIGINAL_MATERIAL_FLOAT_INPUTS
    | ORIGINAL_MATERIAL_BOOL_INPUTS
)


class ValidationError(RuntimeError):
    pass


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ----- ports of the runtime's loaders ---------------------------------------------


def load_atlas_metadata(runtime_dir: Path) -> dict:
    atlas_dir = runtime_dir / "ovrtx_atlas"
    metadata = _load_json(atlas_dir / "von_mises_atlas.json")
    required = {
        "field",
        "frames",
        "layout",
        "tile_side",
        "tile_grid",
        "atlas_file",
        "schema_file",
        "merged_schema_file",
        "encoding_min",
        "encoding_max",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValidationError(f"Atlas metadata is missing: {', '.join(missing)}")
    if metadata["field"] != FACE_SLOT:
        raise ValidationError("Atlas metadata targets an unexpected physics field.")
    if metadata["layout"] != atlas_mod.LAYOUT:
        raise ValidationError(
            "Atlas metadata has an unsupported wireframe coordinate layout."
        )
    frames = tuple(int(f) for f in metadata["frames"])
    if not frames or len(set(frames)) != len(frames):
        raise ValidationError("Atlas metadata has invalid frame indexes.")
    if int(metadata["tile_grid"]) <= 0 or int(metadata["tile_side"]) <= 0:
        raise ValidationError("Atlas metadata has an invalid tile grid.")
    required_files = [
        atlas_dir / str(metadata["atlas_file"]),
        atlas_dir / str(metadata["schema_file"]),
        atlas_dir / str(metadata["merged_schema_file"]),
        atlas_dir / materials.ATLAS_MDL,
    ]
    absent = [str(p) for p in required_files if not p.is_file()]
    if absent:
        raise ValidationError(f"Missing atlas runtime assets: {', '.join(absent)}")
    return metadata


def load_vertex_atlas_metadata(runtime_dir: Path, field_name: str) -> tuple[Path, dict]:
    atlas_dir = runtime_dir / VERTEX_ATLAS_RUNTIME_DIRS[field_name]
    metadata = _load_json(atlas_dir / VERTEX_ATLAS_METADATA_FILES[field_name])
    required = {
        "field",
        "frames",
        "tile_grid",
        "tile_cells",
        "tile_side",
        "atlas_files",
        "schema_file",
        "mdl_file",
        "encoding_min",
        "encoding_max",
        "layout",
        "bank_tile_grid",
        "frames_per_bank",
        "udim_pattern",
        "udim_start",
        "udim_columns",
        "animation_driven",
        "point_count",
        "face_count",
        "face_vertex_count",
        "merged_mesh_path",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValidationError(f"Vertex atlas metadata is missing: {', '.join(missing)}")
    if metadata["field"] != field_name:
        raise ValidationError(
            "Vertex atlas metadata targets an unexpected physics field."
        )
    if metadata["layout"] != atlas_mod.LAYOUT:
        raise ValidationError(
            "Vertex atlas metadata has an unsupported interpolation layout."
        )
    if not metadata["animation_driven"]:
        raise ValidationError(
            "Vertex atlas metadata must use the MDL animation-time selector."
        )
    frames = tuple(int(f) for f in metadata["frames"])
    if not frames or len(set(frames)) != len(frames):
        raise ValidationError("Vertex atlas metadata has invalid frame indexes.")
    if any(
        int(metadata[k]) <= 0
        for k in (
            "tile_grid",
            "tile_cells",
            "tile_side",
            "bank_tile_grid",
            "frames_per_bank",
        )
    ):
        raise ValidationError("Vertex atlas metadata has invalid tile dimensions.")
    if int(metadata["tile_grid"]) != int(metadata["bank_tile_grid"]):
        raise ValidationError(
            "Vertex atlas metadata has inconsistent bank tile dimensions."
        )
    if int(metadata["frames_per_bank"]) != int(metadata["bank_tile_grid"]) ** 2:
        raise ValidationError(
            "Vertex atlas metadata has an invalid bank frame capacity."
        )
    if int(metadata["udim_columns"]) <= 0:
        raise ValidationError("Vertex atlas metadata has an invalid UDIM column count.")
    pattern = str(metadata["udim_pattern"])
    if atlas_mod.UDIM_TOKEN not in pattern:
        raise ValidationError(
            "Vertex atlas metadata must provide a UDIM texture pattern."
        )
    start = int(metadata["udim_start"])
    if start < 1001:
        raise ValidationError("Vertex atlas metadata has an invalid UDIM start tile.")
    files = metadata["atlas_files"]
    if (
        not isinstance(files, list)
        or not files
        or not all(isinstance(f, str) and f for f in files)
    ):
        raise ValidationError("Vertex atlas metadata has invalid atlas file names.")
    expected_banks = math.ceil(len(frames) / int(metadata["frames_per_bank"]))
    if len(files) != expected_banks:
        raise ValidationError(
            "Vertex atlas metadata has an unexpected atlas bank count."
        )
    columns = int(metadata["udim_columns"])
    expected_files = [
        pattern.replace(
            atlas_mod.UDIM_TOKEN,
            f"{start + (i % columns) + (i // columns) * columns:04d}",
        )
        for i in range(expected_banks)
    ]
    if files != expected_files:
        raise ValidationError(
            "Vertex atlas metadata has unexpected UDIM tile file names."
        )
    if float(metadata["encoding_max"]) <= float(metadata["encoding_min"]):
        raise ValidationError("Vertex atlas metadata has an invalid encoding range.")
    if any(
        int(metadata[k]) <= 0
        for k in ("point_count", "face_count", "face_vertex_count")
    ):
        raise ValidationError("Vertex atlas metadata has invalid mesh counts.")
    required_files = [
        *(atlas_dir / f for f in files),
        atlas_dir / str(metadata["schema_file"]),
        atlas_dir / str(metadata["mdl_file"]),
    ]
    absent = [str(p) for p in required_files if not p.is_file()]
    if absent:
        raise ValidationError(
            f"Missing vertex atlas runtime assets: {', '.join(absent)}"
        )
    return atlas_dir, metadata


def original_material_groups(metadata: dict) -> list[dict]:
    groups = metadata.get("original_material_groups")
    schema_file = metadata.get("original_material_schema_file")
    if groups is None and schema_file is None:
        return []
    if (
        not isinstance(groups, list)
        or not groups
        or not isinstance(schema_file, str)
        or not schema_file
    ):
        raise ValidationError(
            "Merged geometry metadata has incomplete original-material data."
        )
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValidationError(
                "Merged geometry metadata has an invalid original-material group."
            )
        required = {
            "source_material_path",
            "section_material_name",
            "parameters",
            "face_count",
            "mesh_count",
        }
        if not required.issubset(group):
            raise ValidationError(
                "Merged geometry metadata has an incomplete original-material group."
            )
        if (
            group["section_material_name"]
            != f"{materials.SECTION_MATERIAL_PREFIX}_{index:02d}"
        ):
            raise ValidationError(
                "Merged geometry metadata has an unexpected original-material name."
            )
        if not isinstance(group["source_material_path"], str) or not group[
            "source_material_path"
        ].startswith("/"):
            raise ValidationError(
                "Merged geometry metadata has an invalid original-material source path."
            )
        if int(group["face_count"]) < 0 or int(group["mesh_count"]) <= 0:
            raise ValidationError(
                "Merged geometry metadata has invalid original-material counts."
            )
        parameters = group["parameters"]
        if not isinstance(parameters, dict):
            raise ValidationError(
                "Merged geometry metadata has invalid original-material parameters."
            )
        unsupported = set(parameters).difference(ORIGINAL_MATERIAL_SUPPORTED_INPUTS)
        if unsupported:
            raise ValidationError(
                f"Unsupported original-material parameters: {', '.join(sorted(unsupported))}"
            )
        for name, value in parameters.items():
            if name in ORIGINAL_MATERIAL_COLOR_INPUTS:
                if (
                    not isinstance(value, list)
                    or len(value) != 3
                    or not all(math.isfinite(float(c)) for c in value)
                ):
                    raise ValidationError(
                        "Merged geometry metadata has an invalid original-material color."
                    )
            elif name in ORIGINAL_MATERIAL_BOOL_INPUTS:
                if not isinstance(value, bool):
                    raise ValidationError(
                        "Merged geometry metadata has an invalid original-material boolean."
                    )
            elif not math.isfinite(float(value)):
                raise ValidationError(
                    "Merged geometry metadata has an invalid original-material value."
                )
    return groups


def load_merged_geometry_metadata(runtime_dir: Path) -> dict:
    merged_dir = runtime_dir / "ovrtx_merged"
    metadata = _load_json(merged_dir / "merged_geometry.json")
    required = {
        "frames",
        "merged_mesh_path",
        "schema_file",
        "positions_file",
        "point_count",
        "face_count",
        "face_vertex_count",
        "atlas_field",
        "passthrough_meshes",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValidationError(
            f"Merged geometry metadata is missing: {', '.join(missing)}"
        )
    if metadata["atlas_field"] != FACE_SLOT:
        raise ValidationError(
            "Merged geometry metadata targets an unexpected physics field."
        )
    frames = tuple(int(f) for f in metadata["frames"])
    if not frames or len(set(frames)) != len(frames):
        raise ValidationError("Merged geometry metadata has invalid frame indexes.")
    if min(frames) < START_FRAME or max(frames) > END_FRAME:
        raise ValidationError(
            "Merged geometry metadata frame indexes are outside the timeline."
        )
    if int(metadata["point_count"]) <= 0 or int(metadata["face_count"]) <= 0:
        raise ValidationError("Merged geometry metadata has invalid mesh counts.")
    required_files = [
        merged_dir / str(metadata["schema_file"]),
        merged_dir / str(metadata["positions_file"]),
    ]
    groups = original_material_groups(metadata)
    if groups:
        required_files.append(
            merged_dir / str(metadata["original_material_schema_file"])
        )
        if sum(int(g["face_count"]) for g in groups) != int(metadata["face_count"]):
            raise ValidationError(
                "Merged original-material groups do not cover the merged mesh faces."
            )
    passthrough = metadata["passthrough_meshes"]
    if not isinstance(passthrough, list):
        raise ValidationError(
            "Merged geometry metadata has invalid passthrough meshes."
        )
    for entry in passthrough:
        if not isinstance(entry, dict) or not {
            "path",
            "point_count",
            "positions_file",
        }.issubset(entry):
            raise ValidationError(
                "Merged geometry metadata has an invalid passthrough mesh."
            )
        required_files.append(merged_dir / str(entry["positions_file"]))
    absent = [str(p) for p in required_files if not p.is_file()]
    if absent:
        raise ValidationError(
            f"Missing merged geometry runtime assets: {', '.join(absent)}"
        )
    positions = np.load(merged_dir / str(metadata["positions_file"]), mmap_mode="r")
    if positions.dtype != np.float32 or positions.shape != (
        len(frames),
        int(metadata["point_count"]),
        3,
    ):
        raise ValidationError(
            "Merged geometry positions have an unexpected dtype or shape."
        )
    del positions
    vertex_fields = metadata.get("vertex_fields", {})
    if not isinstance(vertex_fields, dict):
        raise ValidationError("Merged geometry metadata has invalid vertex fields.")
    for field_name, entry in vertex_fields.items():
        if field_name not in DIRECT_SCALAR_FIELDS:
            raise ValidationError(
                f"Merged geometry metadata has an unsupported vertex field: {field_name}"
            )
        if not isinstance(entry, dict) or not {
            "values_file",
            "interpolation",
            "point_count",
        }.issubset(entry):
            raise ValidationError(
                f"Merged geometry metadata has an invalid vertex field: {field_name}"
            )
        if entry["interpolation"] != "vertex":
            raise ValidationError(
                f"Merged geometry vertex field must use vertex interpolation: {field_name}"
            )
        if int(entry["point_count"]) != int(metadata["point_count"]):
            raise ValidationError(
                f"Merged geometry vertex field count does not match positions: {field_name}"
            )
        values_path = merged_dir / str(entry["values_file"])
        if not values_path.is_file():
            raise ValidationError(
                f"Missing merged geometry vertex field: {values_path}"
            )
        values = np.load(values_path, mmap_mode="r")
        if values.dtype != np.float32 or values.shape != (
            len(frames),
            int(metadata["point_count"]),
        ):
            raise ValidationError(
                f"Merged geometry vertex field has an unexpected dtype or shape: {field_name}"
            )
        del values
    schema = metadata.get("vertex_schema_file")
    if schema is not None and (
        not isinstance(schema, str) or not schema or not (merged_dir / schema).is_file()
    ):
        raise ValidationError(
            "Merged geometry metadata has an invalid vertex schema file."
        )
    indices_file = metadata.get("face_vertex_indices_file")
    if indices_file is not None:
        path = merged_dir / str(indices_file)
        if not path.is_file():
            raise ValidationError(
                f"Missing merged geometry face-vertex indexes: {path}"
            )
        indices = np.load(path, mmap_mode="r")
        if indices.dtype != np.int32 or indices.shape != (
            int(metadata["face_vertex_count"]),
        ):
            raise ValidationError(
                "Merged geometry face-vertex indexes have an unexpected dtype or shape."
            )
        del indices
    return metadata


def can_use_merged_geometry(merged_meta: dict, vertex_meta: dict) -> bool:
    return (
        tuple(int(f) for f in vertex_meta["frames"])
        == tuple(int(f) for f in merged_meta["frames"])
        and str(vertex_meta["merged_mesh_path"]) == str(merged_meta["merged_mesh_path"])
        and int(vertex_meta["point_count"]) == int(merged_meta["point_count"])
        and int(vertex_meta["face_count"]) == int(merged_meta["face_count"])
        and int(vertex_meta["face_vertex_count"])
        == int(merged_meta["face_vertex_count"])
    )


# ----- data round trips ------------------------------------------------------------


def _sample_faces(face_count: int, samples: int) -> np.ndarray:
    if samples <= 0 or samples >= face_count:
        return np.arange(face_count, dtype=np.int64)
    return np.unique(np.linspace(0, face_count - 1, samples, dtype=np.int64))


def check_face_atlas_round_trip(
    runtime_dir: Path, metadata: dict, frames: list[int]
) -> dict:
    """Decode the per-face atlas and compare with the sidecar face values."""
    values_path = runtime_dir / "deformer" / "face_slot_values.npy"
    if not values_path.is_file():
        return {"skipped": "runtime/deformer/face_slot_values.npy not present"}
    values = np.load(values_path, mmap_mode="r")
    layout = atlas_mod.FaceAtlasLayout(
        int(metadata["face_count"]),
        len(metadata["frames"]),
        int(metadata["tile_side"]),
        int(metadata["tile_grid"]),
    )
    atlas = atlas_mod.read_png16(
        runtime_dir / "ovrtx_atlas" / str(metadata["atlas_file"])
    )
    if atlas.shape != (layout.atlas_side, layout.atlas_side):
        raise ValidationError(
            f"face atlas is {atlas.shape}, expected {(layout.atlas_side,) * 2}"
        )
    quantum = (
        float(metadata["encoding_max"]) - float(metadata["encoding_min"])
    ) / 65535.0
    report, worst = {}, 0.0
    for frame in frames:
        index = list(int(f) for f in metadata["frames"]).index(frame)
        decoded = atlas_mod.decode_face_atlas(
            atlas,
            layout,
            index,
            float(metadata["encoding_min"]),
            float(metadata["encoding_max"]),
        )
        source = np.asarray(values[index], dtype=np.float64)
        finite = np.isfinite(source)
        error = np.abs(decoded[finite] - source[finite])
        worst = max(worst, float(error.max()) if error.size else 0.0)
        report[str(frame)] = {
            "max_abs_error": float(error.max()) if error.size else 0.0,
            "mean_abs_error": float(error.mean()) if error.size else 0.0,
        }
    return {
        "frames": report,
        "max_abs_error": worst,
        "quantization_bound": quantum / 2.0,
        "within_quantization_bound": worst <= quantum / 2.0 + 1e-6,
    }


def check_vertex_atlas_round_trip(
    runtime_dir: Path,
    merged_meta: dict,
    atlas_dir: Path,
    metadata: dict,
    frames: list[int],
    samples: int,
) -> dict:
    """Decode sampled faces of the UDIM atlas and compare with the vertex cache."""
    field = metadata["field"]
    values = np.load(
        runtime_dir
        / "ovrtx_merged"
        / str(merged_meta["vertex_fields"][field]["values_file"]),
        mmap_mode="r",
    )
    face_indices = np.load(
        runtime_dir / "ovrtx_merged" / str(merged_meta["face_vertex_indices_file"]),
        mmap_mode="r",
    )
    face_counts = _face_counts_from_schema_or_sidecar(runtime_dir, merged_meta)
    corners = atlas_mod.face_corner_indices(face_counts, face_indices)
    layout = atlas_mod.VertexAtlasLayout(
        int(metadata["face_count"]),
        len(metadata["frames"]),
        int(metadata["tile_cells"]),
        int(metadata["bank_tile_grid"]),
    )
    face_ids = _sample_faces(layout.face_count, samples)
    cached_bank, cached = None, None
    worst, report = 0.0, {}
    fill = float(metadata.get("missing_value_fill", 0.0))
    for frame in frames:
        index = list(int(f) for f in metadata["frames"]).index(frame)
        bank, _, _ = layout.locate(index)
        if bank != cached_bank:
            cached = atlas_mod.read_png16(atlas_dir / metadata["atlas_files"][bank])
            cached_bank = bank
            if cached.shape != (layout.bank_side, layout.bank_side):
                raise ValidationError(
                    f"vertex atlas page {bank} is {cached.shape}, expected {(layout.bank_side,) * 2}"
                )
        decoded = atlas_mod.decode_vertex_atlas(
            cached,
            layout,
            index,
            float(metadata["encoding_min"]),
            float(metadata["encoding_max"]),
            face_ids,
        )
        # Texels are formed in float32 (the triangles' v1 + v2 - v0 included), exactly as
        # the encoder and the demo's verifier do; comparing against float64 values would
        # add float32 rounding of the extrapolation on top of the half-quantum bound.
        source = np.asarray(values[index], dtype=np.float32)
        expected = atlas_mod.vertex_corner_values(
            source, tuple(c[face_ids] for c in corners)
        )
        expected = np.where(np.isfinite(expected), expected, fill).astype(np.float64)
        error = np.abs(decoded - expected)
        worst = max(worst, float(error.max()))
        report[str(frame)] = {
            "udim_file": metadata["atlas_files"][bank],
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
        }
    bound = float(metadata["max_quantization_error"])
    return {
        "frames": report,
        "samples_per_frame": int(len(face_ids)),
        "max_abs_error": worst,
        "quantization_bound": bound,
        "within_quantization_bound": worst <= bound + 1e-6,
    }


def _face_counts_from_schema_or_sidecar(
    runtime_dir: Path, merged_meta: dict
) -> np.ndarray:
    """Face vertex counts of the merged mesh, in merged order.

    The runtime's caches do not store them (``merged_face_vertex_indices.npy``
    is flat, and a mix of triangles and quads cannot be recovered from its
    length alone), so the exporter writes them to
    ``runtime/deformer/face_counts.npy``. Without that sidecar they are read
    from the merged USD schema, which needs ``pxr``.
    """
    sidecar = runtime_dir / "deformer" / "face_counts.npy"
    if sidecar.is_file():
        counts = np.asarray(np.load(sidecar), dtype=np.int64)
        if counts.ndim != 1 or len(counts) != int(merged_meta["face_count"]):
            raise ValidationError(
                f"{sidecar} holds {counts.shape} face counts, expected ({merged_meta['face_count']},)"
            )
        return counts
    try:
        from .usd_writer import read_merged_st

        counts, _, _ = read_merged_st(
            runtime_dir / "ovrtx_merged" / str(merged_meta["schema_file"]),
            str(merged_meta["merged_mesh_path"]),
        )
        return counts
    except ImportError as error:
        raise ValidationError(
            "merged face counts need runtime/deformer/face_counts.npy (written by "
            f"bundle.py) or pxr to read the merged schema: {error}"
        ) from error


# ----- USD checks (need pxr) -------------------------------------------------------


def check_usd(
    bundle: Path, merged_meta: dict, face_meta: dict, vertex_metas: dict, manifest: dict
) -> dict:
    try:
        from pxr import Usd, UsdGeom, UsdShade
    except ImportError:
        return {"skipped": "pxr (usd-core) not importable"}
    from .usd_writer import read_merged_st

    report = {}
    runtime = bundle / "runtime"
    counts, st, interp = read_merged_st(
        runtime / "ovrtx_merged" / str(merged_meta["schema_file"]),
        str(merged_meta["merged_mesh_path"]),
    )
    if len(counts) != int(merged_meta["face_count"]) or int(counts.sum()) != int(
        merged_meta["face_vertex_count"]
    ):
        raise ValidationError(
            "merged schema face topology does not match merged_geometry.json"
        )
    if interp != "faceVarying" or st.shape != (int(counts.sum()), 3):
        raise ValidationError(
            "merged schema primvars:st must be face-varying with one value per corner"
        )
    expected_cells = atlas_mod.face_cell_coordinates(
        counts, 0, int(merged_meta["atlas_tile_side"])
    )
    report["merged_scene_st_max_error"] = float(np.abs(st - expected_cells).max())
    # packed merged schema of the face atlas
    pcounts, pst, pinterp = read_merged_st(
        runtime / "ovrtx_atlas" / str(face_meta["merged_schema_file"]),
        str(merged_meta["merged_mesh_path"]),
    )
    expected_packed = atlas_mod.packed_from_cell_coordinates(
        pcounts, expected_cells, int(face_meta["tile_side"])
    )
    report["face_atlas_merged_schema_st_max_error"] = float(
        np.abs(pst - expected_packed).max()
    )
    # vertex atlas schemas
    for field, (atlas_dir, meta) in vertex_metas.items():
        vcounts, vst, vinterp = read_merged_st(
            atlas_dir / str(meta["schema_file"]), str(meta["merged_mesh_path"])
        )
        expected = atlas_mod.vertex_texture_coordinates(
            vcounts, int(meta["tile_cells"])
        )
        report[f"{field}_schema_st_max_error"] = float(np.abs(vst - expected).max())
        if vinterp != "faceVarying":
            raise ValidationError(f"{field} schema primvars:st is not face-varying")
    # source stage: meshes of the manifest exist with the variant set and matching topology
    stage = Usd.Stage.Open(
        str(bundle / "car_crash_assets" / "main_project.usd"), load=Usd.Stage.LoadAll
    )
    if stage is None:
        raise ValidationError("unable to open car_crash_assets/main_project.usd")
    report["stage"] = {
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "time_codes": [
            stage.GetStartTimeCode(),
            stage.GetEndTimeCode(),
            stage.GetTimeCodesPerSecond(),
        ],
    }
    d3plot = stage.GetPrimAtPath("/World/d3plot")
    if not d3plot:
        raise ValidationError("main_project.usd has no /World/d3plot")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode(0.0))
    checked = 0
    bound = 0
    for entry in manifest["meshes"]:
        prim = stage.GetPrimAtPath(entry["path"])
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise ValidationError(f"missing source Mesh {entry['path']}")
        mesh = UsdGeom.Mesh(prim)
        face_counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode(0.0))
        indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode(0.0))
        points = mesh.GetPointsAttr().Get(Usd.TimeCode(0.0))
        if len(face_counts) != int(entry["face_count"]) or len(indices) != int(
            entry["face_vertex_count"]
        ):
            raise ValidationError(
                f"topology of {entry['path']} does not match the manifest"
            )
        if points is None or len(points) <= 0:
            raise ValidationError(f"{entry['path']} has no points")
        variants = prim.GetVariantSet("PhysicsField")
        for name in (ORIGINAL_MATERIALS_FIELD, FACE_SLOT, U_SLOT, V_SLOT):
            if name not in variants.GetVariantNames():
                raise ValidationError(
                    f"{entry['path']} lacks PhysicsField variant {name}"
                )
        relative = xform_cache.GetLocalToWorldTransform(
            d3plot
        ).GetInverse() * xform_cache.GetLocalToWorldTransform(prim)
        if not np.allclose(
            np.asarray(relative, dtype=np.float64), np.identity(4), atol=1e-8
        ):
            raise ValidationError(
                f"{entry['path']} is transformed relative to /World/d3plot"
            )
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material:
            bound += 1
        checked += 1
    report["source_meshes_checked"] = checked
    report["source_meshes_with_material"] = bound
    # vertex-field primvars under the variants (what the demo's cache generator reads)
    first = stage.GetPrimAtPath(manifest["meshes"][0]["path"])
    for slot, interpolation in (
        (FACE_SLOT, "uniform"),
        (U_SLOT, "vertex"),
        (V_SLOT, "vertex"),
    ):
        first.GetVariantSet("PhysicsField").SetVariantSelection(slot)
        primvar = UsdGeom.Primvar(first.GetAttribute(f"primvars:{slot}"))
        if not primvar or primvar.GetInterpolation() != interpolation:
            raise ValidationError(
                f"primvars:{slot} missing or not {interpolation}-interpolated"
            )
        values = primvar.Get(Usd.TimeCode(0.0))
        expected = (
            manifest["meshes"][0]["face_count"]
            if slot == FACE_SLOT
            else len(first.GetAttribute("points").Get(Usd.TimeCode(0.0)))
        )
        if values is None or len(values) != expected:
            raise ValidationError(
                f"primvars:{slot} has {None if values is None else len(values)} values, expected {expected}"
            )
    first.GetVariantSet("PhysicsField").SetVariantSelection(ORIGINAL_MATERIALS_FIELD)
    return report


# ----- entry point ----------------------------------------------------------------


def _usd_available() -> bool:
    try:
        import pxr  # noqa: F401
    except ImportError:
        return False
    return True


def validate_bundle(bundle, frames=None, samples: int = 65536) -> dict:
    """The validation report of the bundle at ``bundle``: ``ok``, ``errors``,
    per-check results under ``checks`` and the list of what this validator
    does not cover under ``not_verified``."""
    bundle = Path(bundle)
    assets, runtime = bundle / "car_crash_assets", bundle / "runtime"
    report: dict = {"bundle": str(bundle), "ok": True, "errors": [], "checks": {}}
    not_verified = [
        "rendering in OVRTX / OVSTAGE / OVSTREAM (needs the viewer on a GPU host)",
        "MDL compilation of the shader templates",
    ]
    if not _usd_available():
        not_verified.append(
            "USD stage and schema checks (pxr / usd-core is not importable)"
        )

    def fail(message):
        report["ok"] = False
        report["errors"].append(str(message))

    if not bundle.exists():
        fail(f"bundle directory does not exist: {bundle}")
    elif not bundle.is_dir():
        fail(f"bundle path is not a directory: {bundle}")
    elif not assets.is_dir() and not runtime.is_dir():
        fail(
            f"{bundle} holds no bundle (neither car_crash_assets/ nor runtime/ exists)"
        )
    if not report["ok"]:
        report["not_verified"] = not_verified
        return report

    # entrypoint.sh / prepare_runtime_materials requirements, and every shader
    # template of the viewer release at its destination (templates/README.md)
    entry = {}
    for rel in (
        "car_crash_assets/main_project.usd",
        "car_crash_assets/colormaps/contour.png",
        "runtime/ovrtx_primvars/von_mises_manifest.json",
        *(
            rel
            for name in materials.MDL_TEMPLATES
            for rel in materials.TEMPLATE_DESTINATIONS[name]
        ),
    ):
        entry[rel] = (bundle / rel).is_file()
        if not entry[rel]:
            fail(f"required file missing: {rel}")
    for field in (ORIGINAL_MATERIALS_FIELD, FACE_SLOT, U_SLOT, V_SLOT):
        rel = f"runtime/variant_overlays/{field}.usda"
        entry[rel] = (bundle / rel).is_file()
        if not entry[rel]:
            fail(f"PhysicsField overlay missing: {rel}")
    report["checks"]["required_files"] = entry
    if entry.get("car_crash_assets/cae_materials.mdl"):
        try:
            materials.check_cae_materials(
                (assets / "cae_materials.mdl").read_text(encoding="utf-8")
            )
            report["checks"]["cae_materials"] = "data_lookup pattern present"
        except ValueError as error:
            fail(error)
    merged_meta = face_meta = None
    vertex_metas = {}
    try:
        face_meta = load_atlas_metadata(runtime)
        report["checks"]["atlas_metadata"] = "ok"
    except (ValidationError, OSError, ValueError, KeyError) as error:
        fail(f"atlas metadata: {error}")
    try:
        merged_meta = load_merged_geometry_metadata(runtime)
        report["checks"]["merged_geometry_metadata"] = "ok"
        groups = original_material_groups(merged_meta)
        mdl = runtime / "field_materials" / "ovrtx_original_material_section.mdl"
        report["checks"]["original_material_merged_available"] = (
            bool(groups)
            and mdl.is_file()
            and (
                runtime
                / "ovrtx_merged"
                / str(merged_meta["original_material_schema_file"])
            ).is_file()
        )
    except (ValidationError, OSError, ValueError, KeyError) as error:
        fail(f"merged geometry metadata: {error}")
    for field in (U_SLOT, V_SLOT):
        try:
            vertex_metas[field] = load_vertex_atlas_metadata(runtime, field)
            report["checks"][f"vertex_atlas_metadata_{field}"] = "ok"
            if merged_meta is not None and not can_use_merged_geometry(
                merged_meta, vertex_metas[field][1]
            ):
                fail(
                    f"{field}: vertex atlas and merged cache disagree (can_use_merged_geometry false)"
                )
        except (ValidationError, OSError, ValueError, KeyError) as error:
            fail(f"{field} vertex atlas metadata: {error}")
    if merged_meta is not None:
        available = [int(f) for f in merged_meta["frames"]]
        if frames is None:
            frames = sorted(
                {available[0], available[len(available) // 2], available[-1]}
            )
        else:
            missing = sorted(set(frames) - set(available))
            if missing:
                fail(f"frames {missing} not in the cache")
                frames = [f for f in frames if f in available]
        report["frames_checked"] = frames
        if face_meta is not None:
            try:
                report["checks"]["face_atlas_round_trip"] = check_face_atlas_round_trip(
                    runtime, face_meta, frames
                )
                if (
                    report["checks"]["face_atlas_round_trip"].get(
                        "within_quantization_bound"
                    )
                    is False
                ):
                    fail("face atlas decode exceeds the quantisation bound")
            except (ValidationError, OSError, ValueError) as error:
                fail(f"face atlas round trip: {error}")
        for field, (atlas_dir, meta) in vertex_metas.items():
            try:
                result = check_vertex_atlas_round_trip(
                    runtime, merged_meta, atlas_dir, meta, frames, samples
                )
                report["checks"][f"vertex_atlas_round_trip_{field}"] = result
                if not result["within_quantization_bound"]:
                    fail(f"{field} atlas decode exceeds the quantisation bound")
            except (ValidationError, OSError, ValueError) as error:
                fail(f"{field} vertex atlas round trip: {error}")
        if face_meta is not None and entry.get(
            "runtime/ovrtx_primvars/von_mises_manifest.json"
        ):
            try:
                manifest = _load_json(
                    runtime / "ovrtx_primvars" / "von_mises_manifest.json"
                )
                if sum(int(m["face_count"]) for m in manifest["meshes"]) != int(
                    face_meta["face_count"]
                ):
                    fail(
                        "von_mises_manifest.json face total differs from the atlas metadata"
                    )
                usd_report = check_usd(
                    bundle, merged_meta, face_meta, vertex_metas, manifest
                )
                report["checks"]["usd"] = usd_report
                for key, value in usd_report.items():
                    if key.endswith("_max_error") and value > 1e-6:
                        fail(f"{key} = {value}")
            except (ValidationError, OSError, ValueError, KeyError) as error:
                fail(f"usd: {error}")
    report["not_verified"] = not_verified
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--frames",
        default=None,
        help="comma-separated demo frames to decode (default: first, middle, last)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=65536,
        help="faces sampled per frame for the vertex atlases (0 = all)",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="write the report here as well"
    )
    args = parser.parse_args(argv)
    frames = (
        None
        if args.frames is None
        else [int(v) for v in args.frames.split(",") if v.strip()]
    )
    report = validate_bundle(args.bundle, frames, args.samples)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
