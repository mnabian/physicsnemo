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
"""USD authoring for the car-crash OVRTX viewer (needs ``usd-core`` / ``pxr``).

The demo's source stage (``car_crash_assets/main_project.usd``, read from the
2026.09.23 release) is a Y-up stage at 24 time codes per second whose crash
geometry sits under ``/World/d3plot`` (rotated -90 degrees about x, so the
meshes keep their raw z-up LS-DYNA millimetre coordinates) as one
``/World/d3plot/<part>/Mesh`` prim per part: static topology, time-sampled
``points``, a ``PhysicsField`` variant set whose variants carry the
time-sampled ``primvars:<field>`` arrays, an OmniSurface material binding per
mesh (``/World/Looks/...``) and a dome light. This module writes that stage
from a :class:`~scene.CrashScene` plus the static runtime schemas the viewer
loads next to its caches (merged transient mesh, atlas texture-coordinate
schemas, variant overlays).

Units: the stage declares ``metersPerUnit = 0.01`` (:data:`DEMO_METERS_PER_UNIT`)
although the coordinates are millimetres. The release's part files carry the
same declaration and the viewer's fixed camera, lights and section planes are
tuned to it, so the exporter reproduces it rather than the physically correct
``0.001``; ``write_source_stage(meters_per_unit=...)`` overrides it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scene import (
    FACE_SLOT,
    ORIGINAL_MATERIALS_FIELD,
    SCALAR_SLOTS,
    U_SLOT,
    V_SLOT,
    CrashScene,
    MergedLayout,
)

D3PLOT_PATH = "/World/d3plot"
LOOKS_PATH = "/World/Looks"
DOME_LIGHT_PATH = "/World/DomeLight"
MERGED_MESH_PATH = "/World/d3plot/OVRTXMergedTransient/Mesh"
VARIANT_SET = "PhysicsField"
D3PLOT_ROTATE_XYZ = (-90.0, 0.0, 0.0)
DOME_ROTATE_XYZ = (270.0, 0.0, 0.0)
DEMO_FPS = 24.0
DEMO_METERS_PER_UNIT = 0.01  # the release's part files declare centimetres

SLOT_INTERPOLATION = {FACE_SLOT: "uniform", U_SLOT: "vertex", V_SLOT: "vertex"}


def _pxr():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
    except ImportError as error:  # pragma: no cover - exercised without usd-core
        raise ImportError(
            "writing USD needs usd-core (pip install usd-core==24.11, the demo's "
            "pinned version; Python <= 3.12)"
        ) from error
    return Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt


def part_mesh_path(prim_name: str) -> str:
    return f"{D3PLOT_PATH}/{prim_name}/Mesh"


def _vec3f(array):
    from pxr import Vt

    return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(array, dtype=np.float32))


def _ints(array):
    from pxr import Vt

    return Vt.IntArray.FromNumpy(np.ascontiguousarray(array, dtype=np.int32))


def _floats(array):
    from pxr import Vt

    return Vt.FloatArray.FromNumpy(np.ascontiguousarray(array, dtype=np.float32))


def _set_xform(
    prim, translate=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)
):
    Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt = _pxr()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*rotate))
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*scale))


def define_omnisurface_material(stage, path: str, parameters: dict):
    """An ``OmniSurface.mdl`` material with the inputs the demo's merged
    original-material path accepts (colours as 3-tuples, floats, bools)."""
    Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt = _pxr()
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("Shader"))
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniSurface.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
    for name, value in parameters.items():
        if isinstance(value, bool):
            shader.CreateInput(name, Sdf.ValueTypeNames.Bool).Set(value)
        elif isinstance(value, (list, tuple)):
            shader.CreateInput(name, Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*[float(v) for v in value])
            )
        else:
            shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(float(value))
    output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    output.GetAttr().SetMetadata("renderType", "material")
    for terminal in ("surface", "displacement", "volume"):
        material.CreateOutput(
            f"mdl:{terminal}", Sdf.ValueTypeNames.Token
        ).ConnectToSource(output)
    return material


def write_source_stage(
    path,
    scene: CrashScene,
    layout: MergedLayout,
    slot_values: dict[str, np.ndarray],
    material_groups: list[dict],
    part_material_group: list[int],
    *,
    environment_hdr: str | None = None,
    source_frames: str = "all",
    meters_per_unit: float = DEMO_METERS_PER_UNIT,
    double_sided: bool = True,
    custom_layer_data: dict | None = None,
) -> dict:
    """Write ``main_project.usd``.

    ``slot_values`` holds the face slot ``[F, M]`` in merged face order and the
    vertex slots ``[F, P]`` on merged points. ``material_groups`` are
    ``{"name", "parameters"}`` records and ``part_material_group[i]`` the group
    of ``layout.parts[i]``. ``source_frames="all"`` authors every frame as time
    samples (what the demo's own cache generators read); ``"reference"`` writes
    frame 0 only, for a lightweight stage next to a complete runtime cache.
    """
    Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt = _pxr()
    if source_frames not in ("all", "reference"):
        raise ValueError("source_frames must be 'all' or 'reference'")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, float(meters_per_unit))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(max(scene.num_frames - 1, 0))
    stage.SetTimeCodesPerSecond(DEMO_FPS)
    stage.SetFramesPerSecond(DEMO_FPS)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    if custom_layer_data:
        # Python lists become std::vector<VtValue> in a VtDictionary, which the crate
        # writer cannot serialise; keep the metadata as one JSON string instead.
        stage.GetRootLayer().customLayerData = {
            "deformer_json": json.dumps(_plain(custom_layer_data))
        }

    dome = UsdLux.DomeLight.Define(stage, DOME_LIGHT_PATH)
    dome.CreateIntensityAttr(1000.0)
    if environment_hdr:
        dome.CreateTextureFileAttr(Sdf.AssetPath(f"./{Path(environment_hdr).name}"))
        dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    _set_xform(dome.GetPrim(), rotate=DOME_ROTATE_XYZ)

    UsdGeom.Scope.Define(stage, LOOKS_PATH)
    materials = [
        define_omnisurface_material(
            stage, f"{LOOKS_PATH}/{group['name']}", group["parameters"]
        )
        for group in material_groups
    ]
    d3plot = UsdGeom.Xform.Define(stage, D3PLOT_PATH)
    _set_xform(d3plot.GetPrim(), rotate=D3PLOT_ROTATE_XYZ)
    if materials:
        UsdShade.MaterialBindingAPI.Apply(d3plot.GetPrim()).Bind(
            materials[0], bindingStrength=UsdShade.Tokens.weakerThanDescendants
        )

    merged_positions = layout.gather_positions(scene.positions)  # [F, P, 3]
    frames = range(scene.num_frames) if source_frames == "all" else range(1)
    face_values = np.asarray(slot_values[FACE_SLOT], dtype=np.float32)
    vertex_values = {
        slot: np.asarray(slot_values[slot], dtype=np.float32)
        for slot in (U_SLOT, V_SLOT)
    }
    mesh_paths = []
    for index, part in enumerate(layout.parts):
        mesh_path = part_mesh_path(part.prim_name)
        UsdGeom.Xform.Define(stage, f"{D3PLOT_PATH}/{part.prim_name}")
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        prim = mesh.GetPrim()
        counts = layout.part_face_counts(part)
        mesh.CreateFaceVertexCountsAttr(_ints(counts))
        mesh.CreateFaceVertexIndicesAttr(_ints(layout.local_face_indices[index]))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(bool(double_sided))
        mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        points_attr = mesh.CreatePointsAttr()
        extent_attr = mesh.CreateExtentAttr()
        point_slice = slice(part.point_offset, part.point_offset + part.point_count)
        face_slice = slice(part.face_offset, part.face_offset + part.face_count)
        for frame in frames:
            points = merged_positions[frame, point_slice]
            time = (
                Usd.TimeCode(float(frame))
                if source_frames == "all"
                else Usd.TimeCode.Default()
            )
            points_attr.Set(_vec3f(points), time)
            extent_attr.Set(
                Vt.Vec3fArray(
                    [
                        Gf.Vec3f(*points.min(axis=0).tolist()),
                        Gf.Vec3f(*points.max(axis=0).tolist()),
                    ]
                ),
                time,
            )
        if materials:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                materials[part_material_group[index]],
                bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            )
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        declared = {
            slot: primvars_api.CreatePrimvar(
                slot,
                Sdf.ValueTypeNames.FloatArray,
                getattr(UsdGeom.Tokens, SLOT_INTERPOLATION[slot]),
            )
            for slot in SCALAR_SLOTS
        }
        variant_set = prim.GetVariantSets().AddVariantSet(VARIANT_SET)
        for name in (ORIGINAL_MATERIALS_FIELD, *SCALAR_SLOTS):
            variant_set.AddVariant(name)
        for slot in SCALAR_SLOTS:
            variant_set.SetVariantSelection(slot)
            with variant_set.GetVariantEditContext():
                values = (
                    face_values[:, face_slice]
                    if slot == FACE_SLOT
                    else vertex_values[slot][:, point_slice]
                )
                attr = declared[slot].GetAttr()
                for frame in frames:
                    time = (
                        Usd.TimeCode(float(frame))
                        if source_frames == "all"
                        else Usd.TimeCode.Default()
                    )
                    attr.Set(_floats(values[frame]), time)
        variant_set.SetVariantSelection(ORIGINAL_MATERIALS_FIELD)
        mesh_paths.append(mesh_path)
    stage.GetRootLayer().Save()
    return {
        "path": str(path),
        "mesh_paths": mesh_paths,
        "frames_authored": list(frames),
        "up_axis": "Y",
        "meters_per_unit": float(meters_per_unit),
        "d3plot_rotate_xyz": list(D3PLOT_ROTATE_XYZ),
    }


def _plain(value):
    """JSON-like data as the plain Python types USD customLayerData accepts
    (no ``None``: it becomes an empty string; numpy scalars become Python)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_plain(v) for v in value]
        if items and not all(type(i) is type(items[0]) for i in items):
            items = [str(i) for i in items]
        return items
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    return value


def write_variant_overlay(path, mesh_paths: list[str], field_name: str) -> int:
    """``runtime/variant_overlays/<field>.usda``: variant selections only."""
    Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt = _pxr()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    for mesh_path in mesh_paths:
        stage.OverridePrim(mesh_path).GetVariantSet(VARIANT_SET).SetVariantSelection(
            field_name
        )
    stage.GetRootLayer().Save()
    return len(mesh_paths)


def _define_merged_mesh(
    stage,
    layout: MergedLayout,
    points_frame0: np.ndarray,
    st: np.ndarray,
    st_interpolation: str,
    double_sided: bool = False,
):
    Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt = _pxr()
    for part in layout.parts:
        stage.OverridePrim(part_mesh_path(part.prim_name)).SetActive(False)
    mesh = UsdGeom.Mesh.Define(stage, MERGED_MESH_PATH)
    mesh.CreateFaceVertexCountsAttr(_ints(layout.face_counts))
    mesh.CreateFaceVertexIndicesAttr(_ints(layout.face_vertex_indices))
    mesh.CreatePointsAttr(_vec3f(points_frame0))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(bool(double_sided))
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord3fArray,
        getattr(UsdGeom.Tokens, st_interpolation),
    )
    primvar.Set(_vec3f(st))
    return mesh


def write_merged_scene(
    path,
    layout: MergedLayout,
    points_frame0: np.ndarray,
    st: np.ndarray,
    st_interpolation: str = "faceVarying",
    *,
    material_groups=None,
    part_material_group=None,
    double_sided: bool = False,
) -> None:
    """A merged static-topology schema: the source meshes deactivated and one
    ``/World/d3plot/OVRTXMergedTransient/Mesh`` with ``primvars:st``; with
    ``material_groups`` also one ``materialBind`` GeomSubset per group bound to
    ``/World/d3plot/Looks/OVRTXOriginalSection_XX`` (the runtime's materials)."""
    Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt = _pxr()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    mesh = _define_merged_mesh(
        stage, layout, points_frame0, st, st_interpolation, double_sided
    )
    if material_groups:
        for group_index, group in enumerate(material_groups):
            face_indexes = np.concatenate(
                [
                    np.arange(
                        part.face_offset,
                        part.face_offset + part.face_count,
                        dtype=np.int32,
                    )
                    for part, g in zip(layout.parts, part_material_group)
                    if g == group_index
                ]
                or [np.empty(0, dtype=np.int32)]
            )
            if len(face_indexes) == 0:
                continue
            subset = UsdGeom.Subset.CreateGeomSubset(
                mesh,
                f"OriginalMaterial_{group_index:02d}",
                UsdGeom.Tokens.face,
                _ints(face_indexes),
                "materialBind",
            )
            binding = UsdShade.MaterialBindingAPI.Apply(
                subset.GetPrim()
            ).GetDirectBindingRel()
            binding.SetTargets(
                [Sdf.Path(f"{D3PLOT_PATH}/Looks/{group['section_material_name']}")]
            )
    stage.GetRootLayer().Save()


def write_per_mesh_st_schema(
    path, layout: MergedLayout, per_mesh_st: list[np.ndarray], field_name: str
) -> None:
    """Per-mesh overrides selecting ``field_name`` and authoring face-varying
    ``primvars:st`` (``von_mises_atlas_schema.usdc`` / ``von_mises_schema.usdc``)."""
    Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt = _pxr()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    for part, st in zip(layout.parts, per_mesh_st):
        override = stage.OverridePrim(part_mesh_path(part.prim_name))
        override.GetVariantSet(VARIANT_SET).SetVariantSelection(field_name)
        primvar = UsdGeom.PrimvarsAPI(override).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord3fArray, UsdGeom.Tokens.faceVarying
        )
        primvar.Set(_vec3f(st))
    stage.GetRootLayer().Save()


def read_merged_st(path, mesh_path: str = MERGED_MESH_PATH):
    """``(face_counts, st, interpolation)`` of a merged schema (validation)."""
    Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt = _pxr()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"unable to open {path}")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
    if not mesh:
        raise RuntimeError(f"missing merged mesh {mesh_path} in {path}")
    primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    if not primvar:
        raise RuntimeError(f"{path} has no primvars:st")
    return (
        np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32),
        np.asarray(primvar.Get(), dtype=np.float32),
        str(primvar.GetInterpolation()),
    )
