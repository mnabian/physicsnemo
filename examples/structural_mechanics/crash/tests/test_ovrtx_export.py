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
"""OVRTX car-crash viewer export on a synthetic mesh (visualization/ovrtx).

Everything but the USD authoring runs without ``usd-core``: the atlas, layout,
slot-assignment, loader and CLI tests, and the bundle/validator tests that
replace the USD writers with stand-ins (``usd_stubs``). The three tests that
read the written stage need ``pxr`` and are skipped without it (run them with
``uv run --python 3.12 --with usd-core==24.11 --with numpy --with pillow --with pyvista --with pytest``).
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))

from visualization.ovrtx import atlas, materials  # noqa: E402
from visualization.ovrtx.fields import (  # noqa: E402
    FIELD_RANGES,
    SlotAssignment,
    face_max,
    face_mean,
    resolve_slots,
)
from visualization.ovrtx.scene import (  # noqa: E402
    FACE_SLOT,
    U_SLOT,
    V_SLOT,
    CrashScene,
    MergedLayout,
    connected_components,
    faces_from_padded,
    faces_to_padded,
    hex_boundary_faces,
    parse_frame_spec,
    synthetic_scene,
)


def test_parse_frame_spec():
    assert parse_frame_spec(None, 31) == list(range(25))
    assert parse_frame_spec("all", 4) == [0, 1, 2, 3]
    assert parse_frame_spec("2-4,7", 10) == [2, 3, 4, 7]
    assert parse_frame_spec("stride:3", 8) == [0, 3, 6]
    assert (
        len(parse_frame_spec("fit", 31)) == 25 and parse_frame_spec("fit", 31)[-1] == 30
    )
    with pytest.raises(ValueError):
        parse_frame_spec("30-31", 31)


def test_faces_from_padded_handles_both_triangle_conventions():
    cells = np.array(
        [[0, 1, 2, 2], [0, 1, 2, -1], [0, 1, 2, 3], [0, 0, 1, 2], [0, 1, 2, 1]]
    )
    counts, indices, rows = faces_from_padded(cells)
    assert counts.tolist() == [3, 3, 4]
    assert indices.tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2, 3]
    assert rows.tolist() == [0, 1, 2]
    padded = faces_to_padded(counts, indices)
    assert padded.tolist() == [[0, 1, 2, -1], [0, 1, 2, -1], [0, 1, 2, 3]]


def test_hex_boundary_faces_of_two_stacked_cubes():
    # two unit cubes sharing a face: 12 - 2 = 10 exterior faces
    base = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    top = base + 4
    faces = hex_boundary_faces(np.concatenate((base, top)))
    assert faces.shape == (10, 4)
    assert not any(sorted(f) == [4, 5, 6, 7] for f in faces.tolist())
    # LS-DYNA pentahedron: n5 = n6 and n7 = n8 collapse two edges -> 3 quads + 2 triangles
    collapsed = np.array([[0, 1, 2, 3, 4, 4, 6, 6]])
    wedge = hex_boundary_faces(collapsed)
    assert len(wedge) == 5 and (wedge[:, 2] == wedge[:, 3]).sum() == 2
    # one collapsed edge keeps six faces (two of them triangles)
    one_edge = hex_boundary_faces(np.array([[0, 1, 2, 3, 4, 5, 6, 6]]))
    assert len(one_edge) == 6 and (one_edge[:, 2] == one_edge[:, 3]).sum() == 2


def test_merged_layout_duplicates_shared_nodes_per_part():
    scene = synthetic_scene(num_frames=3, parts=2, grid=2)
    layout = MergedLayout.build(scene)
    assert len(layout.parts) == 2
    assert layout.face_count == scene.num_faces
    assert layout.point_count == sum(p.point_count for p in layout.parts)
    # every merged face indexes the part's own points
    for part, local in zip(layout.parts, layout.local_face_indices):
        assert local.max() < part.point_count
    gathered = layout.gather_positions(scene.positions)
    assert gathered.shape == (3, layout.point_count, 3)
    np.testing.assert_array_equal(
        gathered[:, layout.parts[1].point_offset],
        scene.positions[:, layout.parts[1].node_ids[0]],
    )
    assert (
        layout.summary()["triangle_face_count"] == 8
    )  # the last plate is triangulated


def test_face_reduce_and_slots():
    scene = synthetic_scene(num_frames=2, parts=1, grid=1)
    values = np.arange(2 * scene.num_nodes, dtype=np.float32).reshape(2, -1)
    mx = face_max(values, scene.face_counts, scene.face_indices)
    mean = face_mean(values, scene.face_counts, scene.face_indices)
    assert mx.shape == (2, scene.num_faces) and mean.shape == mx.shape
    assert np.all(mx >= mean)
    slots = resolve_slots(scene, SlotAssignment(face="error_mm", u="error_mm", v=None))
    assert slots[FACE_SLOT].shape == (2, scene.num_faces)
    assert np.all(slots[V_SLOT] == 0) and slots[U_SLOT] is scene.node_fields["error_mm"]
    with pytest.raises(KeyError):
        resolve_slots(scene, SlotAssignment(face="nope"))


def test_connected_components_separate_plates():
    scene = synthetic_scene(num_frames=1, parts=3, grid=2)
    labels = connected_components(
        scene.num_nodes, scene.face_counts, scene.face_indices
    )
    assert len(np.unique(labels)) == 3
    assert len(np.unique(labels[scene.node_part == 1])) == 1


def test_face_atlas_round_trip_and_layout():
    frames, faces = 7, 10
    rng = np.random.default_rng(1)
    values = rng.uniform(-3.0, 12.0, size=(frames, faces)).astype(np.float32)
    values[2, 3] = np.nan
    layout = atlas.FaceAtlasLayout.build(faces, frames)
    assert (layout.tile_side, layout.tile_grid) == (4, 3)
    image, lo, hi, nonfinite = atlas.encode_face_atlas(values, layout)
    assert image.shape == (12, 12) and image.dtype == np.uint16 and nonfinite == 1
    quantum = (hi - lo) / 65535.0
    for f in range(frames):
        decoded = atlas.decode_face_atlas(image, layout, f, lo, hi)
        finite = np.isfinite(values[f])
        assert np.abs(decoded[finite] - values[f][finite]).max() <= quantum / 2 + 1e-6
    # packed coordinates follow the demo's layout for triangles and quads
    counts = np.array([4, 3], dtype=np.int32)
    st = atlas.packed_face_coordinates(counts, 5, layout.tile_side)
    assert st.shape == (7, 3)
    m = atlas.PACKED_LOCAL_MARGIN
    np.testing.assert_allclose(
        st[0, :2], [(1 + m) / 4, (1 + m) / 8], rtol=0, atol=1e-7
    )  # face 5: column 1, row 1
    np.testing.assert_allclose(
        st[4, :2], [(2 + m) / 4, (1 + m + 4) / 8], rtol=0, atol=1e-7
    )  # triangle offset by a tile height
    cells = atlas.face_cell_coordinates(counts, 5, layout.tile_side)
    np.testing.assert_allclose(
        atlas.packed_from_cell_coordinates(counts, cells, layout.tile_side),
        st,
        atol=1e-7,
    )


def test_vertex_atlas_round_trip_across_banks():
    scene = synthetic_scene(num_frames=5, parts=2, grid=2)
    layout_mesh = MergedLayout.build(scene)
    values = layout_mesh.gather_nodes(scene.node_fields["displacement_mm"] + 0.5)
    corners = atlas.face_corner_indices(
        layout_mesh.face_counts, layout_mesh.face_vertex_indices
    )
    # force several banks: tile_side = 2*ceil(sqrt(16)) = 8, max side 16 -> grid 2 -> 4 frames/bank -> 2 banks
    layout = atlas.VertexAtlasLayout.build(
        layout_mesh.face_count, 5, max_texture_side=16
    )
    assert (layout.tile_cells, layout.bank_tile_grid, layout.bank_count) == (4, 2, 2)
    lo, hi, nonfinite = atlas.vertex_atlas_range(values, corners)
    for bank in range(layout.bank_count):
        page = atlas.encode_vertex_atlas_bank(values, corners, layout, bank, lo, hi)
        assert page.shape == (16, 16)
        first = bank * layout.frames_per_bank
        for frame in range(first, min(first + layout.frames_per_bank, 5)):
            decoded = atlas.decode_vertex_atlas(page, layout, frame, lo, hi)
            expected = atlas.vertex_corner_values(values[frame], corners)
            assert np.abs(decoded - expected).max() <= (hi - lo) / (2 * 65535.0) + 1e-6
    assert atlas.udim_file_name("u_vertex_atlas", 11) == "u_vertex_atlas.1012.png"
    st = atlas.vertex_texture_coordinates(layout_mesh.face_counts, layout.tile_cells)
    assert st.shape == (layout_mesh.face_vertex_count, 3) and st[:, 1].max() < 1.0


def test_scene_transform_and_frame_selection():
    scene = synthetic_scene(num_frames=4)
    yup = scene.transformed("z-up-to-y-up")
    np.testing.assert_allclose(
        yup.positions[..., 1], scene.positions[..., 2], atol=1e-6
    )
    np.testing.assert_allclose(
        yup.positions[..., 2], -scene.positions[..., 1], atol=1e-6
    )
    sub = scene.select_frames([0, 2])
    assert sub.num_frames == 2 and sub.frames == [0, 1] and sub.time_ms == [0.0, 10.0]
    assert sub.meta["source_frames"] == [0, 2]


def stand_in_templates(directory: Path) -> Path:
    """Stand-ins for the viewer release's shader templates (the bundle only copies them)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / materials.CAE_MATERIALS_MDL).write_text(
        'mdl 1.7;\nexport material ScalarColor(float v = scene::data_lookup_float("von_mises", 0.0f))\n'
        "  = material();\n"
    )
    for name in (
        materials.ATLAS_MDL,
        materials.VERTEX_ATLAS_MDL,
        *materials.RUNTIME_MATERIAL_TEMPLATES,
    ):
        (directory / name).write_text(f"// stand-in for {name}\n")
    return directory


def test_templates_dir_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv(materials.TEMPLATES_DIR_VARIABLE, raising=False)
    assert materials.resolve_templates_dir() == materials.DEFAULT_TEMPLATES_DIR
    monkeypatch.setenv(materials.TEMPLATES_DIR_VARIABLE, str(tmp_path / "env"))
    assert materials.resolve_templates_dir() == tmp_path / "env"
    assert materials.resolve_templates_dir(tmp_path / "arg") == tmp_path / "arg"
    # the repository ships no templates: every file is reported absent until copied in
    assert not any(materials.available_templates(tmp_path / "empty").values())
    assert set(
        materials.available_templates(stand_in_templates(tmp_path / "filled"))
    ) == set(materials.TEMPLATE_FILES)
    assert (
        materials.copy_template(
            materials.ATLAS_MDL, tmp_path / "out" / "a.mdl", tmp_path / "empty"
        )
        is None
    )
    copied = materials.copy_template(
        materials.ATLAS_MDL, tmp_path / "out" / "a.mdl", tmp_path / "filled"
    )
    assert copied is not None and copied.read_text().startswith("// stand-in")


@pytest.fixture(scope="module")
def synthetic_bundle(tmp_path_factory):
    pytest.importorskip("pxr")
    pytest.importorskip("PIL")
    from visualization.ovrtx.bundle import export_bundle

    scene = synthetic_scene(num_frames=6)
    out = tmp_path_factory.mktemp("bundle")
    manifest = export_bundle(
        scene,
        out,
        assignment=SlotAssignment.for_label("model"),
        max_texture_side=64,
        material_group_count=2,
        templates_dir=stand_in_templates(tmp_path_factory.mktemp("templates")),
        log=lambda *_: None,
    )
    assert manifest["missing_templates"] == []
    return scene, out, manifest


@pytest.fixture
def usd_stubs(monkeypatch):
    """Stand-ins for the pxr-dependent writers, so a bundle's non-USD files can
    be written and validated without usd-core (the USD files are empty and the
    validator's USD checks report themselves skipped)."""
    pytest.importorskip("PIL")
    from visualization.ovrtx import bundle, usd_writer, validate

    def touch(path, *args, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")

    def stage(path, scene, layout, *args, **kwargs):
        touch(path)
        return {
            "path": str(path),
            "mesh_paths": [
                usd_writer.part_mesh_path(p.prim_name) for p in layout.parts
            ],
            "frames_authored": list(range(scene.num_frames)),
            "up_axis": "Y",
            "meters_per_unit": usd_writer.DEMO_METERS_PER_UNIT,
            "d3plot_rotate_xyz": list(usd_writer.D3PLOT_ROTATE_XYZ),
        }

    def overlay(path, mesh_paths, field_name):
        touch(path)
        return len(mesh_paths)

    monkeypatch.setattr(usd_writer, "_pxr", lambda: None)
    monkeypatch.setattr(bundle, "write_source_stage", stage)
    monkeypatch.setattr(bundle, "write_variant_overlay", overlay)
    monkeypatch.setattr(bundle, "write_per_mesh_st_schema", touch)
    monkeypatch.setattr(bundle, "write_merged_scene", touch)
    monkeypatch.setattr(
        validate, "check_usd", lambda *a, **k: {"skipped": "USD stand-ins"}
    )
    return bundle


def test_bundle_without_templates_reports_them(tmp_path, usd_stubs):
    from visualization.ovrtx.bundle import export_bundle
    from visualization.ovrtx.validate import validate_bundle

    scene = synthetic_scene(num_frames=2, parts=1, grid=1)
    out = tmp_path / "bundle"
    empty = tmp_path / "no_templates"
    empty.mkdir()
    manifest = export_bundle(
        scene,
        out,
        assignment=SlotAssignment.for_label("model"),
        max_texture_side=64,
        material_group_count=1,
        templates_dir=empty,
        log=lambda *_: None,
    )
    assert set(manifest["missing_templates"]) == set(materials.MDL_TEMPLATES)
    # the colour map is regenerated, the shader files are not
    assert (out / "car_crash_assets/colormaps/contour.png").is_file()
    assert not (out / "car_crash_assets/cae_materials.mdl").exists()
    report = validate_bundle(out, frames=[0], samples=0)
    assert not report["ok"]
    # every one of the five MDL templates is required at its destination
    for name in materials.MDL_TEMPLATES:
        for rel in materials.TEMPLATE_DESTINATIONS[name]:
            assert f"required file missing: {rel}" in report["errors"]
    # a templates directory holding only the three files the entrypoint checks is still incomplete
    partial = stand_in_templates(tmp_path / "partial")
    (partial / materials.ATLAS_MDL).unlink()
    (partial / materials.VERTEX_ATLAS_MDL).unlink()
    export_bundle(
        scene,
        out,
        assignment=SlotAssignment.for_label("model"),
        max_texture_side=64,
        material_group_count=1,
        templates_dir=partial,
        overwrite=True,
        log=lambda *_: None,
    )
    report = validate_bundle(out, frames=[0], samples=0)
    assert not report["ok"]
    missing = {e for e in report["errors"] if e.startswith("required file missing")}
    assert missing == {
        f"required file missing: {rel}"
        for name in (materials.ATLAS_MDL, materials.VERTEX_ATLAS_MDL)
        for rel in materials.TEMPLATE_DESTINATIONS[name]
    }


def test_bundle_non_usd_checks_validate_without_pxr(tmp_path, usd_stubs, monkeypatch):
    from visualization.ovrtx import usd_writer
    from visualization.ovrtx.bundle import export_bundle
    from visualization.ovrtx.validate import validate_bundle

    scene = synthetic_scene(num_frames=6)
    out = tmp_path / "bundle"
    manifest = export_bundle(
        scene,
        out,
        assignment=SlotAssignment.for_label("model"),
        max_texture_side=64,
        material_group_count=2,
        templates_dir=stand_in_templates(tmp_path / "templates"),
        log=lambda *_: None,
    )
    assert manifest["missing_templates"] == []
    for name in materials.MDL_TEMPLATES:
        for rel in materials.TEMPLATE_DESTINATIONS[name]:
            assert (out / rel).is_file(), rel
    layout = MergedLayout.build(scene)
    counts = np.load(out / "runtime/deformer/face_counts.npy")
    assert counts.dtype == np.int32 and counts.tolist() == layout.face_counts.tolist()
    report = validate_bundle(out, frames=[0, 3, 5], samples=0)
    assert report["ok"], report["errors"]
    assert report["checks"]["face_atlas_round_trip"]["within_quantization_bound"]
    for slot in (U_SLOT, V_SLOT):
        assert report["checks"][f"vertex_atlas_round_trip_{slot}"][
            "within_quantization_bound"
        ]
    assert "skipped" in report["checks"]["usd"]
    assert all("environment" not in line for line in report["not_verified"])
    # the sidecar is what the validator reads for the face counts; a wrong one is an error
    from visualization.ovrtx.validate import (
        ValidationError,
        _face_counts_from_schema_or_sidecar,
    )

    merged_meta = json.loads(
        (out / "runtime/ovrtx_merged/merged_geometry.json").read_text()
    )
    np.testing.assert_array_equal(
        _face_counts_from_schema_or_sidecar(out / "runtime", merged_meta),
        layout.face_counts,
    )
    np.save(out / "runtime/deformer/face_counts.npy", counts[:-1])
    with pytest.raises(ValidationError):
        _face_counts_from_schema_or_sidecar(out / "runtime", merged_meta)
    # without the sidecar the counts come from the USD schema, which needs pxr
    (out / "runtime/deformer/face_counts.npy").unlink()

    def no_pxr():
        raise ImportError("no usd-core")

    monkeypatch.setattr(usd_writer, "_pxr", no_pxr)
    with pytest.raises(ValidationError, match="face_counts.npy"):
        _face_counts_from_schema_or_sidecar(out / "runtime", merged_meta)


def test_export_checks_prerequisites_before_writing_and_cleans_up(
    tmp_path, usd_stubs, monkeypatch
):
    from visualization.ovrtx import bundle, usd_writer

    scene = synthetic_scene(num_frames=2, parts=1, grid=1)
    templates = stand_in_templates(tmp_path / "templates")
    out = tmp_path / "new" / "bundle"
    kwargs = dict(
        max_texture_side=64,
        material_group_count=1,
        templates_dir=templates,
        log=lambda *_: None,
    )
    # missing inputs are reported before any directory exists
    with pytest.raises(FileNotFoundError, match="environment HDR"):
        bundle.export_bundle(
            scene, out, environment_hdr=str(tmp_path / "missing.hdr"), **kwargs
        )
    with pytest.raises(FileNotFoundError, match="templates directory"):
        bundle.export_bundle(
            scene, out, **{**kwargs, "templates_dir": tmp_path / "nowhere"}
        )
    assert not (tmp_path / "new").exists()

    # so is a missing usd-core
    def no_pxr():
        raise ImportError("no usd-core here")

    with monkeypatch.context() as m:
        m.setattr(usd_writer, "_pxr", no_pxr)
        with pytest.raises(ImportError, match="usd-core"):
            bundle.export_bundle(scene, out, **kwargs)
    assert not (tmp_path / "new").exists()
    # a failure while writing removes what the call created, and the retry is accepted
    with monkeypatch.context() as m:

        def boom(*args, **kwargs):
            raise RuntimeError("disk full")

        m.setattr(bundle, "write_merged_scene", boom)
        with pytest.raises(RuntimeError, match="disk full"):
            bundle.export_bundle(scene, out, **kwargs)
    assert not (tmp_path / "new").exists()
    manifest = bundle.export_bundle(scene, out, **kwargs)
    assert manifest["missing_templates"] == []
    # inside an existing directory only the bundle's two directories are touched
    sibling = out / "notes.txt"
    sibling.write_text("keep me")
    with monkeypatch.context() as m:
        m.setattr(bundle, "write_merged_scene", boom)
        with pytest.raises(RuntimeError):
            bundle.export_bundle(scene, out, overwrite=True, **kwargs)
    assert sibling.read_text() == "keep me"
    assert not (out / "runtime").exists() and not (out / "car_crash_assets").exists()
    # the directory is free again, so the retry needs no --overwrite; the next one does
    bundle.export_bundle(scene, out, **kwargs)
    with pytest.raises(FileExistsError):
        bundle.export_bundle(scene, out, **kwargs)


def test_bundle_validates_like_the_runtime(synthetic_bundle):
    from visualization.ovrtx.validate import validate_bundle

    scene, out, manifest = synthetic_bundle
    report = validate_bundle(out, frames=[0, 3, 5], samples=0)
    assert report["ok"], report["errors"]
    assert report["checks"]["face_atlas_round_trip"]["within_quantization_bound"]
    assert report["checks"][f"vertex_atlas_round_trip_{U_SLOT}"][
        "within_quantization_bound"
    ]
    assert report["checks"]["usd"]["source_meshes_checked"] == 3
    assert report["checks"]["usd"]["merged_scene_st_max_error"] < 1e-6
    merged = json.loads((out / "runtime/ovrtx_merged/merged_geometry.json").read_text())
    assert merged["point_count"] == manifest["layout"]["point_count"]
    positions = np.load(
        out / "runtime/ovrtx_merged/merged_positions.npy", mmap_mode="r"
    )
    assert positions.shape == (6, merged["point_count"], 3)
    source = np.load(out / "runtime/deformer/merged_point_source.npy")
    np.testing.assert_array_equal(positions[2], scene.positions[2][source])
    assert (out / "runtime/variant_overlays" / f"{V_SLOT}.usda").is_file()
    assert (out / "runtime/ovrtx_atlas/colormaps").is_symlink()


def test_bundle_source_stage_structure(synthetic_bundle):
    from pxr import Usd, UsdGeom, UsdShade

    scene, out, manifest = synthetic_bundle
    stage = Usd.Stage.Open(str(out / "car_crash_assets/main_project.usd"))
    assert UsdGeom.GetStageUpAxis(stage) == "Y" and stage.GetEndTimeCode() == 5
    d3plot = stage.GetPrimAtPath("/World/d3plot")
    rotate = [
        op
        for op in UsdGeom.Xformable(d3plot).GetOrderedXformOps()
        if "rotate" in op.GetOpName()
    ][0]
    assert tuple(rotate.Get()) == (-90.0, 0.0, 0.0)
    mesh_path = manifest["parts"][0]["prim_path"]
    prim = stage.GetPrimAtPath(mesh_path)
    mesh = UsdGeom.Mesh(prim)
    assert len(mesh.GetPointsAttr().GetTimeSamples()) == 6
    variants = prim.GetVariantSet("PhysicsField")
    assert set(variants.GetVariantNames()) >= {
        "Solid_Color_Default",
        FACE_SLOT,
        U_SLOT,
        V_SLOT,
    }
    variants.SetVariantSelection(U_SLOT)
    primvar = UsdGeom.Primvar(prim.GetAttribute(f"primvars:{U_SLOT}"))
    assert primvar.GetInterpolation() == "vertex"
    values = np.asarray(primvar.Get(Usd.TimeCode(4.0)))
    layout = MergedLayout.build(scene)
    part = layout.parts[0]
    np.testing.assert_allclose(
        values, scene.node_fields["error_mm"][4][part.node_ids], rtol=0, atol=1e-6
    )
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    shader = UsdShade.Shader(material.GetPrim().GetChild("Shader"))
    assert shader.GetSourceAssetSubIdentifier("mdl") == "OmniSurface"


def write_vtp_sequence(root: Path, scene: CrashScene) -> tuple[Path, Path]:
    """``frame_XXX_pred.vtp`` / ``frame_XXX_exact.vtp`` files like
    ``inference.save_vtp_sequence`` writes them (frames numbered from 1, the
    prediction offset by 1 mm in z, a ``pred_stress`` point field)."""
    pv = pytest.importorskip("pyvista")
    pred_dir, exact_dir = root / "pred", root / "exact"
    pred_dir.mkdir(parents=True)
    exact_dir.mkdir(parents=True)
    faces = np.concatenate(
        [
            [c, *scene.face_indices[o : o + c]]
            for c, o in zip(
                scene.face_counts, np.cumsum(np.r_[0, scene.face_counts[:-1]])
            )
        ]
    )
    for t in range(scene.num_frames):
        mesh = pv.PolyData(scene.positions[t] + (0.0, 0.0, 1.0), faces)
        mesh.point_data["pred_stress"] = np.full(
            scene.num_nodes, float(t), dtype=np.float32
        )
        mesh.save(pred_dir / f"frame_{t + 1:03d}_pred.vtp")
        pv.PolyData(scene.positions[t], faces).save(
            exact_dir / f"frame_{t + 1:03d}_exact.vtp"
        )
    return pred_dir, exact_dir


def test_vtp_sequence_source(tmp_path):
    from visualization.ovrtx.scene import load_vtp_sequence

    scene = synthetic_scene(num_frames=3, parts=1, grid=2)
    pred_dir, exact_dir = write_vtp_sequence(tmp_path, scene)
    loaded = load_vtp_sequence(pred_dir, exact_dir, parts="single")
    assert loaded.num_frames == 3 and loaded.num_faces == scene.num_faces
    np.testing.assert_allclose(loaded.node_fields["error_mm"], 1.0, atol=1e-6)
    assert loaded.node_fields["stress"][2].max() == 2.0
    assert loaded.meta["source_frames"] == [1, 2, 3]
    # frame numbers count 5 ms steps from the initial state: real milliseconds, not indexes
    assert loaded.time_ms == [5.0, 10.0, 15.0] and loaded.meta["dt_ms"] == 5.0
    coarse = load_vtp_sequence(pred_dir, None, parts="single", dt_ms=2.5)
    assert coarse.time_ms == [2.5, 5.0, 7.5]
    assert "error_mm" not in coarse.node_fields and coarse.meta["exact_dir"] is None
    with pytest.raises(ValueError, match="dt_ms"):
        load_vtp_sequence(pred_dir, None, dt_ms=0.0)
    # an exact directory without *_exact.vtp files is an error, not a silently absent error field
    empty = tmp_path / "exact_typo"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="exact_typo"):
        load_vtp_sequence(pred_dir, empty)
    with pytest.raises(FileNotFoundError, match="no such directory"):
        load_vtp_sequence(pred_dir, tmp_path / "does_not_exist")


def test_slot_assignment_fallbacks_carry_their_own_labels_and_ranges():
    base = synthetic_scene(num_frames=2, parts=1, grid=1)

    def scene_with(node_fields):
        return CrashScene(
            name="s",
            positions=base.positions,
            face_counts=base.face_counts,
            face_indices=base.face_indices,
            node_part=base.node_part,
            face_part=base.face_part,
            node_fields=node_fields,
        )

    displacement = base.node_fields["displacement_mm"]
    error = base.node_fields["error_mm"]
    # the defaults keep the label-specific wording when their fields exist
    assert (
        SlotAssignment.for_scene(base, "model").as_dict()
        == SlotAssignment.for_label("model").as_dict()
    )
    # a ground truth with displacement only (a VTP sequence): every slot shows it
    truth = SlotAssignment.for_scene(
        scene_with({"displacement_mm": displacement}), "ground_truth"
    )
    assert (truth.face, truth.u, truth.v) == ("displacement_mm",) * 3
    assert truth.labels[FACE_SLOT] == (
        "Displacement magnitude [mm], max over the face's nodes"
    )
    assert truth.labels[V_SLOT] == "Displacement magnitude [mm]"
    assert truth.ranges == {slot: (0.0, 1000.0) for slot in (FACE_SLOT, U_SLOT, V_SLOT)}
    assert "crossing" not in json.dumps(truth.as_dict())
    # a model with the error field: the face slot shows the error, not "spurious crossings"
    model = SlotAssignment.for_scene(
        scene_with({"displacement_mm": displacement, "error_mm": error}), "model"
    )
    assert model.face == "error_mm" and model.u == "error_mm"
    assert model.labels[FACE_SLOT].startswith("Position error vs. ground truth [mm]")
    assert model.ranges[FACE_SLOT] == FIELD_RANGES["error_mm"] == (0.0, 30.0)
    assert model.labels[U_SLOT] == "Position error vs. ground truth [mm]"
    assert model.ranges[V_SLOT] == (0.0, 1000.0)
    # an explicit field without a documented range gets its name and the range of its values
    stress = np.full((2, base.num_nodes), 3.0, dtype=np.float32)
    custom = SlotAssignment.for_scene(
        scene_with({"displacement_mm": displacement, "stress": stress}),
        "ground_truth",
        u="stress",
    )
    assert custom.labels[U_SLOT] == "stress" and custom.ranges[U_SLOT] == (0.0, 3.0)
    assert custom.v == "displacement_mm" and custom.ranges[V_SLOT] == (0.0, 1000.0)
    assert custom.as_dict()["suggested_ranges"][U_SLOT] == [0.0, 3.0]


def test_yaris_dump_source(tmp_path):
    from visualization.ovrtx.scene import load_yaris_dump

    base = synthetic_scene(num_frames=4, parts=2, grid=2)  # nodes 0..8 and 9..17
    case = tmp_path / "case_01"
    for label in ("ground_truth", "teacher"):
        (case / label).mkdir(parents=True)
    np.savez(
        case / "mesh.npz",
        part_id=base.node_part,
        shell_faces=faces_to_padded(base.face_counts, base.face_indices),
        shell_part_id=base.face_part,
        solid_hexes=np.arange(9, 17)[
            None
        ],  # one hexahedron on the second plate's nodes
    )
    np.save(case / "ground_truth" / "positions.npy", base.positions)
    np.save(case / "teacher" / "positions.npy", base.positions + (1.0, 0.0, 0.0))
    depth = np.zeros((4, base.num_nodes), dtype=np.float16)
    depth[1:, 0] = 2.0  # node 0 crosses from frame 1 on
    np.save(case / "teacher" / "depth_crossing.npy", depth)
    np.save(case / "ground_truth" / "depth_crossing.npy", np.zeros_like(depth))
    for label in ("ground_truth", "teacher"):
        (case / label / "meta.json").write_text(
            json.dumps(
                {"case": "case_01", "time_ms": [0, 5, 10, 15], "frames": [0, 5, 10, 15]}
            )
        )
    scene, meta = load_yaris_dump(case, "teacher", frame_spec="0-2")
    assert meta["case"] == "case_01" and scene.name == "case_01__teacher"
    assert scene.num_frames == 3 and scene.time_ms == [0.0, 5.0, 10.0]
    assert scene.meta["source_frames"] == [0, 5, 10]
    assert scene.num_faces == base.num_faces + 6
    assert scene.meta["solid_boundary_faces"] == 6
    assert scene.meta["shell_faces_dropped_degenerate"] == 0
    np.testing.assert_allclose(scene.node_fields["error_mm"], 1.0, atol=1e-6)
    assert set(scene.face_fields) == {"penetration_crossing_face_mm"}
    assert "truth_penetration_crossing_mm" in scene.node_fields
    face_depth = scene.face_fields["penetration_crossing_face_mm"]
    padded = faces_to_padded(scene.face_counts, scene.face_indices)
    touches_node0 = (padded == 0).any(axis=1)
    assert face_depth[0].max() == 0.0
    np.testing.assert_array_equal(face_depth[1] > 0, touches_node0)
    assert face_depth[2][touches_node0].min() == 2.0
    truth, _ = load_yaris_dump(case, "ground_truth", include_solids=False)
    assert truth.num_faces == base.num_faces and truth.num_frames == 4
    assert np.all(truth.node_fields["error_mm"] == 0)
    assert "truth_penetration_crossing_mm" not in truth.node_fields


def test_export_cli_frame_selection_and_single_json_report(
    tmp_path, monkeypatch, capsys
):
    from visualization.ovrtx import export_ovrtx, validate

    full = synthetic_scene(num_frames=30, parts=1, grid=1)
    monkeypatch.setattr(
        export_ovrtx,
        "load_scene",
        lambda args: (
            full.select_frames(parse_frame_spec(args.frames, full.num_frames)),
            {},
        ),
    )
    captured = {}

    def fake_export(scene, out, **kwargs):
        captured["scene"], captured["out"] = scene, out
        captured.update(kwargs)
        return {
            "scene": scene.name,
            "frames": scene.frames,
            "layout": {},
            "slots": kwargs["assignment"].as_dict(),
            "bounds_source_frame": {},
            "seconds": 0.0,
        }

    monkeypatch.setattr(export_ovrtx, "export_bundle", fake_export)
    out = str(tmp_path / "bundle")
    # 'all' (and any other selection of more than 25 frames) is rejected with advice
    for spec, count in (("all", 30), ("0-27", 28), ("stride:1", 30)):
        with pytest.raises(SystemExit) as exc:
            export_ovrtx.main(["--synthetic", "--out", out, "--frames", spec])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert f"selects {count} frames but the viewer's timeline holds 25" in err
        assert "--frames fit" in err
    assert not captured
    # 'fit' and 'demo' give 25 frames, 'fit' spread over the trajectory
    assert export_ovrtx.main(["--synthetic", "--out", out, "--frames", "fit"]) == 0
    assert captured["scene"].num_frames == 25
    assert captured["scene"].meta["source_frames"][-1] == 29
    summary = json.loads(capsys.readouterr().out)
    assert summary["frames"] == list(range(25)) and "validation" not in summary
    assert export_ovrtx.main(["--synthetic", "--out", out, "--frames", "demo"]) == 0
    assert captured["scene"].meta["source_frames"] == list(range(25))
    capsys.readouterr()
    # --validate: one JSON document with the report nested, exit status from the report
    monkeypatch.setattr(
        validate, "validate_bundle", lambda path: {"ok": True, "errors": []}
    )
    assert export_ovrtx.main(["--synthetic", "--out", out, "--validate"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["validation"] == {"ok": True, "errors": []}
    assert summary["slots"]["slots"][FACE_SLOT] == "penetration_crossing_face_mm"
    monkeypatch.setattr(
        validate, "validate_bundle", lambda path: {"ok": False, "errors": ["x"]}
    )
    assert export_ovrtx.main(["--synthetic", "--out", out, "--validate"]) == 1
    assert json.loads(capsys.readouterr().out)["validation"]["ok"] is False


def test_validate_distinguishes_missing_and_empty_bundle_dirs(tmp_path):
    from visualization.ovrtx.validate import validate_bundle

    report = validate_bundle(tmp_path / "nowhere")
    assert report["errors"] == [
        f"bundle directory does not exist: {tmp_path / 'nowhere'}"
    ]
    empty = tmp_path / "empty"
    empty.mkdir()
    report = validate_bundle(empty)
    assert not report["ok"] and len(report["errors"]) == 1
    assert "neither car_crash_assets/ nor runtime/" in report["errors"][0]
    (tmp_path / "file").write_text("")
    assert "not a directory" in validate_bundle(tmp_path / "file")["errors"][0]
    assert all("environment" not in line for line in report["not_verified"])


class _RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


def _inference_worker(cfg: dict):
    """An InferenceWorker with only what the OVRTX hook touches."""
    pytest.importorskip("torch")
    pytest.importorskip("hydra")
    import inference
    from omegaconf import OmegaConf

    worker = inference.InferenceWorker.__new__(inference.InferenceWorker)
    worker.cfg = OmegaConf.create(cfg)
    worker.logger = _RecordingLogger()
    worker.dist = SimpleNamespace(rank=0)
    return worker


def test_inference_hook_is_off_or_skips_without_usd_core(tmp_path, monkeypatch):
    from visualization.ovrtx import usd_writer

    cfg = {
        "inference": {
            "ovrtx_export_dir": None,
            "ovrtx_frames": None,
            "ovrtx_parts": None,
            "ovrtx_templates_dir": None,
        }
    }
    worker = _inference_worker(cfg)
    root = tmp_path / "bundles"
    # unset / empty export directory: nothing happens, nothing is logged
    worker._export_ovrtx(None, "run", str(tmp_path / "pred"), None)
    worker._export_ovrtx("", "run", str(tmp_path / "pred"), None)
    assert worker.logger.messages == []

    def no_pxr():
        raise ImportError("writing USD needs usd-core")

    monkeypatch.setattr(usd_writer, "_pxr", no_pxr)
    worker._export_ovrtx(str(root), "run", str(tmp_path / "pred"), None)
    assert worker.logger.messages == [
        ("warning", "[Rank 0] OVRTX export skipped: writing USD needs usd-core")
    ]
    assert not root.exists()


def test_inference_hook_writes_bundle_and_survives_failures(tmp_path, usd_stubs):
    scene = synthetic_scene(num_frames=3, parts=1, grid=2)
    pred_dir, exact_dir = write_vtp_sequence(tmp_path / "vtps", scene)
    root = tmp_path / "bundles"
    worker = _inference_worker(
        {
            "inference": {
                "ovrtx_export_dir": str(root),
                "ovrtx_frames": None,
                "ovrtx_parts": None,
                "ovrtx_templates_dir": str(stand_in_templates(tmp_path / "t")),
            },
            "model": {"dt": 0.002},
        }
    )
    worker._export_ovrtx(str(root), "run_a", str(pred_dir), str(exact_dir))
    levels = [level for level, _ in worker.logger.messages]
    assert "error" not in levels and "warning" not in levels
    manifest = json.loads(
        (root / "rank0" / "run_a" / "runtime/deformer/deformer_scene.json").read_text()
    )
    assert manifest["time_ms"] == [2.0, 4.0, 6.0]  # model.dt = 2 ms, frames 1..3
    assert manifest["slots"]["slots"] == {
        FACE_SLOT: "error_mm",
        U_SLOT: "error_mm",
        V_SLOT: "displacement_mm",
    }
    assert manifest["slots"]["labels"][FACE_SLOT].startswith("Position error")
    assert manifest["missing_templates"] == []
    # a failing export is logged and does not raise (the VTP frames are already written)
    worker.logger.messages.clear()
    worker._export_ovrtx(str(root), "run_b", str(tmp_path / "no_such_pred"), None)
    assert [level for level, _ in worker.logger.messages] == ["error"]
    assert "run_b" in worker.logger.messages[0][1]
    assert not (root / "rank0" / "run_b").exists()
