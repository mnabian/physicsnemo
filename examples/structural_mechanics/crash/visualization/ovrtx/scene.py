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
"""Fixed-topology crash scenes for the OVRTX car-crash viewer export.

A :class:`CrashScene` is one trajectory on one surface mesh: positions per
frame, triangle/quad faces, a part id per node and per face, and per-node or
per-face scalar fields per frame. Scenes come from the Yaris prediction dumps
(``dump_yaris_predictions.py``), from the body-in-white VTP frame sequences written by
``inference.save_vtp_sequence`` or from arrays in memory (tests).

:class:`MergedLayout` is the demo's "merge without welding" view of a scene:
one USD Mesh prim per part whose points are the nodes its faces reference, so
nodes shared by two parts are duplicated, and the concatenation of those prims
in part order is the merged transient mesh the viewer plays back.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Demo field slots (hard-coded in src/car_crash_ovrtx_ovstream.py of the demo).
FACE_SLOT = "Von_Mises_Von_Mises_layer_Outer_ip_Centroid"  # per-face 16-bit atlas
U_SLOT = "U_magnitude_layer_NULL_ip_NULL"  # per-vertex UDIM atlas
V_SLOT = "V_magnitude_layer_NULL_ip_NULL"  # per-vertex UDIM atlas
ORIGINAL_MATERIALS_FIELD = "Solid_Color_Default"
SCALAR_SLOTS = (FACE_SLOT, U_SLOT, V_SLOT)

# The demo's timeline is frames 0..24 at 24 time codes per second.
DEMO_FRAME_COUNT = 25
DEMO_FPS = 24.0

# Named rigid transforms applied to positions (fields are unchanged).
TRANSFORMS = {
    "identity": np.eye(3, dtype=np.float64),
    # LS-DYNA / Yaris and body-in-white data are z-up; the demo camera treats +y as up.
    "z-up-to-y-up": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
}

_FRAME_FILE = re.compile(r"^(?P<prefix>.+)_(?P<frame>\d{3})_(?P<kind>pred|exact)\.vtp$")


def parse_frame_spec(spec: str | None, available: int) -> list[int]:
    """Frame indexes selected by ``spec`` out of ``available`` frames.

    ``None``/``"demo"``: the first ``DEMO_FRAME_COUNT`` frames; ``"all"``: every
    frame; ``"a-b"``: an inclusive range; ``"a,b,c"``: a list; ``"stride:k"``:
    every k-th frame; ``"fit"``: ``DEMO_FRAME_COUNT`` frames spread evenly over
    the whole trajectory (nearest available frames).
    """
    if available <= 0:
        raise ValueError("no frames available")
    if spec is None or spec.strip().lower() in ("", "demo"):
        return list(range(min(available, DEMO_FRAME_COUNT)))
    text = spec.strip().lower()
    if text == "all":
        return list(range(available))
    if text == "fit":
        count = min(available, DEMO_FRAME_COUNT)
        return sorted(set(int(round(v)) for v in np.linspace(0, available - 1, count)))
    if text.startswith("stride:"):
        stride = int(text.split(":", 1)[1])
        if stride < 1:
            raise ValueError("stride must be positive")
        return list(range(0, available, stride))
    frames: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"descending frame range {part!r}")
            frames.extend(range(lo, hi + 1))
        else:
            frames.append(int(part))
    frames = list(dict.fromkeys(frames))
    if not frames:
        raise ValueError("empty frame selection")
    bad = [f for f in frames if f < 0 or f >= available]
    if bad:
        raise ValueError(f"frames {bad} outside [0, {available - 1}]")
    return frames


def faces_from_padded(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangles/quads from ``[M, 4]`` cells that encode triangles either with a
    repeated last id (LS-DYNA / the Yaris stores) or with ``-1`` padding.

    Returns ``(face_counts [K], face_indices [sum], source_rows [K])``. Cells with
    fewer than three distinct nodes, or four nodes with a repetition other than
    the last one, are dropped (degenerate shells).
    """
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] not in (3, 4):
        raise ValueError("cells must be [M, 3] or [M, 4]")
    if cells.shape[1] == 3:
        keep = (
            (cells[:, 0] != cells[:, 1])
            & (cells[:, 1] != cells[:, 2])
            & (cells[:, 0] != cells[:, 2])
        )
        keep &= (cells >= 0).all(axis=1)
        tri = cells[keep]
        return (
            np.full(len(tri), 3, dtype=np.int32),
            tri.reshape(-1).astype(np.int32),
            np.flatnonzero(keep),
        )
    valid3 = (cells[:, :3] >= 0).all(axis=1)
    distinct3 = (
        (cells[:, 0] != cells[:, 1])
        & (cells[:, 1] != cells[:, 2])
        & (cells[:, 0] != cells[:, 2])
    )
    is_tri = (cells[:, 3] < 0) | (cells[:, 3] == cells[:, 2])
    quad_ok = ~is_tri & (cells[:, 3] != cells[:, 0]) & (cells[:, 3] != cells[:, 1])
    tri_ok = is_tri & valid3 & distinct3
    quad_ok &= valid3 & distinct3
    keep = tri_ok | quad_ok
    rows = np.flatnonzero(keep)
    counts = np.where(tri_ok[rows], 3, 4).astype(np.int32)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    indices = np.empty(offsets[-1], dtype=np.int64)
    tri_rows, quad_rows = rows[counts == 3], rows[counts == 4]
    tri_off, quad_off = offsets[:-1][counts == 3], offsets[:-1][counts == 4]
    for k in range(3):
        indices[tri_off + k] = cells[tri_rows, k]
    for k in range(4):
        indices[quad_off + k] = cells[quad_rows, k]
    return counts, indices.astype(np.int32), rows


def faces_to_padded(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    """``[M, 4]`` int64 with ``-1`` padding for triangles (inverse of the above)."""
    counts = np.asarray(face_counts, dtype=np.int64)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    out = np.full((len(counts), 4), -1, dtype=np.int64)
    indices = np.asarray(face_indices, dtype=np.int64)
    for k in range(4):
        rows = counts > k
        out[rows, k] = indices[offsets[:-1][rows] + k]
    return out


def face_offsets(face_counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(face_counts, dtype=np.int64)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


def hex_boundary_faces(hexes: np.ndarray) -> np.ndarray:
    """Exterior faces ``[B, 4]`` (triangles as ``[a, b, c, c]``) of a hexahedral
    mesh (LS-DYNA node order); collapsed hexes contribute their distinct faces."""
    hexes = np.asarray(hexes, dtype=np.int64)
    if hexes.size == 0:
        return np.empty((0, 4), dtype=np.int64)
    if hexes.ndim != 2 or hexes.shape[1] != 8:
        raise ValueError("hexes must be [S, 8]")
    corner_sets = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    faces = np.concatenate([hexes[:, list(c)] for c in corner_sets])
    # Canonical key: sorted distinct ids; faces with < 3 distinct nodes are dropped.
    sorted_faces = np.sort(faces, axis=1)
    distinct = (np.diff(sorted_faces, axis=1) != 0).sum(axis=1) + 1
    faces = faces[distinct >= 3]
    sorted_faces = sorted_faces[distinct >= 3]
    distinct = distinct[distinct >= 3]
    # Triangles (one repeated id) are keyed on their three distinct ids.
    keys = sorted_faces.copy()
    tri = distinct == 3
    if tri.any():
        tri_keys = sorted_faces[tri]
        dup = np.diff(tri_keys, axis=1) == 0  # [T, 3]
        # remove the duplicated entry: keep first occurrence of each id
        cleaned = np.empty((tri_keys.shape[0], 4), dtype=np.int64)
        for row, (k, d) in enumerate(zip(tri_keys, dup)):
            ids = [k[0]] + [k[j + 1] for j in range(3) if not d[j]]
            cleaned[row] = ids + [-1]
        keys[tri] = cleaned
    _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    boundary = faces[np.sort(first[counts == 1])]
    out = boundary.copy()
    tri_rows = (np.diff(np.sort(out, axis=1), axis=1) != 0).sum(axis=1) + 1 == 3
    for row in np.flatnonzero(tri_rows):
        ids = list(dict.fromkeys(out[row].tolist()))
        out[row] = ids + [ids[-1]]
    return out


@dataclass
class CrashScene:
    """One trajectory on a fixed surface mesh (positions in millimetres unless
    ``units`` says otherwise)."""

    name: str
    positions: np.ndarray  # [F, N, 3] float32
    face_counts: np.ndarray  # [M] int32, values 3 or 4
    face_indices: np.ndarray  # [sum(face_counts)] int32 into N
    node_part: np.ndarray  # [N] int32
    face_part: np.ndarray  # [M] int32
    frames: list[int] = field(default_factory=list)  # demo frame numbers 0..F-1
    time_ms: list[float] = field(default_factory=list)  # physical time per frame
    node_fields: dict[str, np.ndarray] = field(default_factory=dict)  # name -> [F, N]
    face_fields: dict[str, np.ndarray] = field(default_factory=dict)  # name -> [F, M]
    part_names: dict[int, str] = field(default_factory=dict)
    units: str = "mm"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float32)
        if self.positions.ndim != 3 or self.positions.shape[2] != 3:
            raise ValueError(f"positions must be [F, N, 3], got {self.positions.shape}")
        self.face_counts = np.ascontiguousarray(self.face_counts, dtype=np.int32)
        self.face_indices = np.ascontiguousarray(self.face_indices, dtype=np.int32)
        if not np.all(np.isin(self.face_counts, (3, 4))):
            raise ValueError("faces must be triangles or quadrilaterals")
        if int(self.face_counts.sum()) != len(self.face_indices):
            raise ValueError("face_counts and face_indices disagree")
        n = self.num_nodes
        if len(self.face_indices) and (
            self.face_indices.min() < 0 or self.face_indices.max() >= n
        ):
            raise ValueError("face_indices outside the node range")
        self.node_part = np.ascontiguousarray(self.node_part, dtype=np.int32).reshape(
            -1
        )
        self.face_part = np.ascontiguousarray(self.face_part, dtype=np.int32).reshape(
            -1
        )
        if self.node_part.shape != (n,):
            raise ValueError("node_part must have one entry per node")
        if self.face_part.shape != (self.num_faces,):
            raise ValueError("face_part must have one entry per face")
        if not self.frames:
            self.frames = list(range(self.num_frames))
        if len(self.frames) != self.num_frames:
            raise ValueError("frames must have one entry per position frame")
        if not self.time_ms:
            self.time_ms = [float(f) for f in self.frames]
        if len(self.time_ms) != self.num_frames:
            raise ValueError("time_ms must have one entry per frame")
        for name, values in list(self.node_fields.items()):
            values = np.ascontiguousarray(values, dtype=np.float32)
            if values.shape != (self.num_frames, n):
                raise ValueError(
                    f"node field {name!r} must be [F, N], got {values.shape}"
                )
            self.node_fields[name] = values
        for name, values in list(self.face_fields.items()):
            values = np.ascontiguousarray(values, dtype=np.float32)
            if values.shape != (self.num_frames, self.num_faces):
                raise ValueError(
                    f"face field {name!r} must be [F, M], got {values.shape}"
                )
            self.face_fields[name] = values
        if not np.isfinite(self.positions).all():
            raise ValueError("non-finite positions")

    # ----- sizes -----
    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.positions.shape[1])

    @property
    def num_faces(self) -> int:
        return int(len(self.face_counts))

    @property
    def part_ids(self) -> list[int]:
        return sorted(int(p) for p in np.unique(self.face_part))

    def bounds(self) -> np.ndarray:
        """``[2, 3]`` min/max over every frame."""
        return np.stack(
            (self.positions.min(axis=(0, 1)), self.positions.max(axis=(0, 1)))
        )

    # ----- derived -----
    def select_frames(self, frames: list[int]) -> "CrashScene":
        """A scene holding only ``frames`` (renumbered 0..F'-1 for the demo)."""
        frames = list(frames)
        return CrashScene(
            name=self.name,
            positions=self.positions[frames],
            face_counts=self.face_counts,
            face_indices=self.face_indices,
            node_part=self.node_part,
            face_part=self.face_part,
            frames=list(range(len(frames))),
            time_ms=[self.time_ms[f] for f in frames],
            node_fields={k: v[frames] for k, v in self.node_fields.items()},
            face_fields={k: v[frames] for k, v in self.face_fields.items()},
            part_names=dict(self.part_names),
            units=self.units,
            meta={**self.meta, "source_frames": [int(self.frames[f]) for f in frames]},
        )

    def transformed(
        self, transform: str | np.ndarray, translation=(0.0, 0.0, 0.0)
    ) -> "CrashScene":
        """Positions mapped through ``x' = R x + t`` (``transform`` a name from
        :data:`TRANSFORMS` or a ``[3, 3]`` matrix); fields are unchanged."""
        matrix = (
            TRANSFORMS[transform]
            if isinstance(transform, str)
            else np.asarray(transform, dtype=np.float64)
        )
        if matrix.shape != (3, 3):
            raise ValueError("transform must be a 3x3 matrix")
        t = np.asarray(translation, dtype=np.float64).reshape(1, 1, 3)
        positions = (self.positions.astype(np.float64) @ matrix.T + t).astype(
            np.float32
        )
        meta = {
            **self.meta,
            "transform": transform if isinstance(transform, str) else matrix.tolist(),
            "translation": [float(v) for v in np.asarray(translation).reshape(-1)],
        }
        return CrashScene(
            name=self.name,
            positions=positions,
            face_counts=self.face_counts,
            face_indices=self.face_indices,
            node_part=self.node_part,
            face_part=self.face_part,
            frames=list(self.frames),
            time_ms=list(self.time_ms),
            node_fields=dict(self.node_fields),
            face_fields=dict(self.face_fields),
            part_names=dict(self.part_names),
            units=self.units,
            meta=meta,
        )

    def with_fields(self, node_fields=None, face_fields=None) -> "CrashScene":
        return CrashScene(
            name=self.name,
            positions=self.positions,
            face_counts=self.face_counts,
            face_indices=self.face_indices,
            node_part=self.node_part,
            face_part=self.face_part,
            frames=list(self.frames),
            time_ms=list(self.time_ms),
            node_fields={**self.node_fields, **(node_fields or {})},
            face_fields={**self.face_fields, **(face_fields or {})},
            part_names=dict(self.part_names),
            units=self.units,
            meta=dict(self.meta),
        )


def face_part_from_nodes(face_counts, face_indices, node_part) -> np.ndarray:
    """Part id per face: the part of its first node (shells of one part share
    their nodes' part id in LS-DYNA exports; a weld node may belong to the
    neighbouring part, which only moves that face's colour group)."""
    offsets = face_offsets(face_counts)
    return np.asarray(node_part, dtype=np.int32)[np.asarray(face_indices)[offsets[:-1]]]


def connected_components(num_nodes: int, face_counts, face_indices) -> np.ndarray:
    """Component label per node from face adjacency (pointer-jumping union-find)."""
    offsets = face_offsets(face_counts)
    indices = np.asarray(face_indices, dtype=np.int64)
    first = np.repeat(indices[offsets[:-1]], np.asarray(face_counts, dtype=np.int64))
    parent = np.arange(num_nodes, dtype=np.int64)
    edges_a, edges_b = first, indices
    for _ in range(64):
        ra, rb = parent[edges_a], parent[edges_b]
        lo = np.minimum(ra, rb)
        changed = False
        for src, dst in ((ra, lo), (rb, lo)):
            mask = parent[src] > dst
            if mask.any():
                np.minimum.at(parent, src[mask], dst[mask])
                changed = True
        # pointer jumping
        for _ in range(8):
            grand = parent[parent]
            if np.array_equal(grand, parent):
                break
            parent = grand
        if not changed:
            break
    _, labels = np.unique(parent, return_inverse=True)
    return labels.astype(np.int32)


@dataclass
class PartRecord:
    part_id: int
    name: str
    prim_name: str
    node_ids: np.ndarray  # global node ids referenced by this part's faces (sorted)
    face_ids: np.ndarray  # global face ids of this part (scene order)
    point_offset: int
    face_offset: int
    face_vertex_offset: int

    @property
    def point_count(self) -> int:
        return int(len(self.node_ids))

    @property
    def face_count(self) -> int:
        return int(len(self.face_ids))


@dataclass
class MergedLayout:
    """Per-part prims and their concatenation (the demo's merged transient mesh)."""

    parts: list[PartRecord]
    merged_source_nodes: np.ndarray  # [P] global node id of every merged point
    face_order: np.ndarray  # [M] global face id in merged (part) order
    face_counts: np.ndarray  # [M] int32 in merged order
    face_vertex_indices: np.ndarray  # [FV] int32 merged point ids
    local_face_indices: list[
        np.ndarray
    ]  # per part: face-vertex indices into the part's points

    @property
    def point_count(self) -> int:
        return int(len(self.merged_source_nodes))

    @property
    def face_count(self) -> int:
        return int(len(self.face_counts))

    @property
    def face_vertex_count(self) -> int:
        return int(len(self.face_vertex_indices))

    @classmethod
    def build(cls, scene: CrashScene, prim_prefix: str = "Part_") -> "MergedLayout":
        order = np.argsort(scene.face_part, kind="stable")
        offsets = face_offsets(scene.face_counts)
        parts: list[PartRecord] = []
        merged_nodes: list[np.ndarray] = []
        local_faces: list[np.ndarray] = []
        point_offset = face_offset = face_vertex_offset = 0
        sorted_parts = scene.face_part[order]
        boundaries = np.flatnonzero(np.diff(sorted_parts)) + 1
        for group in np.split(order, boundaries):
            pid = int(scene.face_part[group[0]])
            lengths = scene.face_counts[group].astype(np.int64)
            total = int(lengths.sum())
            # corner ids of the group's faces: offsets[f] + 0..counts[f]-1, vectorised
            corner_ids = (
                np.arange(total, dtype=np.int64)
                - np.repeat(np.cumsum(lengths) - lengths, lengths)
                + np.repeat(offsets[group], lengths)
            )
            global_nodes = scene.face_indices[corner_ids].astype(np.int64)
            node_ids = np.unique(global_nodes)
            local = np.searchsorted(node_ids, global_nodes).astype(np.int32)
            name = scene.part_names.get(pid, f"{prim_prefix}{pid}")
            parts.append(
                PartRecord(
                    part_id=pid,
                    name=name,
                    prim_name=_prim_name(name, pid, prim_prefix),
                    node_ids=node_ids,
                    face_ids=group.astype(np.int64),
                    point_offset=point_offset,
                    face_offset=face_offset,
                    face_vertex_offset=face_vertex_offset,
                )
            )
            merged_nodes.append(node_ids)
            local_faces.append(local)
            point_offset += len(node_ids)
            face_offset += len(group)
            face_vertex_offset += len(local)
        merged_source = (
            np.concatenate(merged_nodes)
            if merged_nodes
            else np.empty(0, dtype=np.int64)
        )
        face_vertex = (
            np.concatenate(
                [lf + p.point_offset for lf, p in zip(local_faces, parts)]
            ).astype(np.int32)
            if parts
            else np.empty(0, dtype=np.int32)
        )
        return cls(
            parts=parts,
            merged_source_nodes=merged_source,
            face_order=order.astype(np.int64),
            face_counts=scene.face_counts[order],
            face_vertex_indices=face_vertex,
            local_face_indices=local_faces,
        )

    def gather_nodes(self, values: np.ndarray) -> np.ndarray:
        """Node values ``[..., N]`` -> merged point values ``[..., P]``."""
        return np.ascontiguousarray(np.asarray(values)[..., self.merged_source_nodes])

    def gather_positions(self, positions: np.ndarray) -> np.ndarray:
        """Node positions ``[F, N, 3]`` -> merged point positions ``[F, P, 3]``."""
        return np.ascontiguousarray(np.asarray(positions)[:, self.merged_source_nodes])

    def gather_faces(self, values: np.ndarray) -> np.ndarray:
        """Face values ``[..., M]`` (scene order) -> merged face order."""
        return np.ascontiguousarray(np.asarray(values)[..., self.face_order])

    def part_face_counts(self, part: PartRecord) -> np.ndarray:
        return self.face_counts[part.face_offset : part.face_offset + part.face_count]

    def summary(self) -> dict:
        return {
            "mesh_count": len(self.parts),
            "point_count": self.point_count,
            "face_count": self.face_count,
            "face_vertex_count": self.face_vertex_count,
            "triangle_face_count": int((self.face_counts == 3).sum()),
            "quadrilateral_face_count": int((self.face_counts == 4).sum()),
        }


def _prim_name(name: str, pid: int, prefix: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{prefix}{cleaned}"
    return cleaned if cleaned != prefix else f"{prefix}{pid}"


# ----- sources -----------------------------------------------------------------


def load_yaris_dump(
    case_dir, label: str, *, frame_spec: str | None = None, include_solids: bool = True
) -> tuple[CrashScene, dict]:
    """A scene from ``dump_yaris_predictions.py`` output: ``label`` is
    ``ground_truth``, ``teacher``, ``corrector``... Node fields: ``error_mm``
    (against the ground truth; zero for the ground truth), ``displacement_mm``
    (from the reference), ``penetration_envelope_mm``, ``penetration_crossing_mm``,
    ``wall_crossing_mm`` (the dump's depths for this trajectory) and, for a
    model, ``truth_penetration_crossing_mm``. Face field
    ``penetration_crossing_face_mm`` is the maximum of the crossing depth over
    the face's nodes. Returns the scene and the dump's metadata.
    """
    from .fields import displacement_magnitude, error_magnitude, face_max

    case_dir = Path(case_dir)
    mesh = np.load(case_dir / "mesh.npz")
    traj_dir = case_dir / label
    meta = json.loads((traj_dir / "meta.json").read_text())
    positions = np.load(traj_dir / "positions.npy", mmap_mode="r")
    truth = np.load(case_dir / "ground_truth" / "positions.npy", mmap_mode="r")
    available = int(positions.shape[0])
    frames = parse_frame_spec(frame_spec, available)
    positions = np.asarray(positions[frames], dtype=np.float32)
    truth = np.asarray(truth[frames], dtype=np.float32)
    node_part = np.asarray(mesh["part_id"], dtype=np.int32)
    counts, indices, rows = faces_from_padded(mesh["shell_faces"])
    face_part = np.asarray(mesh["shell_part_id"], dtype=np.int32)[rows]
    solid_faces = 0
    if include_solids and "solid_hexes" in mesh and mesh["solid_hexes"].size:
        boundary = hex_boundary_faces(mesh["solid_hexes"])
        if len(boundary):
            s_counts, s_indices, _ = faces_from_padded(boundary)
            s_part = face_part_from_nodes(s_counts, s_indices, node_part)
            counts = np.concatenate((counts, s_counts))
            indices = np.concatenate((indices, s_indices))
            face_part = np.concatenate((face_part, s_part))
            solid_faces = int(len(s_counts))

    def depth(name):
        path = traj_dir / f"{name}.npy"
        return (
            np.asarray(np.load(path, mmap_mode="r")[frames], dtype=np.float32)
            if path.exists()
            else None
        )

    node_fields = {
        "displacement_mm": displacement_magnitude(positions),
        "error_mm": error_magnitude(positions, truth),
    }
    for key, name in (
        ("depth_envelope", "penetration_envelope_mm"),
        ("depth_crossing", "penetration_crossing_mm"),
        ("wall_crossing", "wall_crossing_mm"),
    ):
        values = depth(key)
        if values is not None:
            node_fields[name] = values
    truth_depth = case_dir / "ground_truth" / "depth_crossing.npy"
    if label != "ground_truth" and truth_depth.exists():
        node_fields["truth_penetration_crossing_mm"] = np.asarray(
            np.load(truth_depth, mmap_mode="r")[frames], dtype=np.float32
        )
    face_fields = {}
    if "penetration_crossing_mm" in node_fields:
        face_fields["penetration_crossing_face_mm"] = face_max(
            node_fields["penetration_crossing_mm"], counts, indices
        )
    time_ms = (
        [float(meta["time_ms"][f]) for f in frames]
        if "time_ms" in meta
        else [float(f) for f in frames]
    )
    scene = CrashScene(
        name=f"{meta.get('case', case_dir.name)}__{label}",
        positions=positions,
        face_counts=counts,
        face_indices=indices,
        node_part=node_part,
        face_part=face_part,
        frames=list(range(len(frames))),
        time_ms=time_ms,
        node_fields=node_fields,
        face_fields=face_fields,
        units="mm",
        meta={
            "source": "yaris_dump",
            "case": meta.get("case", case_dir.name),
            "label": label,
            "dump_meta": meta,
            "source_frames": [
                int(meta["frames"][f]) if "frames" in meta else f for f in frames
            ],
            "shell_faces_dropped_degenerate": int(len(mesh["shell_faces"]) - len(rows)),
            "solid_boundary_faces": solid_faces,
            "kept_nodes": int(positions.shape[1]),
        },
    )
    return scene, meta


def find_vtp_frames(directory, kind: str) -> dict[int, Path]:
    """``{frame: path}`` of ``<prefix>_<frame>_<kind>.vtp`` files in ``directory``."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"no such directory: {directory}")
    found = {}
    for path in directory.iterdir():
        match = _FRAME_FILE.match(path.name)
        if match and match.group("kind") == kind:
            found[int(match.group("frame"))] = path
    return dict(sorted(found.items()))


def load_vtp_sequence(
    pred_dir,
    exact_dir=None,
    *,
    frame_spec: str | None = None,
    parts: str = "single",
    name: str | None = None,
    dt_ms: float = 5.0,
) -> CrashScene:
    """A scene from the VTP frame sequences of ``inference.save_vtp_sequence``
    (``frame_XXX_pred.vtp`` with the predicted positions as points; optionally
    the matching ``frame_XXX_exact.vtp`` for the error field). ``parts`` is
    ``"single"`` (one part) or ``"components"`` (connected components as parts).
    Extra predicted point fields (``pred_<name>``) become node fields. The
    frame number in the file name counts frames of ``dt_ms`` milliseconds from
    the initial state (frame 0), so ``time_ms = frame * dt_ms``.
    """
    import pyvista as pv

    from .fields import displacement_magnitude, error_magnitude

    pred_files = find_vtp_frames(pred_dir, "pred")
    if not pred_files:
        raise FileNotFoundError(f"no *_pred.vtp frames under {pred_dir}")
    exact_files = {}
    if exact_dir is not None:
        exact_files = find_vtp_frames(exact_dir, "exact")
        if not exact_files:
            raise FileNotFoundError(f"no *_exact.vtp frames under {exact_dir}")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    available = list(pred_files)
    selected = [available[i] for i in parse_frame_spec(frame_spec, len(available))]
    first = pv.read(pred_files[selected[0]])
    counts, indices, _ = faces_from_pyvista(first)
    positions = np.empty((len(selected), first.n_points, 3), dtype=np.float32)
    truth = np.empty_like(positions) if exact_files else None
    extra: dict[str, list[np.ndarray]] = {}
    for k, frame in enumerate(selected):
        mesh = pv.read(pred_files[frame]) if k else first
        if mesh.n_points != first.n_points:
            raise ValueError(f"point count changes at frame {frame}")
        positions[k] = np.asarray(mesh.points, dtype=np.float32)
        for key in mesh.point_data.keys():
            if key.startswith("pred_"):
                values = np.asarray(mesh.point_data[key], dtype=np.float32)
                if values.ndim == 1:
                    extra.setdefault(key[5:], []).append(values)
        if truth is not None:
            if frame not in exact_files:
                raise FileNotFoundError(
                    f"missing exact frame {frame} under {exact_dir}"
                )
            exact = pv.read(exact_files[frame])
            if exact.n_points != first.n_points:
                raise ValueError(f"exact point count differs at frame {frame}")
            truth[k] = np.asarray(exact.points, dtype=np.float32)
    n = first.n_points
    if parts == "components":
        node_part = connected_components(n, counts, indices)
    elif parts == "single":
        node_part = np.zeros(n, dtype=np.int32)
    else:
        raise ValueError("parts must be 'single' or 'components'")
    for key in ("part_id", "PartID", "part"):
        if key in first.point_data:
            node_part = np.asarray(first.point_data[key]).astype(np.int32)
            break
    face_part = face_part_from_nodes(counts, indices, node_part)
    for key in ("part_id", "PartID", "part"):
        if key in first.cell_data:
            face_part = np.asarray(first.cell_data[key]).astype(np.int32)
            break
    node_fields = {"displacement_mm": displacement_magnitude(positions)}
    if truth is not None:
        node_fields["error_mm"] = error_magnitude(positions, truth)
    for key, values in extra.items():
        if len(values) == len(selected):
            node_fields[key] = np.stack(values)
    return CrashScene(
        name=name or Path(pred_dir).name,
        positions=positions,
        face_counts=counts,
        face_indices=indices,
        node_part=node_part,
        face_part=face_part,
        frames=list(range(len(selected))),
        time_ms=[float(f) * float(dt_ms) for f in selected],
        node_fields=node_fields,
        units="mm",
        meta={
            "source": "vtp_sequence",
            "pred_dir": str(pred_dir),
            "exact_dir": str(exact_dir) if exact_dir is not None else None,
            "dt_ms": float(dt_ms),
            "source_frames": [int(f) for f in selected],
        },
    )


def faces_from_pyvista(poly) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face counts/indices of a PyVista PolyData (triangles and quads only)."""
    faces = np.asarray(poly.faces, dtype=np.int64)
    counts, indices, rows = [], [], []
    i = row = 0
    while i < faces.size:
        size = int(faces[i])
        ids = faces[i + 1 : i + 1 + size]
        if size in (3, 4) and len(np.unique(ids)) == size:
            counts.append(size)
            indices.append(ids)
            rows.append(row)
        i += 1 + size
        row += 1
    if not counts:
        raise ValueError("no triangle or quadrilateral faces")
    return (
        np.asarray(counts, dtype=np.int32),
        np.concatenate(indices).astype(np.int32),
        np.asarray(rows, dtype=np.int64),
    )


def load_biw_vtp(
    path, *, frame_spec: str | None = None, parts: str = "components"
) -> CrashScene:
    """The ground truth of a body-in-white crash VTP (``displacement_t0.000`` ... point
    data, as read by ``vtp_reader.load_vtp_file``) as a scene."""
    import pyvista as pv

    from .fields import displacement_magnitude

    poly = pv.read(str(path))
    names = sorted(
        (k for k in poly.point_data.keys() if k.startswith("displacement_t")),
        key=lambda s: float(s[len("displacement_t") :]),
    )
    if not names:
        raise ValueError(f"no displacement_t fields in {path}")
    frames = parse_frame_spec(frame_spec, len(names))
    coords = np.asarray(poly.points, dtype=np.float32)
    positions = np.stack(
        [
            coords
            if (k == 0)
            else coords + np.asarray(poly.point_data[names[k]], dtype=np.float32)
            for k in frames
        ]
    )
    counts, indices, rows = faces_from_pyvista(poly)
    n = poly.n_points
    node_part = (
        connected_components(n, counts, indices)
        if parts == "components"
        else np.zeros(n, dtype=np.int32)
    )
    face_part = face_part_from_nodes(counts, indices, node_part)
    thickness = (
        {
            "thickness_mm": np.repeat(
                np.asarray(poly.point_data["thickness"], dtype=np.float32)[None],
                len(frames),
                axis=0,
            )
        }
        if "thickness" in poly.point_data
        else {}
    )
    return CrashScene(
        name=Path(path).stem,
        positions=positions,
        face_counts=counts,
        face_indices=indices,
        node_part=node_part,
        face_part=face_part,
        frames=list(range(len(frames))),
        time_ms=[float(names[k][len("displacement_t") :]) * 1000.0 for k in frames],
        node_fields={"displacement_mm": displacement_magnitude(positions), **thickness},
        units="mm",
        meta={
            "source": "biw_vtp",
            "path": str(path),
            "source_frames": [int(f) for f in frames],
        },
    )


def synthetic_scene(
    num_frames: int = 4, seed: int = 0, parts: int = 3, grid: int = 4
) -> CrashScene:
    """A small deterministic scene for tests: ``parts`` plates of ``grid`` x
    ``grid`` quads (the last plate triangulated), bending over time, with node
    and face fields that are simple functions of position and frame."""
    rng = np.random.default_rng(seed)
    plates_pos, counts_all, indices_all, node_part, face_part = [], [], [], [], []
    offset = 0
    for p in range(parts):
        xs, ys = np.meshgrid(
            np.arange(grid + 1, dtype=np.float32) * 10.0,
            np.arange(grid + 1, dtype=np.float32) * 10.0,
        )
        pts = np.stack(
            (xs.ravel() + 60.0 * p, ys.ravel(), np.zeros(xs.size, dtype=np.float32)),
            axis=1,
        )
        node = lambda c, r: r * (grid + 1) + c + offset  # noqa: E731
        for r in range(grid):
            for c in range(grid):
                quad = [node(c, r), node(c + 1, r), node(c + 1, r + 1), node(c, r + 1)]
                if p == parts - 1:
                    counts_all += [3, 3]
                    indices_all += quad[:3] + [quad[0], quad[2], quad[3]]
                    face_part += [p, p]
                else:
                    counts_all.append(4)
                    indices_all += quad
                    face_part.append(p)
        plates_pos.append(pts)
        node_part += [p] * len(pts)
        offset += len(pts)
    reference = np.concatenate(plates_pos)
    positions = np.stack(
        [
            reference
            + np.stack(
                (
                    np.zeros(len(reference)),
                    np.zeros(len(reference)),
                    0.5 * f * np.sin(reference[:, 0] / 20.0),
                ),
                axis=1,
            ).astype(np.float32)
            for f in range(num_frames)
        ]
    )
    positions += rng.normal(scale=1e-3, size=positions.shape).astype(np.float32)
    frames = np.arange(num_frames, dtype=np.float32)[:, None]
    node_fields = {
        "error_mm": (0.1 * frames * (1 + np.arange(len(reference)) % 7)).astype(
            np.float32
        ),
        "displacement_mm": np.linalg.norm(positions - positions[:1], axis=-1).astype(
            np.float32
        ),
    }
    counts = np.asarray(counts_all, dtype=np.int32)
    indices = np.asarray(indices_all, dtype=np.int32)
    face_fields = {
        "penetration_crossing_face_mm": (
            frames * (np.arange(len(counts)) % 5) * 0.25
        ).astype(np.float32)
    }
    return CrashScene(
        name="synthetic",
        positions=positions,
        face_counts=counts,
        face_indices=indices,
        node_part=np.asarray(node_part, dtype=np.int32),
        face_part=np.asarray(face_part, dtype=np.int32),
        frames=list(range(num_frames)),
        time_ms=[5.0 * f for f in range(num_frames)],
        node_fields=node_fields,
        face_fields=face_fields,
        part_names={0: "Plate_A"},
        units="mm",
        meta={"source": "synthetic"},
    )
