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
"""Derived per-node and per-face fields and their assignment to the demo's
three physical-field slots.

The viewer hard-codes three scalar display modes (see ``scene.py``): a
per-face 16-bit atlas labelled "Von Mises Stress" and two per-vertex UDIM
atlases labelled "U Displacement Magnitude" and "V Velocity Magnitude". The
exporter fills those slots with DeFormer quantities; the labels in the demo's
web UI are not data-driven (see the README's known limitations), so the mapping
is written to ``deformer_scene.json`` next to the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scene import FACE_SLOT, U_SLOT, V_SLOT, CrashScene, face_offsets

# Labels and suggested viewer colour ranges of the fields the scene loaders
# produce (millimetres throughout). A field not listed here is labelled by its
# name and gets a range derived from its values (:func:`field_range`).
FIELD_LABELS = {
    "penetration_crossing_face_mm": "Shell crossing depth per face [mm]",
    "penetration_crossing_mm": "Shell crossing depth per node [mm]",
    "truth_penetration_crossing_mm": "Ground-truth shell crossing depth per node [mm]",
    "penetration_envelope_mm": "Shell penetration into the thickness envelope [mm]",
    "wall_crossing_mm": "Rigid-wall crossing depth [mm]",
    "error_mm": "Position error vs. ground truth [mm]",
    "displacement_mm": "Displacement magnitude [mm]",
    "thickness_mm": "Shell thickness [mm]",
}
FIELD_RANGES = {
    "penetration_crossing_face_mm": (0.0, 5.0),
    "penetration_crossing_mm": (0.0, 5.0),
    "truth_penetration_crossing_mm": (0.0, 5.0),
    "penetration_envelope_mm": (0.0, 5.0),
    "wall_crossing_mm": (0.0, 5.0),
    "error_mm": (0.0, 30.0),
    "displacement_mm": (0.0, 1000.0),
    "thickness_mm": (0.0, 5.0),
}


def field_label(name: str, per_face_reduce: str | None = None) -> str:
    """The display label of scene field ``name``; ``per_face_reduce`` names the
    reduction when a node field fills the per-face slot."""
    label = FIELD_LABELS.get(name, name)
    if per_face_reduce:
        label = f"{label}, {per_face_reduce} over the face's nodes"
    return label


def field_range(name: str, scene: "CrashScene") -> tuple[float, float]:
    """Suggested colour range of scene field ``name``: the documented default
    for a known field, else ``[min(0, min), max]`` of its finite values."""
    if name in FIELD_RANGES:
        return FIELD_RANGES[name]
    values = scene.face_fields.get(name)
    if values is None:
        values = scene.node_fields.get(name)
    if values is None:
        raise KeyError(f"field {name!r} not in the scene")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = min(float(finite.min()), 0.0), float(finite.max())
    return (lo, hi if hi > lo else lo + 1.0)


def error_magnitude(positions: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """``|x - x_true|`` per frame and node, ``[F, N]``."""
    positions = np.asarray(positions, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    if positions.shape != truth.shape:
        raise ValueError("positions and truth must have the same shape")
    return np.linalg.norm(positions.astype(np.float64) - truth, axis=-1).astype(
        np.float32
    )


def displacement_magnitude(
    positions: np.ndarray, reference: np.ndarray | None = None
) -> np.ndarray:
    """``|x_t - x_ref|`` per frame and node (reference: frame 0), ``[F, N]``."""
    positions = np.asarray(positions, dtype=np.float32)
    reference = (
        positions[0] if reference is None else np.asarray(reference, dtype=np.float32)
    )
    return np.linalg.norm(positions.astype(np.float64) - reference, axis=-1).astype(
        np.float32
    )


def face_reduce(
    values: np.ndarray, face_counts, face_indices, how: str = "max"
) -> np.ndarray:
    """Per-face reduction (``max`` or ``mean``) of per-node values ``[F, N]`` -> ``[F, M]``."""
    values = np.asarray(values, dtype=np.float32)
    counts = np.asarray(face_counts, dtype=np.int64)
    indices = np.asarray(face_indices, dtype=np.int64)
    offsets = face_offsets(counts)
    gathered = values[..., indices]  # [F, FV]
    out = np.full(
        (values.shape[0], len(counts)),
        -np.inf if how == "max" else 0.0,
        dtype=np.float32,
    )
    for k in range(4):
        rows = counts > k
        column = gathered[..., offsets[:-1][rows] + k]
        if how == "max":
            out[..., rows] = np.maximum(out[..., rows], column)
        elif how == "mean":
            out[..., rows] += column
        else:
            raise ValueError("how must be 'max' or 'mean'")
    if how == "mean":
        out /= counts.astype(np.float32)
    return out


def face_max(values, face_counts, face_indices) -> np.ndarray:
    return face_reduce(values, face_counts, face_indices, "max")


def face_mean(values, face_counts, face_indices) -> np.ndarray:
    return face_reduce(values, face_counts, face_indices, "mean")


@dataclass(frozen=True)
class SlotAssignment:
    """Which scene field feeds which demo slot (``None`` leaves a slot at zero).

    ``face`` names a face field (or a node field, reduced with ``face_reduce``
    = ``max``); ``u`` and ``v`` name node fields. ``labels`` and ``ranges``
    document the slots for the viewer's operator (suggested colour range).
    """

    face: str | None = "penetration_crossing_face_mm"
    u: str | None = "error_mm"
    v: str | None = "displacement_mm"
    face_reduce: str = "max"
    labels: dict = field(default_factory=dict)
    ranges: dict = field(default_factory=dict)

    @classmethod
    def for_label(cls, label: str) -> "SlotAssignment":
        """Defaults: models show error vs. the ground truth in U; the ground
        truth shows its own displacement in U and penetration in V."""
        if label == "ground_truth":
            return cls(
                face="penetration_crossing_face_mm",
                u="displacement_mm",
                v="penetration_crossing_mm",
                labels={
                    FACE_SLOT: FIELD_LABELS["penetration_crossing_face_mm"],
                    U_SLOT: FIELD_LABELS["displacement_mm"],
                    V_SLOT: FIELD_LABELS["penetration_crossing_mm"],
                },
                ranges={
                    FACE_SLOT: FIELD_RANGES["penetration_crossing_face_mm"],
                    U_SLOT: FIELD_RANGES["displacement_mm"],
                    V_SLOT: FIELD_RANGES["penetration_crossing_mm"],
                },
            )
        return cls(
            face="penetration_crossing_face_mm",
            u="error_mm",
            v="displacement_mm",
            labels={
                FACE_SLOT: "Spurious shell crossing depth per face [mm]",
                U_SLOT: FIELD_LABELS["error_mm"],
                V_SLOT: FIELD_LABELS["displacement_mm"],
            },
            ranges={
                FACE_SLOT: FIELD_RANGES["penetration_crossing_face_mm"],
                U_SLOT: FIELD_RANGES["error_mm"],
                V_SLOT: FIELD_RANGES["displacement_mm"],
            },
        )

    @classmethod
    def for_scene(
        cls, scene: "CrashScene", label: str, face=None, u=None, v=None
    ) -> "SlotAssignment":
        """The label's defaults, overridden by explicit names, and each slot
        falling back to a field the scene has (VTP scenes have no penetration
        fields, for instance); ``None`` leaves a slot at zero. A slot that ends
        up with another field than the label's default gets that field's label
        and suggested range (:data:`FIELD_LABELS`, :func:`field_range`)."""
        base = cls.for_label(label)
        face = base.face if face is None else face
        u = base.u if u is None else u
        v = base.v if v is None else v
        available = set(scene.face_fields) | set(scene.node_fields)
        if face is not None and face not in available:
            face = next(
                (
                    n
                    for n in (
                        "penetration_crossing_face_mm",
                        "error_mm",
                        "displacement_mm",
                    )
                    if n in available
                ),
                None,
            )
        if u is not None and u not in scene.node_fields:
            u = next(
                (n for n in ("error_mm", "displacement_mm") if n in scene.node_fields),
                None,
            )
        if v is not None and v not in scene.node_fields:
            v = next(
                (n for n in ("displacement_mm", "error_mm") if n in scene.node_fields),
                None,
            )
        labels, ranges = {}, {}
        for slot, name, default in (
            (FACE_SLOT, face, base.face),
            (U_SLOT, u, base.u),
            (V_SLOT, v, base.v),
        ):
            if name is None:
                continue
            if name == default:
                labels[slot], ranges[slot] = base.labels[slot], base.ranges[slot]
                continue
            reduced = slot == FACE_SLOT and name not in scene.face_fields
            labels[slot] = field_label(name, base.face_reduce if reduced else None)
            ranges[slot] = field_range(name, scene)
        return cls(
            face=face,
            u=u,
            v=v,
            face_reduce=base.face_reduce,
            labels=labels,
            ranges=ranges,
        )

    def as_dict(self) -> dict:
        return {
            "slots": {FACE_SLOT: self.face, U_SLOT: self.u, V_SLOT: self.v},
            "face_reduce": self.face_reduce,
            "labels": {k: v for k, v in self.labels.items()},
            "suggested_ranges": {k: list(v) for k, v in self.ranges.items()},
        }


def resolve_slots(
    scene: CrashScene, assignment: SlotAssignment
) -> dict[str, np.ndarray]:
    """``{slot: values}`` with the face slot ``[F, M]`` and vertex slots ``[F, N]``
    in scene order (zeros for unassigned slots; missing names raise)."""
    out: dict[str, np.ndarray] = {}
    if assignment.face is None:
        out[FACE_SLOT] = np.zeros((scene.num_frames, scene.num_faces), dtype=np.float32)
    elif assignment.face in scene.face_fields:
        out[FACE_SLOT] = scene.face_fields[assignment.face]
    elif assignment.face in scene.node_fields:
        out[FACE_SLOT] = face_reduce(
            scene.node_fields[assignment.face],
            scene.face_counts,
            scene.face_indices,
            assignment.face_reduce,
        )
    else:
        raise KeyError(
            f"face slot field {assignment.face!r} not in the scene "
            f"(face fields {sorted(scene.face_fields)}, node fields {sorted(scene.node_fields)})"
        )
    for slot, name in ((U_SLOT, assignment.u), (V_SLOT, assignment.v)):
        if name is None:
            out[slot] = np.zeros((scene.num_frames, scene.num_nodes), dtype=np.float32)
        elif name in scene.node_fields:
            out[slot] = scene.node_fields[name]
        else:
            raise KeyError(
                f"vertex slot field {name!r} not in the scene (node fields {sorted(scene.node_fields)})"
            )
    return out
