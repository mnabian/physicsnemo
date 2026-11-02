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
"""16-bit field atlases in the layout of the car-crash OVRTX demo.

The demo (``tools/generate_von_mises_atlas.py`` and
``tools/generate_vertex_field_atlas.py`` of the car-crash OVRTX demo)
stores physical fields as linear 16-bit grayscale PNG textures and samples them
in MDL from packed per-face texture coordinates:

* **per-face atlas** (the "Von Mises" slot): one texel per face, faces in
  row-major order inside a ``tile_side`` x ``tile_side`` tile, one tile per
  frame on a ``tile_grid`` x ``tile_grid`` grid, one PNG for every frame.
* **per-vertex UDIM atlas** (the "U" and "V" slots): a 2x2 texel cell per
  face holding the field at its corners (triangles store the bilinear
  extrapolation ``v1 + v2 - v0`` in the fourth texel so interpolation inside
  the face is linear), ``tile_cells`` cells per row, one tile per frame,
  ``bank_tile_grid**2`` frames per UDIM texture page.
* **packed texture coordinates** (``primvars:st``, face-varying): every corner
  encodes its face cell and its local (0/1) corner position, triangles offset
  by one tile height in v so the shader can tell them from quads; the same
  coordinates drive the wireframe overlay.

The numerics are a faithful port of the demo's generators so that the demo's
``tools/verify_*.py`` accept the output; the layout name is
``per-face-packed-local-uv-v3``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LAYOUT = "per-face-packed-local-uv-v3"
PACKED_LOCAL_MARGIN = 1.0 / 4096.0
UDIM_START = 1001
UDIM_COLUMNS = 10
UDIM_TOKEN = "<UDIM>"  # noqa: S105  (a file-name placeholder, not a secret)
ENCODING = "png-gray16-linear"
DEFAULT_MAX_TEXTURE_SIDE = 8192


def encode16(
    values: np.ndarray, encoding_min: float, encoding_max: float, fill: float = 0.0
) -> np.ndarray:
    """Linear 16-bit encoding of ``values`` over ``[encoding_min, encoding_max]``;
    non-finite values take ``fill`` (the vertex generator's convention)."""
    finite = np.where(np.isfinite(values), values, fill).astype(np.float64)
    normalized = np.clip(
        (finite - encoding_min) / (encoding_max - encoding_min), 0.0, 1.0
    )
    return np.rint(normalized * 65535.0).astype(np.uint16)


def encode16_face(values: np.ndarray, encoding_min: float, span: float) -> np.ndarray:
    """The per-face generator's encoding: NaN -> 0, +inf -> 1, -inf -> 0 after
    normalisation over ``span``."""
    normalized = np.nan_to_num(
        (values - encoding_min) / span, nan=0.0, posinf=1.0, neginf=0.0
    )
    return np.rint(np.clip(normalized, 0.0, 1.0) * 65535.0).astype(np.uint16)


def decode16(
    encoded: np.ndarray, encoding_min: float, encoding_max: float
) -> np.ndarray:
    span = float(encoding_max) - float(encoding_min)
    return float(encoding_min) + encoded.astype(np.float64) * (span / 65535.0)


# ----- per-face atlas ----------------------------------------------------------


@dataclass(frozen=True)
class FaceAtlasLayout:
    face_count: int
    frame_count: int
    tile_side: int
    tile_grid: int

    @classmethod
    def build(cls, face_count: int, frame_count: int) -> "FaceAtlasLayout":
        if face_count <= 0 or frame_count <= 0:
            raise ValueError("face_count and frame_count must be positive")
        return cls(
            face_count=int(face_count),
            frame_count=int(frame_count),
            tile_side=math.ceil(math.sqrt(face_count)),
            tile_grid=math.ceil(math.sqrt(frame_count)),
        )

    @property
    def atlas_side(self) -> int:
        return self.tile_side * self.tile_grid

    def tile_origin(self, frame_index: int) -> tuple[int, int]:
        """``(x, y)`` texel origin of a frame's tile."""
        return (frame_index % self.tile_grid) * self.tile_side, (
            frame_index // self.tile_grid
        ) * self.tile_side


def face_atlas_range(values: np.ndarray) -> tuple[float, float, int]:
    """Global finite min/max of ``[F, M]`` face values and the non-finite count."""
    finite = np.isfinite(values)
    nonfinite = int(values.size - finite.sum())
    if not finite.any():
        raise ValueError("unable to determine the field range: no finite values")
    return float(values[finite].min()), float(values[finite].max()), nonfinite


def encode_face_atlas(
    values: np.ndarray, layout: FaceAtlasLayout
) -> tuple[np.ndarray, float, float, int]:
    """The per-face atlas ``uint16 [side, side]`` of ``values [F, M]``."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (layout.frame_count, layout.face_count):
        raise ValueError(
            f"face values must be {(layout.frame_count, layout.face_count)}, got {values.shape}"
        )
    encoding_min, encoding_max, nonfinite = face_atlas_range(values)
    span = max(encoding_max - encoding_min, float(np.finfo(np.float32).eps))
    atlas = np.zeros((layout.atlas_side, layout.atlas_side), dtype=np.uint16)
    tile = np.zeros((layout.tile_side, layout.tile_side), dtype=np.uint16)
    for frame_index in range(layout.frame_count):
        tile.fill(0)
        tile.flat[: layout.face_count] = encode16_face(
            values[frame_index], encoding_min, span
        )
        x, y = layout.tile_origin(frame_index)
        atlas[y : y + layout.tile_side, x : x + layout.tile_side] = tile
    return atlas, encoding_min, encoding_max, nonfinite


def decode_face_atlas(
    atlas: np.ndarray,
    layout: FaceAtlasLayout,
    frame_index: int,
    encoding_min: float,
    encoding_max: float,
) -> np.ndarray:
    x, y = layout.tile_origin(frame_index)
    encoded = atlas[y : y + layout.tile_side, x : x + layout.tile_side].reshape(-1)[
        : layout.face_count
    ]
    return decode16(encoded, encoding_min, encoding_max)


def face_cell_coordinates(
    face_counts: np.ndarray, atlas_face_offset: int, tile_side: int
) -> np.ndarray:
    """Face-varying ``[FV, 3]`` texture coordinates pointing at every face's
    texel centre (the merged scene's ``primvars:st``; the demo's
    ``create_static_mesh``). Faces are numbered ``atlas_face_offset + i``."""
    counts = np.asarray(face_counts, dtype=np.int64)
    face_ids = np.arange(len(counts), dtype=np.int64) + int(atlas_face_offset)
    per_face = np.zeros((len(counts), 3), dtype=np.float32)
    per_face[:, 0] = (face_ids % tile_side + 0.5) / tile_side
    per_face[:, 1] = (face_ids // tile_side + 0.5) / tile_side
    return np.repeat(per_face, counts, axis=0)


def packed_coordinates(
    face_counts: np.ndarray,
    face_columns: np.ndarray,
    face_rows: np.ndarray,
    tile_side: int,
) -> np.ndarray:
    """Face-varying ``[FV, 3]`` packed local coordinates (the demo's
    ``build_packed_coordinates``): corner k of a face in cell (column, row) is
    at ``((column + u_k) / tile_side, (row + v_k + tri * tile_side) / (2 tile_side))``
    with ``u_k, v_k`` in ``{margin, 1 - margin}`` following the corner order
    (0,0), (1,0), (1,1), (0,1) for quads and (0,0), (1,0), (0,1) for triangles."""
    counts = np.asarray(face_counts, dtype=np.int64)
    if not np.all(np.isin(counts, (3, 4))):
        raise ValueError("only triangular and quadrilateral faces are supported")
    face_columns = np.asarray(face_columns, dtype=np.int64)
    face_rows = np.asarray(face_rows, dtype=np.int64)
    if face_columns.shape != (len(counts),) or face_rows.shape != (len(counts),):
        raise ValueError("face cell arrays do not match the face count")
    face_vertex_count = int(counts.sum())
    face_ids = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    offsets = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    local_indices = np.arange(face_vertex_count, dtype=np.int64) - np.repeat(
        offsets, counts
    )
    per_corner_counts = counts[face_ids]
    x_bit = (local_indices == 1) | ((local_indices == 2) & (per_corner_counts == 4))
    y_bit = (local_indices == 3) | (local_indices == 2)
    triangle = per_corner_counts == 3
    local_u = PACKED_LOCAL_MARGIN + (1.0 - 2.0 * PACKED_LOCAL_MARGIN) * x_bit
    local_v = PACKED_LOCAL_MARGIN + (1.0 - 2.0 * PACKED_LOCAL_MARGIN) * y_bit
    packed_rows = (
        face_rows[face_ids] + local_v + triangle.astype(np.float32) * tile_side
    )
    coordinates = np.zeros((face_vertex_count, 3), dtype=np.float32)
    coordinates[:, 0] = (face_columns[face_ids] + local_u) / tile_side
    coordinates[:, 1] = packed_rows / (2.0 * tile_side)
    return coordinates


def packed_face_coordinates(
    face_counts: np.ndarray, atlas_face_offset: int, tile_side: int
) -> np.ndarray:
    """Packed coordinates of faces numbered ``atlas_face_offset + i`` in the
    per-face atlas (``build_wireframe_coordinates`` of the demo)."""
    face_ids = np.arange(len(face_counts), dtype=np.int64) + int(atlas_face_offset)
    return packed_coordinates(
        face_counts, face_ids % tile_side, face_ids // tile_side, tile_side
    )


def packed_from_cell_coordinates(
    face_counts: np.ndarray, cell_coordinates: np.ndarray, tile_side: int
) -> np.ndarray:
    """Packed coordinates derived from a merged scene's texel-centre ``st``
    (``build_merged_wireframe_coordinates`` of the demo)."""
    counts = np.asarray(face_counts, dtype=np.int64)
    offsets = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    face_uv = np.asarray(cell_coordinates, dtype=np.float32)[offsets, :2]
    cells = np.floor(face_uv * tile_side).astype(np.int64)
    if np.any(cells < 0) or np.any(cells >= tile_side):
        raise ValueError("atlas coordinates outside the expected tile")
    return packed_coordinates(counts, cells[:, 0], cells[:, 1], tile_side)


# ----- per-vertex UDIM atlas --------------------------------------------------


@dataclass(frozen=True)
class VertexAtlasLayout:
    face_count: int
    frame_count: int
    tile_cells: int
    bank_tile_grid: int
    max_texture_side: int = DEFAULT_MAX_TEXTURE_SIDE

    @classmethod
    def build(
        cls,
        face_count: int,
        frame_count: int,
        max_texture_side: int = DEFAULT_MAX_TEXTURE_SIDE,
    ) -> "VertexAtlasLayout":
        if face_count <= 0 or frame_count <= 0:
            raise ValueError("face_count and frame_count must be positive")
        if max_texture_side <= 0:
            raise ValueError("max_texture_side must be positive")
        tile_cells = math.ceil(math.sqrt(face_count))
        tile_side = tile_cells * 2
        bank_tile_grid = max_texture_side // tile_side
        if bank_tile_grid <= 0:
            raise ValueError(
                f"a frame tile needs {tile_side}x{tile_side} texels, more than the "
                f"{max_texture_side} texture limit"
            )
        return cls(
            int(face_count),
            int(frame_count),
            tile_cells,
            bank_tile_grid,
            int(max_texture_side),
        )

    @property
    def tile_side(self) -> int:
        return self.tile_cells * 2

    @property
    def frames_per_bank(self) -> int:
        return self.bank_tile_grid**2

    @property
    def bank_count(self) -> int:
        return math.ceil(self.frame_count / self.frames_per_bank)

    @property
    def bank_side(self) -> int:
        return self.tile_side * self.bank_tile_grid

    def locate(self, frame_index: int) -> tuple[int, int, int]:
        """``(bank, x, y)`` of a frame's tile."""
        bank = frame_index // self.frames_per_bank
        tile = frame_index - bank * self.frames_per_bank
        return (
            bank,
            (tile % self.bank_tile_grid) * self.tile_side,
            (tile // self.bank_tile_grid) * self.tile_side,
        )


def face_corner_indices(face_counts: np.ndarray, face_vertex_indices: np.ndarray):
    """``(i0, i1, i2, i3, quad_mask)`` point indices of every face's corners
    (``i3 = i0`` for triangles)."""
    counts = np.asarray(face_counts, dtype=np.int64)
    indices = np.asarray(face_vertex_indices, dtype=np.int64)
    if not np.all(np.isin(counts, (3, 4))):
        raise ValueError("only triangular and quadrilateral faces are supported")
    offsets = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    if len(counts) and int(offsets[-1] + counts[-1]) != len(indices):
        raise ValueError("face topology does not match the face-vertex indices")
    i0 = indices[offsets].astype(np.int32)
    i1 = indices[offsets + 1].astype(np.int32)
    i2 = indices[offsets + 2].astype(np.int32)
    i3 = i0.copy()
    quad = counts == 4
    i3[quad] = indices[offsets[quad] + 3]
    return i0, i1, i2, i3, quad


def vertex_corner_values(frame_values: np.ndarray, corners) -> np.ndarray:
    """``[M, 4]`` texel values (v0, v1, top-right, top-left) of one frame."""
    i0, i1, i2, i3, quad = corners
    v0, v1, v2, v3 = (frame_values[i] for i in (i0, i1, i2, i3))
    top_right = np.where(quad, v2, v1 + v2 - v0)
    top_left = np.where(quad, v3, v2)
    return np.stack((v0, v1, top_right, top_left), axis=1)


def vertex_atlas_range(values: np.ndarray, corners) -> tuple[float, float, int]:
    """Global encoding range over every frame's point values and the triangles'
    extrapolated texel (``scan_encoding_range`` of the demo)."""
    i0, i1, i2, _, quad = corners
    tri = ~quad
    minimum, maximum, nonfinite = math.inf, -math.inf, 0
    for frame_values in values:
        finite = np.isfinite(frame_values)
        nonfinite += int(finite.size - np.count_nonzero(finite))
        if not finite.any():
            continue
        finite_values = frame_values[finite]
        minimum = min(minimum, float(finite_values.min()))
        maximum = max(maximum, float(finite_values.max()))
        extra = frame_values[i1[tri]] + frame_values[i2[tri]] - frame_values[i0[tri]]
        extra = extra[np.isfinite(extra)]
        if extra.size:
            minimum = min(minimum, float(extra.min()))
            maximum = max(maximum, float(extra.max()))
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("unable to determine a valid atlas encoding range")
    if minimum >= maximum:
        # A constant field still needs a non-empty range (the demo refuses it).
        maximum = minimum + 1.0
    return minimum, maximum, nonfinite


def encode_vertex_atlas_bank(
    values: np.ndarray,
    corners,
    layout: VertexAtlasLayout,
    bank_index: int,
    encoding_min: float,
    encoding_max: float,
    fill: float = 0.0,
) -> np.ndarray:
    """One UDIM texture page ``uint16 [bank_side, bank_side]`` of ``values [F, P]``."""
    face_ids = np.arange(layout.face_count, dtype=np.int64)
    rows = np.asarray(2 * (face_ids // layout.tile_cells), dtype=np.intp)
    columns = np.asarray(2 * (face_ids % layout.tile_cells), dtype=np.intp)
    atlas = np.zeros((layout.bank_side, layout.bank_side), dtype=np.uint16)
    tile = np.zeros((layout.tile_side, layout.tile_side), dtype=np.uint16)
    first = bank_index * layout.frames_per_bank
    last = min(first + layout.frames_per_bank, layout.frame_count)
    for frame_index in range(first, last):
        tile.fill(0)
        texels = vertex_corner_values(
            np.asarray(values[frame_index], dtype=np.float32), corners
        )
        encoded = encode16(texels, encoding_min, encoding_max, fill)
        tile[rows, columns] = encoded[:, 0]
        tile[rows, columns + 1] = encoded[:, 1]
        tile[rows + 1, columns + 1] = encoded[:, 2]
        tile[rows + 1, columns] = encoded[:, 3]
        _, x, y = layout.locate(frame_index)
        atlas[y : y + layout.tile_side, x : x + layout.tile_side] = tile
    return atlas


def decode_vertex_atlas(
    atlas: np.ndarray,
    layout: VertexAtlasLayout,
    frame_index: int,
    encoding_min: float,
    encoding_max: float,
    face_ids=None,
) -> np.ndarray:
    """``[M', 4]`` decoded texel values of ``face_ids`` (all faces by default)."""
    face_ids = (
        np.arange(layout.face_count)
        if face_ids is None
        else np.asarray(face_ids, dtype=np.int64)
    )
    _, x, y = layout.locate(frame_index)
    rows = 2 * (face_ids // layout.tile_cells)
    columns = 2 * (face_ids % layout.tile_cells)
    encoded = np.stack(
        (
            atlas[y + rows, x + columns],
            atlas[y + rows, x + columns + 1],
            atlas[y + rows + 1, x + columns + 1],
            atlas[y + rows + 1, x + columns],
        ),
        axis=1,
    )
    return decode16(encoded, encoding_min, encoding_max)


def vertex_texture_coordinates(face_counts: np.ndarray, tile_cells: int) -> np.ndarray:
    """Face-varying packed coordinates of the vertex atlas schema (the demo's
    ``build_texture_coordinates``: cells of ``tile_cells`` per row)."""
    face_ids = np.arange(len(face_counts), dtype=np.int64)
    return packed_coordinates(
        face_counts, face_ids % tile_cells, face_ids // tile_cells, tile_cells
    )


def udim_number(bank_index: int) -> int:
    return (
        UDIM_START
        + (bank_index % UDIM_COLUMNS)
        + (bank_index // UDIM_COLUMNS) * UDIM_COLUMNS
    )


def udim_file_name(stem: str, bank_index: int) -> str:
    return f"{stem}.{udim_number(bank_index):04d}.png"


def udim_pattern(stem: str) -> str:
    return f"{stem}.{UDIM_TOKEN}.png"


# ----- PNG i/o -------------------------------------------------------------------


def write_png16(path, array: np.ndarray, compress_level: int = 1) -> None:
    """A linear 16-bit grayscale PNG (what the demo's MDL samples)."""
    from PIL import Image

    if array.dtype != np.uint16 or array.ndim != 2:
        raise ValueError("expected a 2-D uint16 array")
    Image.MAX_IMAGE_PIXELS = None
    # Pillow >= 12 infers "I;16" for uint16 and rejects an explicit mode argument.
    image = Image.fromarray(np.ascontiguousarray(array))
    if not image.mode.startswith("I;16"):
        raise RuntimeError(f"unexpected PIL mode {image.mode} for a uint16 atlas")
    image.save(str(path), compress_level=compress_level)


def read_png16(path) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(str(path)) as image:
        array = np.asarray(image, dtype=np.uint16)
    if array.ndim != 2:
        raise ValueError(f"not a grayscale texture: {path}")
    return array
