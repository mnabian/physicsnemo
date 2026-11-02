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
"""Compose rendered frames of several trajectories into labelled comparison
grids (rows: trajectories, columns: frames) with the viewer's contour colour
bar, from the PNGs and ``render_manifest.json`` files of ``render_headless.py``.

    python -m visualization.ovrtx.montage --renders <dir with ground_truth/ teacher/ corrector/> \\
        --field Von_Mises_Von_Mises_layer_Outer_ip_Centroid --frames 0,8,16,24 --out crossing_depth.png \\
        --title "Shell crossing depth per face [mm]" --time-ms 0,40,80,120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .materials import CONTOUR_RGB8

LABELS = {
    "ground_truth": "Ground truth (LS-DYNA)",
    "teacher": "Teacher (DeFormer, no contact)",
    "corrector": "Teacher + contact corrector",
}


def _font(size: int):
    for name in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def crop_box(image: Image.Image, crop: tuple[float, float, float, float] | None):
    if not crop:
        return image
    w, h = image.size
    left, top, right, bottom = crop
    return image.crop((int(left * w), int(top * h), int(right * w), int(bottom * h)))


def colour_bar(
    width: int, height: int, lower: float, upper: float, title: str, font
) -> Image.Image:
    bar = Image.new("RGB", (width, height), "white")
    ramp = np.asarray(CONTOUR_RGB8, dtype=np.uint8)
    # the MDL samples the 8-texel LUT with clamping and linear filtering -> smooth ramp
    xs = np.linspace(0, len(ramp) - 1, width)
    lo = np.floor(xs).astype(int)
    hi = np.minimum(lo + 1, len(ramp) - 1)
    t = (xs - lo)[:, None]
    row = (ramp[lo] * (1 - t) + ramp[hi] * t).astype(np.uint8)
    strip = np.repeat(row[None], max(1, height // 3), axis=0)
    bar.paste(Image.fromarray(strip, "RGB"), (0, height - strip.shape[0]))
    draw = ImageDraw.Draw(bar)
    draw.text((0, 0), title, fill="black", font=font)
    draw.text((0, height // 3), f"{lower:g}", fill="black", font=font)
    text = f"{upper:g}"
    draw.text(
        (width - draw.textlength(text, font=font), height // 3),
        text,
        fill="black",
        font=font,
    )
    return bar


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--renders",
        type=Path,
        required=True,
        help="directory holding one render_headless.py output directory per trajectory",
    )
    parser.add_argument(
        "--labels",
        default="ground_truth,teacher,corrector",
        help="comma-separated trajectory directories under --renders, one row each; "
        f"the captions {sorted(LABELS)} describe the shipped models, any other name is "
        "printed as is; a 'hero_' prefix (a second render of the same trajectory, e.g. "
        "hero_teacher) is stripped before the caption lookup",
    )
    parser.add_argument(
        "--field", required=True, help="display mode whose frames are tiled"
    )
    parser.add_argument(
        "--frames", default="0,8,16,24", help="comma-separated frames, one column each"
    )
    parser.add_argument(
        "--time-ms", default=None, help="column captions (defaults to frame numbers)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--crop",
        type=float,
        nargs=4,
        default=None,
        metavar=("L", "T", "R", "B"),
        help="fractional crop of every frame before tiling",
    )
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--no-colorbar", action="store_true")
    args = parser.parse_args(argv)
    labels = [v for v in args.labels.split(",") if v]
    frames = [int(v) for v in args.frames.split(",") if v.strip()]
    captions = args.time_ms.split(",") if args.time_ms else [str(f) for f in frames]
    font, small = _font(26), _font(22)
    tiles = []
    lower = upper = None
    row_ranges: list[tuple[float, float] | None] = []
    for label in labels:
        row = []
        row_range = None
        manifest_path = args.renders / label / "render_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.is_file()
            else {"images": []}
        )
        for frame in frames:
            path = args.renders / label / f"{args.field}_frame{frame:02d}.png"
            image = crop_box(Image.open(path).convert("RGB"), args.crop)
            scale = args.tile_width / image.width
            row.append(
                image.resize(
                    (args.tile_width, int(image.height * scale)), Image.LANCZOS
                )
            )
            for entry in manifest["images"]:
                if entry["field"] == args.field and entry["frame"] == frame:
                    lower, upper = entry["lower"], entry["upper"]
                    row_range = (entry["lower"], entry["upper"])
        tiles.append(row)
        row_ranges.append(row_range)
    tile_h = tiles[0][0].height
    left_margin, top_margin, gap = 300, 60, 8
    bar_h = 0 if args.no_colorbar or args.field == "Solid_Color_Default" else 90
    width = left_margin + len(frames) * (args.tile_width + gap)
    height = top_margin + len(labels) * (tile_h + gap) + bar_h + 20
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    if args.title:
        draw.text((left_margin, 8), args.title, fill="black", font=font)
    for j, caption in enumerate(captions):
        x = left_margin + j * (args.tile_width + gap)
        draw.text(
            (x, top_margin - 30),
            f"t = {caption} ms" if args.time_ms else f"frame {caption}",
            fill="black",
            font=small,
        )
    for i, label in enumerate(labels):
        y = top_margin + i * (tile_h + gap)
        text = LABELS.get(label.removeprefix("hero_"), label)
        lines = text.split(" (")
        if row_ranges[i] is not None and args.field != "Solid_Color_Default":
            lines.append(f"colour range {row_ranges[i][0]:g}..{row_ranges[i][1]:g}")
        for k, line in enumerate(lines):
            draw.text(
                (10, y + tile_h // 2 - 20 + 28 * k),
                (("(" + line) if (k and k < len(text.split(" ("))) else line),
                fill="black",
                font=small,
            )
        for j, tile in enumerate(tiles[i]):
            canvas.paste(tile, (left_margin + j * (args.tile_width + gap), y))
    if bar_h and lower is not None:
        distinct = {r for r in row_ranges if r is not None}
        caption = (
            (args.title or args.field)
            if len(distinct) <= 1
            else "contour ramp (each row's range is printed at its label)"
        )
        bar = colour_bar(
            min(900, width - left_margin - 20), bar_h - 10, lower, upper, caption, small
        )
        if len(distinct) > 1:
            # per-row ranges differ: label the bar as relative (0 .. 1 of each row's range)
            bar = colour_bar(
                min(900, width - left_margin - 20), bar_h - 10, 0.0, 1.0, caption, small
            )
        canvas.paste(bar, (left_margin, height - bar_h - 10))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"wrote {args.out} ({canvas.size[0]}x{canvas.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
