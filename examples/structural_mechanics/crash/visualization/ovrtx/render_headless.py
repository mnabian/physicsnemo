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
"""Render an exported bundle to PNG frames through the car-crash OVRTX demo's
own server class, without the WebRTC stream.

The demo's ``src/car_crash_ovrtx_ovstream.py`` is loaded as a module (as its
``tools/benchmark_playback.py`` does); ``CarCrashServer`` validates the bundle,
builds the OVSTAGE scene for the requested display mode and drives the OVRTX
renderer. This script steps the timeline and copies the ``LdrColor`` render
variable to PNG files, so every image comes from exactly the code path the
viewer uses. Needs the demo's Python stack (``ovrtx``, ``ovstage``,
``ovstream``, ``warp-lang``) and an NVIDIA GPU with the Vulkan ICD::

    CAR_CRASH_GEOMETRY_BACKEND=merged python render_headless.py --server-source <demo>/src/car_crash_ovrtx_ovstream.py \\
        --bundle <bundle> --out <dir> --fields Solid_Color_Default,U_magnitude_layer_NULL_ip_NULL --frames 0,12,24

The bundle directory must hold ``car_crash_assets/`` and ``runtime/``; the
server reads them as ``<root>/car_crash_assets`` and ``<root>/runtime``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_server_module(source: Path):
    spec = importlib.util.spec_from_file_location("car_crash_ovrtx_server", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server-source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--fields",
        default="Solid_Color_Default,Von_Mises_Von_Mises_layer_Outer_ip_Centroid,U_magnitude_layer_NULL_ip_NULL,V_magnitude_layer_NULL_ip_NULL",
    )
    parser.add_argument("--frames", default="0,8,16,24")
    parser.add_argument(
        "--ranges",
        default=None,
        help="JSON {field: [lower, upper]} colour ranges; default from runtime/deformer/deformer_scene.json",
    )
    parser.add_argument(
        "--yaw", type=float, default=0.0, help="extra orbit yaw in radians"
    )
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument(
        "--zoom",
        type=float,
        default=0.0,
        help="camera distance offset (scene units; negative moves closer)",
    )
    parser.add_argument("--wireframe", action="store_true")
    parser.add_argument(
        "--section", default=None, help="axis:offset[:invert], e.g. Y:500:1"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=24,
        help="renderer steps before each capture (path-tracer convergence)",
    )
    parser.add_argument(
        "--eye",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="override the server's fixed camera eye (stage world units, Y up)",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="override the server's fixed camera target (the orbit centre)",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    module = load_server_module(args.server_source.resolve())
    wp = module.wp
    # The server's overview camera is two module constants read by camera_matrix();
    # overriding them is how another vehicle (or a close-up) gets framed.
    if args.eye is not None:
        module.EYE = np.asarray(args.eye, dtype=np.float64)
    if args.target is not None:
        module.TARGET = np.asarray(args.target, dtype=np.float64)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    # Colour ranges and labels per display mode: the exporter's slot assignment
    # (fields.SlotAssignment, written to runtime/deformer/deformer_scene.json),
    # overridden by --ranges, else the server's defaults for its own fields.
    scene_meta_path = args.bundle / "runtime" / "deformer" / "deformer_scene.json"
    ranges, labels = {}, {}
    if scene_meta_path.is_file():
        slots = json.loads(scene_meta_path.read_text()).get("slots", {})
        ranges = {k: tuple(v) for k, v in slots.get("suggested_ranges", {}).items()}
        labels = dict(slots.get("labels", {}))
    if args.ranges:
        ranges.update({k: tuple(v) for k, v in json.loads(args.ranges).items()})
    frames = [int(v) for v in parse_list(args.frames)]
    fields = parse_list(args.fields)

    started = time.perf_counter()
    app = module.CarCrashServer(args.bundle.resolve())
    print(
        f"server created: geometry backend {app.geometry_backend}, scalar backend {app.scalar_backend}, "
        f"vertex atlases {sorted(module.VERTEX_ATLAS_LABELS[f] for f in app.vertex_atlases)}",
        flush=True,
    )
    manifest = {
        "bundle": str(args.bundle.resolve()),
        "geometry_backend": app.geometry_backend,
        "camera": {
            "eye": [float(v) for v in module.EYE],
            "target": [float(v) for v in module.TARGET],
            "yaw": args.yaw,
            "pitch": args.pitch,
            "zoom": args.zoom,
        },
        "scalar_backend": app.scalar_backend,
        "images": [],
        "render_mode": module.render_mode_from_environment(),
        "width": module.WIDTH,
        "height": module.HEIGHT,
    }
    app.state.view.yaw, app.state.view.pitch, app.state.view.zoom = (
        args.yaw,
        args.pitch,
        args.zoom,
    )
    if args.section:
        parts = args.section.split(":")
        app.state.section.enabled = True
        app.state.section.axis = parts[0].upper()
        app.state.section.offset = float(parts[1])
        app.state.section.invert = bool(int(parts[2])) if len(parts) > 2 else False
    app.state.wireframe.enabled = bool(args.wireframe)

    def capture(path: Path) -> None:
        products = app.renderer.step(
            render_products={module.RENDER_PRODUCT},
            delta_time=0.0,
            ordinal=app.stage_ordinal,
        )
        frame = products[module.RENDER_PRODUCT].frames[0]
        with frame.render_vars["LdrColor"].map(
            device=module.ovrtx.Device.CUDA
        ) as mapping:
            tensor = wp.from_dlpack(mapping.tensor)
            image = tensor.numpy()
        del tensor, frame, products
        from PIL import Image

        rgba = np.ascontiguousarray(
            image.reshape(image.shape[0], image.shape[1], -1)[..., :4]
        )
        Image.fromarray(rgba, mode="RGBA").convert("RGB").save(str(path))

    for field in fields:
        app.state.field_name = field
        if field in ranges:
            app.state.lower, app.state.upper = ranges[field]
        else:
            app.state.lower, app.state.upper = module.FIELD_DEFAULT_RANGES[field]
        app.state.current_frame = float(frames[0])
        app.state.playing = False
        t0 = time.perf_counter()
        if app.renderer is not None:
            # load_scene() reuses the renderer; recreate it so the animation clock restarts at 0
            # (load_scene itself steps once with 1/FPS for non-time-driven fields).
            app.release_scalar_runtime()
            app.release_original_material_runtime()
            app.release_merged_geometry_runtime()
            app.renderer.detach_ovstage()
            app.renderer.destroy()
            app.renderer = None
            app.stream_buf = None
        app.load_scene()
        print(
            f"[{field}] scene loaded in {time.perf_counter() - t0:.1f}s (active geometry backend {app.active_geometry_backend}, "
            f"range {app.state.lower}..{app.state.upper})",
            flush=True,
        )
        if app.state.wireframe.enabled and app.wireframe_available():
            app.update_wireframe_style()
        if app.state.section.enabled and app.section_available():
            app.update_section_plane()
        for frame_index in frames:
            app.state.current_frame = float(frame_index)
            if app.uses_time_driven_vertex_atlas():
                app.load_scene()
            else:
                app.update_stage_time()
            for _ in range(max(0, args.warmup)):
                # delta_time 0: the frame is selected by the explicit position sync (and, for the
                # U/V atlases, by animation_start_frame); advancing the renderer's animation time
                # would shift the atlas tile the MDL samples.
                products = app.renderer.step(
                    render_products={module.RENDER_PRODUCT},
                    delta_time=0.0,
                    ordinal=app.stage_ordinal,
                )
                del products
            name = f"{field}_frame{frame_index:02d}.png"
            capture(out / name)
            manifest["images"].append(
                {
                    "field": field,
                    "label": labels.get(field),
                    "frame": frame_index,
                    "file": name,
                    "lower": app.state.lower,
                    "upper": app.state.upper,
                }
            )
            print(f"  frame {frame_index} -> {name}", flush=True)
    manifest["seconds"] = time.perf_counter() - started
    (args.manifest or out / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    app.release_scalar_runtime()
    app.release_original_material_runtime()
    app.release_merged_geometry_runtime()
    if app.renderer is not None:
        app.renderer.detach_ovstage()
        app.renderer.destroy()
    if app.stage is not None:
        app.stage.destroy()
    print(
        f"done: {len(manifest['images'])} images in {manifest['seconds']:.0f}s -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
