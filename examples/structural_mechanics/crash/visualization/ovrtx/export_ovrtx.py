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
"""Export DeFormer crash predictions as car-crash OVRTX viewer bundles.

Sources (one per invocation):

* ``--yaris-dump <case dir> --label ground_truth|teacher|corrector|...``: a
  case directory written by ``dump_yaris_predictions.py`` on the cluster;
* ``--vtp-pred-dir <dir> [--vtp-exact-dir <dir>]``: the ``frame_XXX_pred.vtp``
  / ``frame_XXX_exact.vtp`` sequences of ``inference.save_vtp_sequence`` (body-in-white);
* ``--biw-vtp <file.vtp>``: a raw body-in-white crash VTP (its ground truth);
* ``--synthetic``: the test scene.

Example (one bundle per trajectory of one Yaris case)::

    python -m visualization.ovrtx.export_ovrtx --yaris-dump dumps/<case> --label corrector \\
        --out bundles/<case>/corrector --frames 0-24 --environment-hdr <release>/ZetoCGcom_ExhibitionHall_Interior1.hdr

Then ``python -m visualization.ovrtx.validate bundles/<case>/corrector`` (or
``--validate`` here, which adds the report to the JSON summary). The viewer's
MDL shader templates are copied from ``--templates-dir`` (see
``templates/README.md``). The bundle metadata records this machine's paths and
the command line.
Needs ``usd-core`` (pxr), ``numpy``, ``pillow`` (and ``pyvista`` for VTP input);
nothing is written when one of them, the environment HDR or the templates
directory is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bundle import export_bundle
from .fields import SlotAssignment
from .scene import (
    DEMO_FRAME_COUNT,
    FACE_SLOT,
    TRANSFORMS,
    U_SLOT,
    V_SLOT,
    CrashScene,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--yaris-dump", type=Path, help="case directory of dump_yaris_predictions.py"
    )
    source.add_argument(
        "--vtp-pred-dir", type=Path, help="directory of frame_XXX_pred.vtp files"
    )
    source.add_argument(
        "--biw-vtp", type=Path, help="a raw body-in-white crash VTP (ground truth)"
    )
    source.add_argument(
        "--synthetic", action="store_true", help="export the synthetic test scene"
    )
    parser.add_argument(
        "--label", default="ground_truth", help="trajectory of the Yaris dump to export"
    )
    parser.add_argument(
        "--vtp-exact-dir",
        type=Path,
        default=None,
        help="directory of frame_XXX_exact.vtp files (error field)",
    )
    parser.add_argument(
        "--vtp-parts",
        choices=("single", "components"),
        default="components",
        help="part grouping for VTP input without part ids",
    )
    parser.add_argument(
        "--vtp-dt-ms",
        type=float,
        default=5.0,
        help="frame interval of a VTP sequence in milliseconds (time_ms = frame number x interval "
        "in the bundle metadata; default 5.0)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="bundle directory (car_crash_assets/ and runtime/ are created inside)",
    )
    parser.add_argument(
        "--frames",
        default="demo",
        help=f"frames to export: 'demo' (first {DEMO_FRAME_COUNT}), 'fit' ({DEMO_FRAME_COUNT} spread evenly "
        "over the trajectory), 'a-b', 'a,b,c', 'stride:k', or 'all'. The viewer's timeline holds "
        f"{DEMO_FRAME_COUNT} frames, so a selection of more frames ('all' on a longer trajectory) "
        "is an error: use 'fit' or an explicit range/list instead",
    )
    parser.add_argument(
        "--slot-face",
        default=None,
        help=f"scene field for the per-face slot ({FACE_SLOT})",
    )
    parser.add_argument(
        "--slot-u", default=None, help=f"scene field for the U vertex slot ({U_SLOT})"
    )
    parser.add_argument(
        "--slot-v", default=None, help=f"scene field for the V vertex slot ({V_SLOT})"
    )
    parser.add_argument(
        "--transform",
        choices=sorted(TRANSFORMS),
        default="identity",
        help="rigid transform of the positions before export (default keeps the z-up mm data; the stage rotates /World/d3plot like the demo)",
    )
    parser.add_argument(
        "--translate",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="translation (scene units, mm) added to the positions after --transform",
    )
    parser.add_argument(
        "--environment-hdr",
        type=Path,
        default=None,
        help="dome light texture copied into car_crash_assets/",
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=None,
        help="directory holding the viewer release's MDL shader templates (default: $OVRTX_TEMPLATES_DIR, "
        "else visualization/ovrtx/templates/; see templates/README.md)",
    )
    parser.add_argument(
        "--source-frames",
        choices=("all", "reference"),
        default="all",
        help="time samples in main_project.usd: every frame (demo-tool compatible) or the reference only (small)",
    )
    parser.add_argument(
        "--max-texture-side",
        type=int,
        default=8192,
        help="largest side (texels) of a per-vertex UDIM atlas page; frames that do not fit "
        "go to further pages (default 8192, the viewer release's size)",
    )
    parser.add_argument(
        "--material-groups",
        type=int,
        default=12,
        help="number of OmniSurface colour groups the parts are spread over round-robin "
        "(1..12, the 'Solid_Color_Default' display mode)",
    )
    parser.add_argument(
        "--no-solids",
        action="store_true",
        help="Yaris: skip the exterior faces of solid elements",
    )
    parser.add_argument(
        "--single-sided",
        action="store_true",
        help="author doubleSided=false like the demo's source meshes",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the car_crash_assets/ and runtime/ directories of an existing bundle",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="run visualization.ovrtx.validate on the result and add its report to the "
        "JSON summary under 'validation' (exit status 1 when it fails)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="scene name recorded in the bundle metadata (default: from the source)",
    )
    return parser


def load_scene(args) -> tuple[CrashScene, dict]:
    extra: dict = {}
    if args.synthetic:
        from .scene import synthetic_scene

        scene = synthetic_scene(num_frames=6)
    elif args.yaris_dump is not None:
        from .scene import load_yaris_dump

        scene, meta = load_yaris_dump(
            args.yaris_dump,
            args.label,
            frame_spec=args.frames,
            include_solids=not args.no_solids,
        )
        extra["dump_meta"] = meta
    elif args.vtp_pred_dir is not None:
        from .scene import load_vtp_sequence

        scene = load_vtp_sequence(
            args.vtp_pred_dir,
            args.vtp_exact_dir,
            frame_spec=args.frames,
            parts=args.vtp_parts,
            name=args.name,
            dt_ms=args.vtp_dt_ms,
        )
    else:
        from .scene import load_biw_vtp

        scene = load_biw_vtp(args.biw_vtp, frame_spec=args.frames, parts=args.vtp_parts)
    if args.name:
        scene.name = args.name
    if args.transform != "identity" or any(abs(v) > 0 for v in args.translate):
        scene = scene.transformed(args.transform, args.translate)
    return scene, extra


def default_assignment(args, scene: CrashScene) -> SlotAssignment:
    label = (
        args.label
        if args.yaris_dump is not None
        else ("ground_truth" if "error_mm" not in scene.node_fields else "model")
    )
    return SlotAssignment.for_scene(
        scene, label, face=args.slot_face, u=args.slot_u, v=args.slot_v
    )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    scene, extra = load_scene(args)
    if scene.num_frames > DEMO_FRAME_COUNT:
        parser.error(
            f"--frames {args.frames!r} selects {scene.num_frames} frames but the viewer's "
            f"timeline holds {DEMO_FRAME_COUNT}; use --frames fit ({DEMO_FRAME_COUNT} frames "
            "spread over the trajectory), 'demo' (the first "
            f"{DEMO_FRAME_COUNT}) or an explicit range or list"
        )
    assignment = default_assignment(args, scene)
    print(
        f"[export] slots: face={assignment.face} u={assignment.u} v={assignment.v}",
        file=sys.stderr,
    )
    manifest = export_bundle(
        scene,
        args.out,
        assignment=assignment,
        max_texture_side=args.max_texture_side,
        material_group_count=args.material_groups,
        environment_hdr=str(args.environment_hdr) if args.environment_hdr else None,
        source_frames=args.source_frames,
        overwrite=args.overwrite,
        double_sided=not args.single_sided,
        extra_metadata={**extra, "command": sys.argv},
        templates_dir=args.templates_dir,
        log=lambda message: print(message, file=sys.stderr),
    )
    summary = {
        k: manifest[k]
        for k in (
            "scene",
            "frames",
            "layout",
            "slots",
            "bounds_source_frame",
            "seconds",
        )
    }
    status = 0
    if args.validate:
        from .validate import validate_bundle

        summary["validation"] = validate_bundle(args.out)
        status = 0 if summary["validation"]["ok"] else 1
    print(json.dumps(summary, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
