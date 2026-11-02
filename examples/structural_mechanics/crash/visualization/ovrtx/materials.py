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
"""Materials, colour maps and MDL templates for the OVRTX bundle.

The viewer's runtime expects a handful of shader files next to the data. They
belong to the car-crash OVRTX viewer release, not to this repository, and are
copied into a bundle from a *templates directory*: the ``templates_dir``
argument of :func:`bundle.export_bundle` (``--templates-dir`` on the command
line), else the ``OVRTX_TEMPLATES_DIR`` environment variable, else
``templates/`` next to this module, whose ``README.md`` lists the files and
where the release keeps them: ``cae_materials.mdl`` (the source scalar
material whose ``scene::data_lookup_float(...)`` call the runtime rewrites per
field), ``ovrtx_atlas_scalar.mdl`` and ``ovrtx_vertex_atlas_scalar.mdl`` (the
atlas samplers) and ``ovrtx_original_material_section.mdl`` and
``ovrtx_st_scalar.mdl`` (the runtime materials). A bundle written without them
is complete except for those files: the manifest lists them under
``missing_templates`` and :mod:`validate` reports them. The 8-texel
``contour.png`` colour map is regenerated from :data:`CONTOUR_RGB8` unless the
templates directory holds the release's copy.

Part colours: parts are spread over a small palette of OmniSurface materials
(the demo's merged original-material path needs one OmniSurface material per
group with only its supported inputs), so neighbouring parts differ in colour
while the group count stays small.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import numpy as np

DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATES_DIR_VARIABLE = "OVRTX_TEMPLATES_DIR"
RUNTIME_MATERIAL_TEMPLATES = (
    "ovrtx_original_material_section.mdl",
    "ovrtx_st_scalar.mdl",
)
ATLAS_MDL = "ovrtx_atlas_scalar.mdl"
VERTEX_ATLAS_MDL = "ovrtx_vertex_atlas_scalar.mdl"
CAE_MATERIALS_MDL = "cae_materials.mdl"
CONTOUR_PNG = "contour.png"
# The shader files a bundle needs from the viewer release, and the optional colour map.
MDL_TEMPLATES = (
    CAE_MATERIALS_MDL,
    ATLAS_MDL,
    VERTEX_ATLAS_MDL,
    *RUNTIME_MATERIAL_TEMPLATES,
)
TEMPLATE_FILES = (*MDL_TEMPLATES, CONTOUR_PNG)
# Where a bundle places each template (paths relative to the bundle directory);
# :mod:`bundle` writes them there and :mod:`validate` requires every MDL file.
TEMPLATE_DESTINATIONS = {
    CAE_MATERIALS_MDL: (f"car_crash_assets/{CAE_MATERIALS_MDL}",),
    ATLAS_MDL: (f"runtime/ovrtx_atlas/{ATLAS_MDL}",),
    VERTEX_ATLAS_MDL: (
        f"runtime/ovrtx_vertex_atlas/{VERTEX_ATLAS_MDL}",
        f"runtime/ovrtx_vertex_atlas_v/{VERTEX_ATLAS_MDL}",
    ),
    **{
        name: (f"runtime/field_materials/{name}",)
        for name in RUNTIME_MATERIAL_TEMPLATES
    },
    CONTOUR_PNG: (f"car_crash_assets/colormaps/{CONTOUR_PNG}",),
}
# The release's colormaps/contour.png, an 8 x 1 RGB ramp (blue -> cyan -> green -> yellow -> red).
CONTOUR_RGB8 = (
    (0, 0, 255),
    (0, 145, 255),
    (0, 255, 218),
    (0, 255, 73),
    (72, 255, 0),
    (218, 255, 0),
    (255, 145, 0),
    (255, 0, 0),
)
SECTION_MATERIAL_PREFIX = "OVRTXOriginalSection"
# The pattern the runtime's prepare_runtime_materials rewrites in cae_materials.mdl.
DATA_LOOKUP_PATTERN = re.compile(r'scene::data_lookup_float\("[^"]+", 0\.0f\)')

# A muted automotive palette (sRGB 0..1) for the part groups.
PALETTE = (
    (0.62, 0.64, 0.66),
    (0.80, 0.33, 0.27),
    (0.25, 0.45, 0.72),
    (0.86, 0.70, 0.30),
    (0.40, 0.62, 0.40),
    (0.55, 0.38, 0.62),
    (0.90, 0.55, 0.25),
    (0.35, 0.70, 0.72),
    (0.72, 0.72, 0.52),
    (0.50, 0.50, 0.58),
    (0.78, 0.45, 0.55),
    (0.30, 0.55, 0.50),
)


def resolve_templates_dir(templates_dir=None) -> Path:
    """The directory the viewer's shader templates are copied from: the
    argument, else ``$OVRTX_TEMPLATES_DIR``, else ``templates/`` next to this module."""
    if templates_dir is not None:
        return Path(templates_dir).expanduser()
    configured = os.environ.get(TEMPLATES_DIR_VARIABLE)
    return Path(configured).expanduser() if configured else DEFAULT_TEMPLATES_DIR


def template_path(name: str, templates_dir=None) -> Path:
    return resolve_templates_dir(templates_dir) / name


def available_templates(templates_dir=None) -> dict[str, bool]:
    """Which of :data:`TEMPLATE_FILES` the templates directory holds."""
    directory = resolve_templates_dir(templates_dir)
    return {name: (directory / name).is_file() for name in TEMPLATE_FILES}


def copy_template(name: str, destination, templates_dir=None) -> Path | None:
    """Copy template ``name`` to ``destination``; ``None`` when the templates
    directory has no such file (``templates/README.md`` says where to get it)."""
    source = template_path(name, templates_dir)
    if not source.is_file():
        return None
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def write_contour_png(path) -> Path:
    """The viewer's 8-texel contour colour map (identical to the release's)."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray([CONTOUR_RGB8], dtype=np.uint8), mode="RGB")
    image.save(str(path))
    return path


def check_cae_materials(text: str) -> None:
    """The runtime rewrites exactly one scene::data_lookup_float(..., 0.0f) call."""
    if not DATA_LOOKUP_PATTERN.search(text):
        raise ValueError(
            'cae_materials.mdl has no scene::data_lookup_float("...", 0.0f) call to rewrite'
        )
    if "export material ScalarColor" not in text:
        raise ValueError("cae_materials.mdl must export material ScalarColor")


def part_material_groups(
    part_ids: list[int], group_count: int = len(PALETTE)
) -> tuple[list[dict], list[int]]:
    """``(groups, group_of_part)``: ``groups[i] = {"name", "section_material_name",
    "parameters"}`` with OmniSurface inputs, parts assigned round-robin in part
    order so adjacent part ids get different colours."""
    group_count = max(1, min(int(group_count), len(PALETTE)))
    groups = []
    for index in range(group_count):
        rgb = PALETTE[index]
        groups.append(
            {
                "name": f"PartGroup_{index:02d}",
                "section_material_name": f"{SECTION_MATERIAL_PREFIX}_{index:02d}",
                "parameters": {
                    "diffuse_reflection_color": [float(v) for v in rgb],
                    "diffuse_reflection_weight": 1.0,
                    "metalness": 0.15,
                    "specular_reflection_roughness": 0.45,
                    "specular_reflection_weight": 1.0,
                },
            }
        )
    assignment = [index % group_count for index in range(len(part_ids))]
    return groups, assignment
