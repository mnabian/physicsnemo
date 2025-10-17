# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from lasso.dyna import D3plot, ArrayType
import pyvista as pv
import os
import re
import numpy as np
from scipy.spatial import cKDTree


def find_run_folders(base_data_dir):
    """Find all run folders in the base directory matching crash simulation patterns."""
    # Look for directories that contain d3plot files
    run_dirs = []
    if os.path.isdir(base_data_dir):
        for item in os.listdir(base_data_dir):
            item_path = os.path.join(base_data_dir, item)
            if os.path.isdir(item_path):
                d3plot_path = os.path.join(item_path, "d3plot")
                if os.path.exists(d3plot_path):
                    run_dirs.append(item)
    return run_dirs


def parse_k_file(k_file_path):
    """Parse LS-DYNA .k file to extract part thickness information."""
    part_to_section = {}
    section_thickness = {}

    with open(k_file_path, "r") as f:
        lines = [
            line.strip() for line in f if line.strip() and not line.startswith("$")
        ]

    i = 0
    while i < len(lines):
        line = lines[i]
        if "*PART" in line.upper():
            # After *PART:
            # i+1 = part name (skip)
            # i+2 = part id, section id, material id
            if i + 2 < len(lines):
                tokens = lines[i + 2].split()
                if len(tokens) >= 2:
                    part_id = int(tokens[0])
                    section_id = int(tokens[1])
                    part_to_section[part_id] = section_id
            i += 3
        elif "*SECTION_SHELL" in line.upper():
            # Multiple sections can be defined under one *SECTION_SHELL keyword
            # Each section has two lines: header line and thickness line
            i += 1  # Skip the *SECTION_SHELL line
            while i < len(lines) and not lines[i].startswith("*"):
                # Check if this line looks like a section header (starts with a number)
                if i < len(lines) and lines[i].strip() and lines[i][0].isdigit():
                    header_line = lines[i]
                    thickness_line = lines[i + 1] if i + 1 < len(lines) else ""

                    # Extract section ID from header line (first number)
                    header_tokens = header_line.split()
                    if len(header_tokens) >= 1:
                        try:
                            section_id = int(header_tokens[0])
                        except ValueError:
                            section_id = None
                    else:
                        section_id = None

                    # Extract thickness values from thickness line
                    thickness_values = []
                    thickness_tokens = thickness_line.split()
                    for t in thickness_tokens:
                        try:
                            thickness_values.append(float(t))
                        except ValueError:
                            pass

                    # Calculate average thickness (ignore zeros)
                    non_zero_thicknesses = [t for t in thickness_values if t > 0.0]
                    if non_zero_thicknesses:
                        thickness = sum(non_zero_thicknesses) / len(
                            non_zero_thicknesses
                        )
                    elif thickness_values:
                        thickness = sum(thickness_values) / len(thickness_values)
                    else:
                        thickness = 0.0

                    if section_id is not None:
                        section_thickness[section_id] = thickness

                    i += 2  # Skip both header and thickness lines
                else:
                    i += 1
        else:
            i += 1

    part_thickness = {
        pid: section_thickness.get(sid, 0.0) for pid, sid in part_to_section.items()
    }
    return part_thickness


def load_d3plot_data(data_path):
    """Load node coordinates and displacements from a d3plot file."""
    dp = D3plot(data_path)
    coords = dp.arrays[ArrayType.node_coordinates]  # (num_nodes, 3)
    pos_raw = dp.arrays[ArrayType.node_displacement]  # (timesteps, num_nodes, 3)
    mesh_connectivity = dp.arrays[ArrayType.element_shell_node_indexes]
    part_ids = dp.arrays[ArrayType.element_shell_part_indexes]

    # Get actual part IDs if available
    actual_part_ids = None
    if ArrayType.part_ids in dp.arrays:
        actual_part_ids = dp.arrays[ArrayType.part_ids]

    # assert np.allclose(coords, pos_raw[0, :, :])
    return coords, pos_raw, mesh_connectivity, part_ids, actual_part_ids


def compute_node_type(pos_raw, threshold=1.0):
    variation = np.max(np.abs(pos_raw - pos_raw[0:1, :, :]), axis=0)
    variation = np.max(variation, axis=1)
    is_wall = variation < threshold
    node_type = np.where(is_wall, 1, 0).astype(np.uint8)
    return node_type


def build_edges_from_mesh_connectivity(mesh_connectivity):
    """Build unique edges from mesh connectivity (cells of any size)."""
    edges = set()
    for cell in mesh_connectivity:
        n = len(cell)
        for idx in range(n):
            edge = tuple(sorted((cell[idx], cell[(idx + 1) % n])))
            edges.add(edge)
    return edges


def compute_node_thickness(
    mesh_connectivity, part_ids, part_thickness_map, actual_part_ids=None
):
    """
    Compute thickness for each node based on elements connected to it.

    Args:
        mesh_connectivity: Element connectivity array
        part_ids: Part IDs for each element
        part_thickness_map: Mapping from part ID to thickness
        actual_part_ids: Actual part IDs if available

    Returns:
        node_thickness: Array of thickness values for each node
    """
    # Create mapping from part index to actual part ID
    if actual_part_ids is not None:
        part_index_to_id = {}
        for i, actual_part_id in enumerate(actual_part_ids):
            if i > 0:  # Skip index 0
                part_index_to_id[i] = actual_part_id
    else:
        sorted_part_ids = sorted(part_thickness_map.keys())
        part_index_to_id = {}
        for i, part_id in enumerate(sorted_part_ids, 1):
            part_index_to_id[i] = part_id

    # Get element thickness
    element_thickness = np.zeros(len(part_ids))
    for i, part_index in enumerate(part_ids):
        actual_part_id = part_index_to_id.get(part_index)
        if actual_part_id is not None:
            thickness = part_thickness_map.get(actual_part_id, 0.0)
            element_thickness[i] = thickness

    # Find maximum node index to initialize node thickness array
    max_node_idx = 0
    for element in mesh_connectivity:
        max_node_idx = max(max_node_idx, max(element))

    node_thickness = np.zeros(max_node_idx + 1)
    node_thickness_count = np.zeros(max_node_idx + 1)

    # Accumulate thickness from all elements connected to each node
    for i, element in enumerate(mesh_connectivity):
        thickness = element_thickness[i]
        for node_idx in element:
            node_thickness[node_idx] += thickness
            node_thickness_count[node_idx] += 1

    # Average thickness for nodes connected to multiple elements
    for i in range(len(node_thickness)):
        if node_thickness_count[i] > 0:
            node_thickness[i] /= node_thickness_count[i]

    return node_thickness


def collect_mesh_pos(
    output_dir, pos_raw, filtered_mesh_connectivity, node_thickness, write_vtp=False
):
    """Write VTP files for each timestep and collect mesh/point data."""
    n_timesteps = pos_raw.shape[0]
    mesh_pos_all = []
    for t in range(n_timesteps):
        pos = pos_raw[t, :, :]

        faces = []
        for cell in filtered_mesh_connectivity:
            if len(cell) == 3:  # Triangle
                faces.extend([3, *cell])
            elif len(cell) == 4:  # Quad or Tetrahedron
                # For 2D quads, we can use them directly
                # For 3D tetrahedra, we need to extract the triangular faces
                # Since we're working with shell elements, assume they're quads
                faces.extend([4, *cell])
            elif len(cell) > 4:  # Higher order elements
                # Triangulate or skip
                continue

        faces = np.array(faces)
        mesh = pv.PolyData(pos, faces)

        # Add thickness as point data
        if len(node_thickness) >= len(pos):
            mesh.point_data["thickness"] = node_thickness[: len(pos)]

        if write_vtp:
            filename = os.path.join(output_dir, f"frame_{t:03d}.vtp")
            mesh.save(filename)
            logger.info(f"Saved: {filename}")
        mesh_pos_all.append(pos)
    return np.stack(mesh_pos_all)


def process_d3plot_data(
    data_dir, num_samples, wall_node_disp_threshold=1.0, write_vtp=False, logger=None
):
    """
    Preprocesses LS-DYNA crash simulation data for a given directory.
    For each run, computes node connectivity, node types, thickness, and writes VTP files for each timestep.
    Returns lists of source/destination node indices and point data for all runs.
    """
    processed_runs = 0
    base_data_dir = data_dir
    run_folders = find_run_folders(base_data_dir)
    srcs, dsts = [], []
    point_data_all = []

    if not run_folders:
        raise ValueError("No run folders found in:", base_data_dir)

    if logger is None:
        logger = PythonLogger()

    for run_folder in sorted(run_folders):
        logger.info(f"Processing {run_folder}...")
        data_path = os.path.join(base_data_dir, run_folder, "d3plot")
        output_dir = f"./output_{run_folder}"
        os.makedirs(output_dir, exist_ok=True)

        coords, pos_raw, mesh_connectivity, part_ids, actual_part_ids = (
            load_d3plot_data(data_path)
        )
        flat = np.fromiter(
            (n for cell in mesh_connectivity for n in cell), dtype=np.int64
        )
        assert flat.min() >= 0 and flat.max() < coords.shape[0], (
            f"Connectivity out of bounds: min={flat.min()}, max={flat.max()}, num_nodes={coords.shape[0]}"
        )

        # Get thickness from .k file
        k_file_path = None
        run_dir = os.path.join(base_data_dir, run_folder)
        for f in os.listdir(run_dir):
            if f.endswith(".k"):
                k_file_path = os.path.join(run_dir, f)
                break

        node_thickness = np.zeros(len(coords))
        if k_file_path and os.path.exists(k_file_path):
            part_thickness_map = parse_k_file(k_file_path)
            node_thickness = compute_node_thickness(
                mesh_connectivity, part_ids, part_thickness_map, actual_part_ids
            )
        else:
            logger.warning("No .k file found, using zero thickness")

        # Identify “wall” nodes (low displacement variation)
        node_type = compute_node_type(pos_raw, threshold=wall_node_disp_threshold)
        keep_nodes = sorted(np.where(node_type == 0)[0])  # keep structure
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_nodes)}

        # Filter arrays
        filtered_pos_raw = pos_raw[:, keep_nodes, :]
        filtered_node_thickness = node_thickness[keep_nodes]

        # Remap mesh connectivity to filtered node indices
        filtered_mesh_connectivity = []
        for cell in mesh_connectivity:
            filtered_cell = [node_map[n] for n in cell if n in node_map]
            if len(filtered_cell) >= 3:
                filtered_mesh_connectivity.append(filtered_cell)

        # Compact to only nodes that are actually used by any cell
        used = np.unique(
            np.array(
                [i for cell in filtered_mesh_connectivity for i in cell], dtype=np.int64
            )
        )
        if used.size == 0:
            raise ValueError(
                "No cells left after filtering; consider lowering the wall threshold."
            )

        # If not contiguous 0..(num_kept-1), compact and reindex everything
        num_kept = filtered_pos_raw.shape[1]
        if (used.min() != 0) or (used.max() != num_kept - 1) or (used.size != num_kept):
            keep2 = used.tolist()
            remap2 = {old_idx: new_idx for new_idx, old_idx in enumerate(keep2)}
            filtered_pos_raw = filtered_pos_raw[:, keep2, :]
            filtered_node_thickness = filtered_node_thickness[keep2]
            filtered_mesh_connectivity = [
                [remap2[n] for n in cell] for cell in filtered_mesh_connectivity
            ]
            num_kept = filtered_pos_raw.shape[1]

        # Final sanity checks
        used = np.unique(
            np.array(
                [i for cell in filtered_mesh_connectivity for i in cell], dtype=np.int64
            )
        )
        assert used.min() == 0 and used.max() == num_kept - 1
        # Build edges and sanity-check ranges
        edges = build_edges_from_mesh_connectivity(filtered_mesh_connectivity)
        edge_arr = np.array(list(edges), dtype=np.int64)
        assert edge_arr.min() >= 0 and edge_arr.max() < filtered_pos_raw.shape[1]

        # Prepare lines and src/dst arrays
        lines = []
        for edge in edges:
            lines.extend([2, edge[0], edge[1]])
        lines = np.array(lines)
        src, dst = np.array(list(edges)).T
        srcs.append(src)
        dsts.append(dst)

        # Write VTPs and collect point data
        mesh_pos_all = collect_mesh_pos(
            output_dir,
            filtered_pos_raw,
            filtered_mesh_connectivity,
            filtered_node_thickness,
            write_vtp=write_vtp,
        )
        point_data_all.append(
            {"mesh_pos": mesh_pos_all, "thickness": filtered_node_thickness}
        )

        processed_runs += 1
        if processed_runs >= num_samples:
            break

    return srcs, dsts, point_data_all
