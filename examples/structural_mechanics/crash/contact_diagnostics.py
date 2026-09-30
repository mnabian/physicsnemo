# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate sparse contact construction over a bumper VTP trajectory.

This utility checks graph invariants and analytic cylinder gaps. The bumper VTP
files do not contain finite-element contact labels, so the report intentionally
does not present contact precision or recall.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from contact_graph import BumperCylinderContactEncoder
from datapipe import CrashGraphDataset
from utils import get_global_features_for_run, load_global_features
from vtp_reader import build_edges_from_mesh_connectivity, load_vtp_file

from physicsnemo.experimental.models.meshtransolver import (
    SparseContactGraphBuilder,
)


def _structural_edge_index(
    mesh_connectivity: list[list[int]], num_nodes: int
) -> torch.Tensor:
    """Build a bidirectional structural graph, including self loops."""

    edges = sorted(build_edges_from_mesh_connectivity(mesh_connectivity))
    if edges:
        undirected = torch.tensor(edges, dtype=torch.long)
        edge_index = torch.cat((undirected.T, undirected[:, [1, 0]].T), dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    self_loops = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)
    return torch.cat((edge_index, self_loops), dim=1)


def _metric_range(records: list[dict], key: str) -> list[float | int | None]:
    values = [record[key] for record in records if record[key] is not None]
    return [min(values), max(values)] if values else [None, None]


def _frame_record(
    frame: int,
    num_nodes: int,
    component_ids: torch.Tensor,
    structural_codes: torch.Tensor,
    node_graph,
    cylinder_graph,
    node_scale: float,
    cylinder_scale: float,
    elapsed_seconds: float,
) -> dict:
    source, destination = node_graph.edge_index
    degree = torch.bincount(destination, minlength=num_nodes)
    combined_degree = degree + torch.bincount(
        cylinder_graph.edge_index[1], minlength=num_nodes
    )
    if source.numel():
        same_component = component_ids[source] == component_ids[destination]
        candidate_codes = source * num_nodes + destination
        locations = torch.searchsorted(structural_codes, candidate_codes)
        bounded = locations < structural_codes.numel()
        structural_overlap = bounded & (
            structural_codes[locations.clamp_max(structural_codes.numel() - 1)]
            == candidate_codes
        )
        node_gap = node_graph.edge_features[:, 4] * node_scale
    else:
        same_component = torch.empty(0, dtype=torch.bool, device=component_ids.device)
        structural_overlap = same_component
        node_gap = torch.empty(0, device=component_ids.device)

    if cylinder_graph.edge_index.shape[1]:
        cylinder_gap = cylinder_graph.edge_features[:, 4] * cylinder_scale
    else:
        cylinder_gap = torch.empty(0, device=component_ids.device)

    return {
        "frame": frame,
        "node_edges": int(source.numel()),
        "node_active_nodes": int((degree > 0).sum()),
        "node_max_in_degree": int(degree.max()),
        "combined_max_in_degree": int(combined_degree.max()),
        "same_component_edges": int(same_component.sum()),
        "cross_component_edges": int((~same_component).sum()),
        "structural_edge_overlap": int(structural_overlap.sum()),
        "node_gap_min_mm": float(node_gap.min()) if node_gap.numel() else None,
        "node_gap_median_mm": (float(node_gap.median()) if node_gap.numel() else None),
        "node_nonpositive_gap_edges": int((node_gap <= 0).sum()),
        "cylinder_edges": int(cylinder_graph.edge_index.shape[1]),
        "cylinder_gap_min_mm": (
            float(cylinder_gap.min()) if cylinder_gap.numel() else None
        ),
        "cylinder_gap_median_mm": (
            float(cylinder_gap.median()) if cylinder_gap.numel() else None
        ),
        "cylinder_nonpositive_gap_edges": int((cylinder_gap <= 0).sum()),
        "build_seconds": elapsed_seconds,
    }


def diagnose(args: argparse.Namespace) -> dict:
    positions, connectivity, point_data = load_vtp_file(str(args.vtp))
    num_nodes = positions.shape[1]
    structural_edges = _structural_edge_index(connectivity, num_nodes)
    component_ids = CrashGraphDataset.connected_component_ids(
        structural_edges, num_nodes
    )

    global_features = get_global_features_for_run(
        load_global_features(str(args.global_features)), args.vtp.stem
    )
    thickness_scale = float(global_features[args.thickness_scale_feature])
    exported_thickness = np.asarray(
        point_data.get("thickness", np.zeros(num_nodes)), dtype=np.float32
    ).reshape(-1)
    if exported_thickness.shape != (num_nodes,):
        raise ValueError(
            "VTP thickness must have one value per node; got "
            f"{exported_thickness.shape}"
        )
    fallback_thickness = args.base_shell_thickness * thickness_scale
    if np.all(exported_thickness > 0.0):
        shell_thickness = torch.from_numpy(exported_thickness.copy())
        thickness_source = "vtp"
    elif np.any(exported_thickness > 0.0):
        shell_thickness = torch.from_numpy(
            np.where(
                exported_thickness > 0.0,
                exported_thickness,
                fallback_thickness,
            ).astype(np.float32)
        )
        thickness_source = "vtp_with_scaled_nominal_fallback"
    else:
        shell_thickness = torch.full((num_nodes,), fallback_thickness)
        thickness_source = "scaled_nominal_fallback"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    structural_edges = structural_edges.to(device)
    component_ids = component_ids.to(device)
    shell_thickness = shell_thickness.to(device)

    node_builder = SparseContactGraphBuilder(
        search_radius=args.node_contact_radius,
        max_neighbors=args.max_neighbors,
        candidate_neighbors=args.candidate_neighbors,
        exclude_structural_edges=True,
        exclude_same_component=args.exclude_same_component,
    ).to(device)
    cylinder_encoder = BumperCylinderContactEncoder(
        center_x=args.cylinder_center_x,
        center_z=args.cylinder_center_z,
        radius=args.cylinder_radius,
        search_distance=args.cylinder_search_distance,
    ).to(device)

    structural_codes = (
        torch.unique(structural_edges[0] * num_nodes + structural_edges[1])
        .sort()
        .values
    )
    records = []
    for frame, frame_positions in enumerate(positions):
        current = torch.as_tensor(frame_positions, dtype=torch.float32, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        node_graph = node_builder(
            current,
            structural_edge_index=structural_edges,
            shell_thickness=shell_thickness,
            component_ids=component_ids,
        )
        cylinder_graph = cylinder_encoder(
            current,
            center_y=torch.tensor(
                [float(global_features[args.cylinder_center_y_feature])],
                dtype=current.dtype,
                device=device,
            ),
            shell_thickness=shell_thickness,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        records.append(
            _frame_record(
                frame=frame,
                num_nodes=num_nodes,
                component_ids=component_ids,
                structural_codes=structural_codes,
                node_graph=node_graph,
                cylinder_graph=cylinder_graph,
                node_scale=args.node_contact_radius,
                cylinder_scale=args.cylinder_search_distance,
                elapsed_seconds=time.perf_counter() - started,
            )
        )

    component_sizes = torch.bincount(component_ids).cpu().tolist()
    return {
        "input": str(args.vtp),
        "device": str(device),
        "trajectory_shape": list(positions.shape),
        "component_sizes": component_sizes,
        "thickness_source": thickness_source,
        "shell_thickness_range_mm": [
            float(shell_thickness.min()),
            float(shell_thickness.max()),
        ],
        "settings": {
            "node_contact_radius_mm": args.node_contact_radius,
            "max_neighbors": args.max_neighbors,
            "candidate_neighbors": args.candidate_neighbors,
            "exclude_same_component": args.exclude_same_component,
            "cylinder_radius_mm": args.cylinder_radius,
            "cylinder_search_distance_mm": args.cylinder_search_distance,
        },
        "summary": {
            "node_edges_range": _metric_range(records, "node_edges"),
            "node_active_nodes_range": _metric_range(records, "node_active_nodes"),
            "node_max_in_degree": max(
                record["node_max_in_degree"] for record in records
            ),
            "combined_max_in_degree": max(
                record["combined_max_in_degree"] for record in records
            ),
            "structural_edge_overlap_total": sum(
                record["structural_edge_overlap"] for record in records
            ),
            "same_component_edges_total": sum(
                record["same_component_edges"] for record in records
            ),
            "cross_component_edges_total": sum(
                record["cross_component_edges"] for record in records
            ),
            "node_gap_min_mm": min(
                record["node_gap_min_mm"]
                for record in records
                if record["node_gap_min_mm"] is not None
            ),
            "cylinder_edges_range": _metric_range(records, "cylinder_edges"),
            "cylinder_gap_min_mm": min(
                record["cylinder_gap_min_mm"]
                for record in records
                if record["cylinder_gap_min_mm"] is not None
            ),
            "mean_build_seconds": sum(record["build_seconds"] for record in records)
            / len(records),
        },
        "frames": records,
        "limitations": (
            "The VTP contains no finite-element contact labels; this report checks "
            "graph invariants and analytic cylinder gaps, not precision or recall."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtp", required=True, type=Path)
    parser.add_argument("--global-features", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--node-contact-radius", type=float, default=10.0)
    parser.add_argument("--max-neighbors", type=int, default=32)
    parser.add_argument("--candidate-neighbors", type=int, default=128)
    parser.add_argument("--exclude-same-component", action="store_true")
    parser.add_argument("--base-shell-thickness", type=float, default=2.0)
    parser.add_argument("--thickness-scale-feature", default="thickness_scale")
    parser.add_argument("--cylinder-center-x", type=float, default=-170.0)
    parser.add_argument("--cylinder-center-z", type=float, default=0.0)
    parser.add_argument("--cylinder-radius", type=float, default=127.0)
    parser.add_argument("--cylinder-search-distance", type=float, default=200.0)
    parser.add_argument("--cylinder-center-y-feature", default="rwall_origin_y")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = diagnose(arguments)
    serialized = json.dumps(report, indent=2)
    if arguments.output:
        arguments.output.write_text(serialized + "\n")
    else:
        print(serialized)
