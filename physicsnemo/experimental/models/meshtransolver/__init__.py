# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Hybrid mesh-attention models for irregular simulation meshes.

The models in this package combine local MeshGraphNet processing with one of
PhysicsNeMo's global physics-attention processors.
"""

from .meshtransolver import (
    MeshAttentionHybrid,
    MeshContextFiLM,
    MeshGeometryContextEncoder,
    MeshGeoFLARE,
    MeshGeoTransolver,
    MeshTransolver,
)
from .contact import (
    CONTACT_FEATURE_DIM,
    KINEMATIC_CONTACT_FEATURE_DIM,
    ContactGraph,
    SparseContactBlock,
    SparseContactGraphBuilder,
    contact_edge_features,
    merge_contact_graphs,
    smooth_contact_cutoff,
)
from .functional_contact import ContactSearchTopology, FunctionalContactGraphBuilder
from .surface_contact import SurfaceContactGraphBuilder

__all__ = [
    "MeshAttentionHybrid",
    "MeshGeometryContextEncoder",
    "MeshContextFiLM",
    "MeshTransolver",
    "MeshGeoTransolver",
    "MeshGeoFLARE",
    "CONTACT_FEATURE_DIM",
    "KINEMATIC_CONTACT_FEATURE_DIM",
    "ContactSearchTopology",
    "FunctionalContactGraphBuilder",
    "SurfaceContactGraphBuilder",
    "smooth_contact_cutoff",
    "ContactGraph",
    "SparseContactBlock",
    "SparseContactGraphBuilder",
    "contact_edge_features",
    "merge_contact_graphs",
]
