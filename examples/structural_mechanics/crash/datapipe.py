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

import os
import numpy as np
import torch

from typing import Optional, List, Dict, Any

from d3plot_reader import process_d3plot_data
from physicsnemo.datapipes.gnn.utils import load_json, save_json
from physicsnemo.launch.logging import PythonLogger

try:
    import dgl
except ImportError:
    raise ImportError(
        "Graph dataset requires DGL. Install a suitable CUDA version: https://www.dgl.ai/pages/start.html"
    )


class GraphLike:
    """
    Minimal wrapper to mimic the graph object interface used by training.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.ndata = {"x": x, "y": y}

    def to(self, device):
        self.ndata = {k: v.to(device) for k, v in self.ndata.items()}
        return self


class CrashBaseDataset:
    """
    Shared base for Crash datasets (graph and point-cloud).
    - Loads raw data via process_d3plot_data.
    - Computes/loads node and thickness stats.
    - Normalizes position trajectories and thickness.
    - Provides a common x/y builder to keep training interchangeable.
    """

    def __init__(
        self,
        name: str = "dataset",
        data_dir: Optional[str] = None,
        split: str = "train",
        num_samples: int = 1000,
        num_steps: int = 400,
        wall_node_disp_threshold: float = 1.0,
        write_vtp: bool = False,
        logger=None,
    ):
        super().__init__()
        self.name = name
        self.split = split
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.length = num_samples
        self.logger = logger or PythonLogger()

        self.logger.info(f"Preparing the {split} dataset...")

        # Load raw records; we keep (srcs, dsts) for graph dataset; point-cloud ignores them
        self.srcs, self.dsts, point_data = process_d3plot_data(
            data_dir,
            num_samples,
            wall_node_disp_threshold,
            write_vtp,
            logger=self.logger,
        )

        # Storage for per-sample tensors
        self.mesh_pos_seq: List[torch.Tensor] = []  # [T, N, 3], float32
        self.thickness_data: List[torch.Tensor] = []  # [N], float32

        for rec in point_data:
            data_np = {
                k: (rec[k][:num_steps] if k != "thickness" else rec[k]) for k in rec
            }
            self.mesh_pos_seq.append(
                torch.tensor(data_np["mesh_pos"], dtype=torch.float32)
            )
            self.thickness_data.append(
                torch.tensor(data_np["thickness"], dtype=torch.float32)
            )

        # Stats (node + thickness)
        if self.split == "train":
            self.node_stats = self._get_areg_node_stats(dt=5e-3)
            self.thickness_stats = self._get_thickness_stats()
        else:
            self.node_stats = load_json("node_stats.json")
            self.thickness_stats = load_json("thickness_stats.json")

        # Normalize trajectories and thickness
        for i in range(self.num_samples):
            self.mesh_pos_seq[i] = self._normalize_node_tensor(
                self.mesh_pos_seq[i],
                self.node_stats["pos_mean"],
                self.node_stats["pos_std"],
            )
            self.thickness_data[i] = self._normalize_thickness_tensor(
                self.thickness_data[i],
                self.thickness_stats["thickness_mean"],
                self.thickness_stats["thickness_std"],
            )

    def __len__(self):
        return self.length

    # Common x/y construction used by both datasets
    def build_xy(self, idx: int):
        """
        x: [N, 4] = pos_t0(3) + thickness(1)
        y: [N, (T-1)*3] flattened all future positions
        """
        thickness_expanded = self.thickness_data[idx].unsqueeze(1)  # [N, 1]
        pos_t0 = self.mesh_pos_seq[idx][0]
        x = torch.cat(
            [
                pos_t0,
                thickness_expanded,
            ],
            dim=1,
        )
        y = self.mesh_pos_seq[idx][1:].transpose(0, 1).flatten(start_dim=1)
        return x, y

    # Shared stats helpers
    def _get_areg_node_stats(self, dt: float):
        stats = {
            "pos_mean": 0,
            "pos_meansqr": 0,
            "norm_vel_mean": 0,
            "norm_vel_meansqr": 0,
            "norm_acc_mean": 0,
            "norm_acc_meansqr": 0,
        }
        # position stats
        for i in range(self.num_samples):
            stats["pos_mean"] += (
                torch.mean(self.mesh_pos_seq[i], dim=(0, 1)) / self.num_samples
            )
            stats["pos_meansqr"] += (
                torch.mean(torch.square(self.mesh_pos_seq[i]), dim=(0, 1))
                / self.num_samples
            )
        stats["pos_std"] = torch.sqrt(
            stats["pos_meansqr"] - torch.square(stats["pos_mean"])
        )
        stats.pop("pos_meansqr")

        # normalized velocity stats
        for i in range(self.num_samples):
            vel = (self.mesh_pos_seq[i][1:] - self.mesh_pos_seq[i][:-1]) / (
                dt * stats["pos_std"]
            )
            stats["norm_vel_mean"] += torch.mean(vel, dim=(0, 1)) / self.num_samples
            stats["norm_vel_meansqr"] += (
                torch.mean(torch.square(vel), dim=(0, 1)) / self.num_samples
            )
        stats["norm_vel_std"] = torch.sqrt(
            stats["norm_vel_meansqr"] - torch.square(stats["norm_vel_mean"])
        )
        stats.pop("norm_vel_meansqr")

        # normalized acceleration stats
        for i in range(self.num_samples):
            acc = (
                self.mesh_pos_seq[i][:-2]
                + self.mesh_pos_seq[i][2:]
                - 2 * self.mesh_pos_seq[i][1:-1]
            ) / (dt**2 * stats["pos_std"])
            stats["norm_acc_mean"] += torch.mean(acc, dim=(0, 1)) / self.num_samples
            stats["norm_acc_meansqr"] += (
                torch.mean(torch.square(acc), dim=(0, 1)) / self.num_samples
            )
        stats["norm_acc_std"] = torch.sqrt(
            stats["norm_acc_meansqr"] - torch.square(stats["norm_acc_mean"])
        )
        stats.pop("norm_acc_meansqr")

        save_json(stats, "node_stats.json")
        return stats

    def _get_thickness_stats(self):
        all_thickness = torch.cat(self.thickness_data, dim=0)
        stats = {
            "thickness_mean": torch.mean(all_thickness),
            "thickness_std": torch.std(all_thickness),
        }
        save_json(stats, "thickness_stats.json")
        return stats

    @staticmethod
    def _normalize_node_tensor(
        invar: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
    ):
        if (invar.size()[-1] != mu.size()[-1]) or (invar.size()[-1] != std.size()[-1]):
            raise AssertionError("input and stats must have the same size")
        return (invar - mu.expand(invar.size())) / std.expand(invar.size())

    @staticmethod
    def _normalize_thickness_tensor(
        thickness: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
    ):
        if std == 0:
            return thickness
        return (thickness - mu) / std


class CrashGraphDataset(CrashBaseDataset):
    """
    Graph version:
    - Builds DGL graphs using your existing method (create_graph + add_self_loop)
    - Computes/loads edge stats and normalizes edge features
    - Emits DGLGraph with ndata["x"], ndata["y"]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Filter self-edges and create graphs
        _srcs, _dsts = [], []
        for src, dst in zip(self.srcs, self.dsts):
            _src, _dst = [], []
            for si, di in zip(src, dst):
                if si != di:
                    _src.append(si)
                    _dst.append(di)
            _srcs.append(np.array(_src))
            _dsts.append(np.array(_dst))
        self.srcs, self.dsts = _srcs, _dsts

        self.graphs: List[dgl.DGLGraph] = []
        for i in range(self.num_samples):
            g = self.create_graph(self.srcs[i], self.dsts[i], dtype=torch.int32)
            # Use initial raw position (t=0) to build edge features (relative disp + norm)
            pos0 = (
                self.mesh_pos_seq[i][0] * self.node_stats["pos_std"]
                + self.node_stats["pos_mean"]
            )  # denormalize to match original behavior
            g = self.add_edge_features(g, pos0)
            self.graphs.append(g)

        # Edge stats
        if self.split == "train":
            self.edge_stats = self._get_edge_stats()
        else:
            self.edge_stats = load_json("edge_stats.json")

        # Normalize edge features
        for i in range(self.num_samples):
            self.graphs[i].edata["x"] = self.normalize_edge(
                self.graphs[i],
                self.edge_stats["edge_mean"],
                self.edge_stats["edge_std"],
            )

    def __getitem__(self, idx: int):
        graph = self.graphs[idx].clone()
        x, y = self.build_xy(idx)
        graph.ndata["x"] = x
        graph.ndata["y"] = y
        return graph

    # ----- graph-specific helpers (same as your current implementation) -----
    @staticmethod
    def create_graph(src, dst, dtype=torch.int32):
        src_bidirected = np.concatenate([src, dst])
        dst_bidirected = np.concatenate([dst, src])
        graph = dgl.graph((src_bidirected, dst_bidirected), idtype=dtype)
        graph = dgl.to_simple(graph)
        graph = dgl.add_self_loop(graph)
        return graph

    @staticmethod
    def add_edge_features(graph, pos):
        row, col = graph.edges()
        disp = torch.tensor(
            torch.tensor(pos)[row.long()] - torch.tensor(pos)[col.long()],
            dtype=torch.float32,
        )
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
        graph.edata["x"] = torch.cat((disp, disp_norm), dim=1)
        return graph

    def _get_edge_stats(self):
        stats = {
            "edge_mean": 0,
            "edge_meansqr": 0,
        }
        for i in range(self.num_samples):
            stats["edge_mean"] += (
                torch.mean(self.graphs[i].edata["x"], dim=0) / self.num_samples
            )
            stats["edge_meansqr"] += (
                torch.mean(torch.square(self.graphs[i].edata["x"]), dim=0)
                / self.num_samples
            )
        stats["edge_std"] = torch.sqrt(
            stats["edge_meansqr"] - torch.square(stats["edge_mean"])
        )
        stats.pop("edge_meansqr")
        save_json(stats, "edge_stats.json")
        return stats

    @staticmethod
    def normalize_edge(graph, mu, std):
        if (graph.edata["x"].size()[-1] != mu.size()[-1]) or (
            graph.edata["x"].size()[-1] != std.size()[-1]
        ):
            raise AssertionError("Graph edge data must be same size as stats.")
        return (graph.edata["x"] - mu) / std


class CrashPointCloudDataset(CrashBaseDataset):
    """
    Point-cloud version:
    - No graphs or edges.
    - Emits GraphLike with ndata["x"], ndata["y"] for training compatibility.
    - Provides empty edge_stats dict for compatibility.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_stats: Dict[str, Any] = {}

    def __getitem__(self, idx: int):
        x, y = self.build_xy(idx)
        return GraphLike(x=x, y=y)
