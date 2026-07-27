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

"""DrivAerML → FlareJEPA data adapter.

Reuses GeoTransolver's ``TransolverDataPipe`` end-to-end (its ``CAEDataset``
reader, the surface sampling, the surface-normal handling, and the
``mean_std`` field normalisation) so PhysicsJEPA and the GeoTransolver
baseline consume **identical** preprocessed surface data — the fair
head-to-head the DrivAerML comparison requires. This module only re-shapes
one preprocessed sample into the batch dictionary that ``FlareJEPA.forward``
consumes (the SuperWing key convention), so the FlareJEPA training loop and
its unit-tested ``_forward_batch`` path are reused verbatim.

The Transolver datapipe is GPU-resident (``CAEDataset`` moves tensors to the
device as it reads), so this dataset must be driven with ``num_workers=0``.
``superwing_collate`` stacks the per-sample tensors — on GPU here — and the
training loop's ``.to(device)`` is then a no-op.

Key mapping (surface mode), for one sampled cloud of ``N`` points:

    Transolver output            FlareJEPA batch key
    ---------------------------  ----------------------------------------
    embeddings[:, :3]  (pos)     context_pos / target_surface_pos / query_pos
    embeddings[:, 3:6] (normals) context_normals / target_surface_normals /
                                 query_normals
    fields (N, out_dim)          query_target ; the field half of
                                 target_surface_main_feat = [pos, field]
    air_density, stream_velocity gen_params (scaled, cond_dim=2)

``surface_points`` / ``target_encoder_points`` / ``query_points`` select
independent random subsets of the ``resolution`` cloud, so the geometry
encoder, the target encoder, and the decoder query can each run at a
memory-appropriate count while the *query* count is the loss resolution to
match against GeoTransolver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from physicsnemo.datapipes.cae.transolver_datapipe import create_transolver_dataset

# Reuse the exact collate the FlareJEPA loop already uses.
from .superwing import superwing_collate


def _load_surface_factors(
    normalization_dir: str, device: torch.device
) -> dict[str, torch.Tensor]:
    """Load the ``mean_std`` surface-field factors GeoTransolver trains on.

    Identical file and keys as the GeoTransolver ``train.py`` surface path
    (``surface_fields_normalization.npz`` with ``mean`` / ``std``), so the
    field normalisation is bit-for-bit the same across both models.
    """
    norm_file = Path(normalization_dir) / "surface_fields_normalization.npz"
    if not norm_file.exists():
        raise FileNotFoundError(
            f"surface field normalization file not found: {norm_file}. "
            "Point data.normalization_dir at the directory holding "
            "surface_fields_normalization.npz (the transformer_models 'src' dir)."
        )
    data = np.load(norm_file)
    return {
        "mean": torch.from_numpy(data["mean"]).to(device),
        "std": torch.from_numpy(data["std"]).to(device),
    }


def _transolver_cfg(data_cfg: DictConfig, split: str) -> DictConfig:
    """Assemble the config namespace ``create_transolver_dataset`` reads.

    Sourced entirely from the FlareJEPA ``data`` config so the DrivAerML
    knobs live in one place. Surface-only, SDF off, geometry off (the
    FlareJEPA geometry encoder consumes the sampled surface directly).
    """
    return OmegaConf.create(
        {
            "mode": "surface",
            "train": {"data_path": str(data_cfg.train_path)},
            "val": {"data_path": str(data_cfg.val_path)},
            "data_keys": [
                "surface_fields",
                "surface_mesh_centers",
                "surface_normals",
                "surface_areas",
                "stl_faces",
                "stl_centers",
                "stl_coordinates",
                "air_density",
                "stream_velocity",
            ],
            "resolution": int(data_cfg.resolution),
            "preload_depth": int(data_cfg.get("preload_depth", 1)),
            "pin_memory": False,  # GPU-resident reader; pinning is meaningless
            "include_normals": True,
            "include_sdf": False,
            "include_geometry": False,
            "broadcast_global_features": False,
            # Match GeoTransolver's surface defaults (core.yaml) EXACTLY so the
            # geometry is preprocessed identically for a fair head-to-head:
            # centre on the STL CoM and scale by the reference extents. This
            # also keeps coordinates ~unit so FlareJEPA's fixed NeRF-octave
            # FourierPositionalEmbedding stays in its designed range (raw
            # metres would push top bands to ~1e6 -> fp32 noise).
            "translational_invariance": bool(
                data_cfg.get("translational_invariance", True)
            ),
            "scale_invariance": bool(data_cfg.get("scale_invariance", True)),
            "reference_scale": list(
                data_cfg.get("reference_scale", [12.0, 4.5, 3.25])
            ),
            "return_mesh_features": False,
        }
    )


class DrivAerFlareJEPADataset(Dataset):
    """Wrap ``TransolverDataPipe`` and emit FlareJEPA (SuperWing-shaped) items."""

    def __init__(
        self,
        data_cfg: DictConfig,
        *,
        split: str,
        surface_points: int,
        target_encoder_points: int,
        query_points: int,
        cond_scale: tuple[float, float],
        deterministic: bool,
    ):
        phase = "train" if split == "train" else "val"
        self.datapipe = create_transolver_dataset(
            _transolver_cfg(data_cfg, split),
            phase=phase,
            surface_factors=None,  # set below once we know the device
        )
        self.device = self.datapipe.dataset.output_device
        # Wire the mean_std field factors the datapipe scales with. Exposed as
        # self.surface_factors so the training loop can un-normalise fields for
        # GeoTransolver-comparable per-channel rel-L2/L1 validation metrics.
        self.surface_factors = _load_surface_factors(
            str(data_cfg.normalization_dir), self.device
        )
        self.datapipe.config.surface_factors = self.surface_factors
        self.surface_points = int(surface_points)
        self.target_encoder_points = int(target_encoder_points)
        self.query_points = int(query_points)
        self.cond_scale = torch.tensor(
            list(cond_scale), dtype=torch.float32, device=self.device
        )
        self.deterministic = bool(deterministic)

    def __len__(self) -> int:
        return len(self.datapipe)

    def _sample(self, n_pts: int, k: int, gen: torch.Generator | None) -> torch.Tensor:
        """Random ``k`` indices without replacement from ``n_pts`` (capped)."""
        k = min(k, n_pts)
        if self.deterministic:
            # Even stride — reproducible eval coverage of the sampled cloud.
            return torch.linspace(
                0, n_pts - 1, steps=k, device=self.device
            ).round().long()
        return torch.randperm(n_pts, device=self.device, generator=gen)[:k]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # process_data returns un-batched (N, C) GPU tensors for surface mode.
        d = self.datapipe.process_data(self.datapipe.dataset[idx])
        emb = d["embeddings"]  # (N, 6) = [pos(3), normals(3)]
        pos = emb[:, :3]
        normals = emb[:, 3:6]
        fields = d["fields"]  # (N, out_dim), mean_std scaled
        n_pts = pos.shape[0]

        gen = None
        if self.deterministic:
            gen = None  # linspace path ignores the generator
        else:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(idx)

        s_idx = self._sample(n_pts, self.surface_points, gen)
        t_idx = self._sample(n_pts, self.target_encoder_points, gen)
        q_idx = self._sample(n_pts, self.query_points, gen)

        # Conditioning: air_density, stream_velocity. GeoTransolver surface
        # sets broadcast_global_features=false, so these come through as
        # scalar tensors on the processed dict.
        rho = d["air_density"].reshape(-1)[0]
        vel = d["stream_velocity"].reshape(-1)[0]
        gen_params = torch.stack([rho, vel]).to(torch.float32) / self.cond_scale

        out: dict[str, torch.Tensor] = {
            "gen_params": gen_params,
            "context_pos": pos[s_idx],
            "context_normals": normals[s_idx],
            "context_feat": normals[s_idx],  # unused by forward; kept for parity
            "target_surface_pos": pos[t_idx],
            "target_surface_main_feat": torch.cat([pos[t_idx], fields[t_idx]], dim=-1),
            "target_surface_normals": normals[t_idx],
            "target_volume_pos": pos.new_zeros((0, 3)),
            "target_volume_feat": pos.new_zeros((0, 3 + fields.shape[-1])),
            "query_pos": pos[q_idx],
            "query_normals": normals[q_idx],
            "query_sdf": pos.new_zeros((q_idx.shape[0], 1)),
            "query_target": fields[q_idx],
        }
        return out


def build_drivaer_flarejepa_loader(
    data_cfg: DictConfig,
    *,
    split: str,
    batch_size: int,
    shuffle: bool,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[DataLoader, Any]:
    """DrivAerML loader mirroring ``_build_loader`` (num_workers forced to 0)."""
    deterministic = split != "train"
    dataset = DrivAerFlareJEPADataset(
        data_cfg,
        split=split,
        surface_points=int(data_cfg.surface_points),
        target_encoder_points=int(data_cfg.target_encoder_points),
        query_points=int(data_cfg.query_points),
        cond_scale=tuple(data_cfg.get("cond_scale", [1.2, 40.0])),
        deterministic=deterministic,
    )
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=bool(shuffle),
            drop_last=(split == "train"),
        )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=0,  # CAEDataset yields GPU tensors; workers would break it
        pin_memory=False,
        collate_fn=superwing_collate,
        drop_last=False,
    )
    return loader, sampler
