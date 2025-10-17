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
import hydra
from hydra.utils import to_absolute_path
from dgl.dataloading import GraphDataLoader
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.nn.parallel import DistributedDataParallel
from physicsnemo.models.meshgraphnet import MeshGraphNet
from crash_dataset import CrashDataset
from physicsnemo.launch.logging import PythonLogger
from physicsnemo.launch.utils import load_checkpoint
import pyvista as pv

from mgnrollout import MGNRollout
import tempfile, shutil, os


# SIMPLIFIED: This function now saves results for a single sample at a time.
def save_vtp_with_predictions_and_exact(
    rollout, vtp_dir, output_dir_pred, output_dir_exact
):
    os.makedirs(output_dir_pred, exist_ok=True)
    os.makedirs(output_dir_exact, exist_ok=True)

    # The dataloader for this rollout only contains one sample, so we loop through it.
    # rollout.preds will have the shape [[preds_t0, preds_t1, ...]]
    for testi, (preds, exacts) in enumerate(zip(rollout.preds, rollout.exacts)):
        # Create a subfolder for the batch item, e.g., "test_000"
        test_pred_dir = os.path.join(output_dir_pred, f"test_{testi:03d}")
        test_exact_dir = os.path.join(output_dir_exact, f"test_{testi:03d}")
        os.makedirs(test_pred_dir, exist_ok=True)
        os.makedirs(test_exact_dir, exist_ok=True)

        for i, (pred, exact) in enumerate(zip(preds, exacts)):
            vtp_file = os.path.join(vtp_dir, f"frame_{i:03d}.vtp")
            if not os.path.exists(vtp_file):
                print(f"Warning: {vtp_file} does not exist, skipping.")
                continue

            # Save predicted
            mesh_pred = pv.read(vtp_file)
            pred_np = pred.cpu().numpy() if hasattr(pred, "cpu") else pred
            mesh_pred.points = pred_np
            mesh_pred.point_data["prediction"] = pred_np
            out_file_pred = os.path.join(test_pred_dir, f"frame_{i:03d}_pred.vtp")
            mesh_pred.save(out_file_pred)

            # Save exact
            mesh_exact = pv.read(vtp_file)
            exact_np = exact.cpu().numpy() if hasattr(exact, "cpu") else exact
            mesh_exact.points = exact_np
            mesh_exact.point_data["exact"] = exact_np
            mesh_exact.point_data["difference"] = pred_np - exact_np
            out_file_exact = os.path.join(test_exact_dir, f"frame_{i:03d}_exact.vtp")
            mesh_exact.save(out_file_exact)


class MGNTest:
    # This class is now designed to handle one sample at a time.
    def __init__(self, cfg: DictConfig, logger: PythonLogger, model=None):
        self.num_time_steps = cfg.num_time_steps
        self.num_output_features = cfg.num_output_features
        self.rollout_steps = cfg.num_time_steps - 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # The dataset is now created for a SINGLE sample directory
        self.dataset = CrashDataset(
            name="crash_test",
            data_dir=to_absolute_path(cfg.raw_data_dir_test),
            split="test",
            num_samples=1,
            num_steps=cfg.num_time_steps,
            write_vtp=cfg.write_vtp_in_inference,
        )

        self.data_stats = dict(
            node={k: v.to(self.device) for k, v in self.dataset.node_stats.items()},
            edge={k: v.to(self.device) for k, v in self.dataset.edge_stats.items()},
        )

        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        # Use the pre-loaded model passed from the main loop
        # if model is not None:
        #     self.model = model
        # else:
        # Fallback to load model here if not provided (not used in the main loop)
        self.model = MGNRollout(
            functional_dim=5,
            out_dim=cfg.num_output_features,
            embedding_dim=3,
            slice_num=128,
            n_layers=8,
            unified_pos=False,
            structured_shape=None,
            use_te=False,
            time_input=False,
            rollout_steps=self.rollout_steps,
        )
        # Abridged for brevity...
        self.model = self.model.to(self.device)
        if hasattr(cfg, "ckpt_path"):
            load_checkpoint(
                to_absolute_path(cfg.ckpt_path), models=self.model, device=self.device
            )

        self.model.eval()

    @torch.no_grad()
    def predict(self):
        # Simplified: no need to track sample names inside the class
        self.preds, self.exacts, self.graphs = [], [], []
        stats = {
            key: value.to(self.device) for key, value in self.dataset.node_stats.items()
        }

        # The dataloader will yield the single graph for the current sample
        for i, graph in enumerate(self.dataloader):
            graph = graph.to(self.device)
            T_pred = self.rollout_steps

            num_nodes = graph.ndata["y"].shape[0]
            exact_sequence = graph.ndata["y"].view(num_nodes, T_pred, 3).transpose(0, 1)

            model_module = (
                self.model.module
                if isinstance(self.model, DistributedDataParallel)
                else self.model
            )

            pred_sequence = model_module.forward_rollout(
                graph.ndata["x"],
                self.data_stats,
            )

            # Store results for this sample
            current_preds, current_exacts = [], []
            for t in range(T_pred):
                exact_denorm = self.dataset.denormalize(
                    exact_sequence[t], stats["pos_mean"], stats["pos_std"]
                )
                pred_denorm = self.dataset.denormalize(
                    pred_sequence[t], stats["pos_mean"], stats["pos_std"]
                )
                current_preds.append(pred_denorm)
                current_exacts.append(exact_denorm)

            self.preds.append(current_preds)
            self.exacts.append(current_exacts)
            self.graphs.append([graph] * T_pred)

        self.preds = [[pi.cpu() for pi in p_list] for p_list in self.preds]
        self.exacts = [[ei.cpu() for ei in e_list] for e_list in self.exacts]
        self.graphs = [[gi.cpu() for gi in g_list] for g_list in self.graphs]


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")
    logger.file_logging()
    logger.info("Batch inference started...")

    # ===== 1. Load Model Once =====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device for model.")

    # ===== 2. Get Parent Directory and Find Samples =====
    parent_dir = to_absolute_path(cfg.raw_data_dir_test)
    if not os.path.isdir(parent_dir):
        logger.error(f"Parent directory not found: {parent_dir}")
        return

    try:
        sample_paths = [d.path for d in os.scandir(parent_dir) if d.is_dir()]
    except OSError as e:
        logger.error(f"Error reading directories from {parent_dir}: {e}")
        return

    logger.info(f"Found {len(sample_paths)} samples in {parent_dir}")

    # ===== 3. Loop Through Each Sample =====
    for sample_path in sample_paths:
        sample_name = os.path.basename(sample_path)
        logger.info(f"--- Processing sample: {sample_name} ---")

        with tempfile.TemporaryDirectory() as tmpdir:
            os.symlink(sample_path, os.path.join(tmpdir, os.path.basename(sample_path)))
            sample_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))
            sample_cfg.raw_data_dir_test = (
                tmpdir  # points to a parent with exactly one run inside
            )
            sample_cfg.num_test_samples = 1
            rollout = MGNTest(sample_cfg, logger, model=None)
            rollout.predict()
            if sample_cfg.write_vtp_in_inference:
                vtp_dir = "output_" + sample_name
                out_pred = os.path.join(
                    to_absolute_path(
                        sample_cfg.get("output_dir_pred", "./predicted_vtps")
                    ),
                    sample_name,
                )
                out_exact = os.path.join(
                    to_absolute_path(
                        sample_cfg.get("output_dir_exact", "./exact_vtps")
                    ),
                    sample_name,
                )
                save_vtp_with_predictions_and_exact(
                    rollout, vtp_dir, out_pred, out_exact
                )

    logger.info("Batch inference finished successfully.")


if __name__ == "__main__":
    main()
