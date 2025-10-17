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

import os, sys

sys.path.insert(0, os.path.dirname(__file__))
import time
import hydra

from hydra.utils import to_absolute_path
import torch
from tqdm import tqdm

from omegaconf import DictConfig
from omegaconf import OmegaConf
import logging

from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from hydra.utils import instantiate
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

import os

os.makedirs(os.path.expanduser("~/.dgl"), exist_ok=True)

from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """Trainer for the crash model."""

    def __init__(self, cfg: DictConfig, logger0: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.cfg = cfg
        self.rollout_steps = cfg.training.num_time_steps - 1
        self.amp = cfg.training.amp

        dataset = instantiate(
            cfg.datapipe,
            name="crash_train",
            split="train",
            logger=logger0,
        )
        self.data_stats = dict(
            node={k: v.to(self.dist.device) for k, v in dataset.node_stats.items()},
            edge={
                k: v.to(self.dist.device)
                for k, v in getattr(dataset, "edge_stats", {}).items()
            },
            thickness={
                k: v.to(self.dist.device) for k, v in dataset.thickness_stats.items()
            },
        )
        if self.dist.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.dist.world_size,
                rank=self.dist.rank,
                shuffle=True,
            )
        else:
            sampler = None

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=(sampler is None),
            drop_last=True,
            pin_memory=True,
            num_workers=cfg.training.num_dataloader_workers,
            sampler=sampler,
            collate_fn=lambda batch: batch[0],
        )
        self.sampler = sampler

        # instantiate the model
        self.model = instantiate(cfg.model)
        # Note: Hydra's instantiate() may reset the global logging level to WARNING,
        # which suppresses INFO messages. Restore it to INFO.
        logging.getLogger().setLevel(logging.INFO)

        # enable train mode
        self.model.to(self.dist.device)
        self.model.train()

        self.criterion = torch.nn.MSELoss()

        self.optimizer = None
        try:
            if cfg.training.use_apex:
                from apex.optimizers import FusedAdam

                self.optimizer = FusedAdam(
                    self.model.parameters(), lr=cfg.training.start_lr
                )
        except ImportError:
            logger0.warning(
                "NVIDIA Apex (https://github.com/nvidia/apex) is not installed, "
                "FusedAdam optimizer will not be used."
            )
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=cfg.training.start_lr
            )
        logger0.info(f"Using {self.optimizer.__class__.__name__} optimizer")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.end_lr
        )
        self.scaler = GradScaler()

        # load checkpoint
        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.training.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

        if self.dist.rank == 0:
            self.writer = SummaryWriter(log_dir=cfg.training.tensorboard_log_dir)

    def train(self, graph, epoch):
        self.optimizer.zero_grad()
        loss = self.forward(graph, epoch)
        self.backward(loss)
        return loss

    def forward(self, graph, epoch):
        # forward pass
        with autocast(enabled=self.amp):
            # # Build per-step conditioning sequence [T, N, 1]
            T = self.rollout_steps

            # Predict rollout: [T, N, Fo]
            pred = self.model(
                node_features=graph.ndata["x"],  # [N, Fn] at t=0
                data_stats=self.data_stats,
            )

            # Target is currently [N, T*Fo] -> reshape to [T, N, Fo]
            target_flat = graph.ndata["y"]  # [N, T*Fo]
            N = target_flat.size(0)
            Fo = 3  # self.cfg.num_output_features
            assert target_flat.size(1) == T * Fo, (
                f"target dim {target_flat.size(1)} != T*Fo {T * Fo}"
            )
            target = (
                target_flat.view(N, T, Fo).transpose(0, 1).contiguous()
            )  # [T, N, Fo]

            # Loss
            loss = self.criterion(pred, target)
            return loss

    def backward(self, loss):
        # backward pass
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("main")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    logger0.file_logging()

    trainer = Trainer(cfg, logger0)
    start = time.time()
    logger0.info("Training started...")
    for epoch in range(trainer.epoch_init, cfg.training.epochs):
        if trainer.sampler is not None:
            trainer.sampler.set_epoch(epoch)
        start = time.time()
        # Wrap the dataloader with tqdm and add description with epoch info
        progress_bar = tqdm(
            trainer.dataloader,
            desc=f"Epoch {epoch + 1}/{cfg.training.epochs}",
            leave=False,
            disable=True,
        )

        total_loss = 0.0
        num_batches = 0

        for graph in progress_bar:
            graph = graph.to(dist.device)
            loss = trainer.train(graph, epoch)
            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix(loss=f"{loss.item():.3e}")
            del graph
            torch.cuda.empty_cache()
        trainer.scheduler.step()

        avg_loss = total_loss / num_batches if num_batches > 0 else float("nan")
        logger0.info(
            f"epoch: {epoch + 1}, avg_loss: {avg_loss:10.3e}, lr: {trainer.optimizer.param_groups[0]['lr']:.3e}, time per epoch: {(time.time() - start):10.3e}"
        )
        if dist.rank == 0:
            trainer.writer.add_scalar("loss", avg_loss, epoch)
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            trainer.writer.add_scalar("learning_rate", current_lr, epoch)

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                cfg.training.ckpt_path,
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch + 1,
            )
            logger.info(f"Saved model on rank {dist.rank}")

        torch.cuda.empty_cache()
        start = time.time()
    logger0.info("Training completed!")
    if dist.rank == 0:
        trainer.writer.close()


if __name__ == "__main__":
    main()
