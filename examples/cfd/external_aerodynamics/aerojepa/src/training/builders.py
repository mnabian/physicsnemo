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

r"""Hydra-friendly builders for the AeroJEPA training stack.

These helpers keep ``train.py`` short: they delegate optimizer
construction to ``hydra.utils.instantiate`` and provide the small
linear-warmup helper used to ramp the JEPA loss weights at the start
of training.
"""

from __future__ import annotations

from typing import Any

import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig

from .combined_optimizer import CombinedOptimizer


def build_optimizer(
    model: torch.nn.Module,
    optimizer_cfg: DictConfig | dict[str, Any],
    *,
    extra_params: list[torch.nn.Parameter] | None = None,
) -> torch.optim.Optimizer:
    r"""Instantiate an optimizer from a Hydra config.

    Calls :func:`hydra.utils.instantiate` on ``optimizer_cfg`` with
    ``params=model.parameters() (+ extra_params)`` injected at call
    time. The config is expected to carry a ``_target_`` pointing at
    an optimizer class (e.g. ``torch.optim.AdamW``) and the optimizer's
    own kwargs (``lr``, ``weight_decay``, ``betas`` …).

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters are optimised.
    optimizer_cfg : DictConfig or dict
        Hydra group with ``_target_`` and optimizer kwargs.
    extra_params : list of torch.nn.Parameter, optional
        Additional parameter tensors to optimise alongside the model
        (used by prototype / anchor heads that live outside the main
        module). Default ``None``.

    Returns
    -------
    torch.optim.Optimizer
        The instantiated optimizer.
    """
    # Recipe-only keys that must NEVER reach an optimizer ctor.
    _RECIPE_KEYS = ("_target_", "muon_2d", "no_decay_norms_and_gains")
    if bool(optimizer_cfg.get("muon_2d", False)):
        # transformer_models reference recipe: Muon on all 2-D weights
        # (adjust_lr_fn="match_rms_adamw"), the configured optimizer
        # (AdamW) on everything else. Same lr/weight_decay on both.
        params_all = list(model.parameters())
        if extra_params:
            params_all.extend(extra_params)
        muon_params = [p for p in params_all if p.ndim == 2]
        other_params = [p for p in params_all if p.ndim != 2]
        kwargs = {k: v for k, v in dict(optimizer_cfg).items()
                  if k not in _RECIPE_KEYS}
        adamw = get_class(str(optimizer_cfg["_target_"]))(
            other_params, **kwargs
        )
        muon = torch.optim.Muon(
            muon_params,
            lr=float(optimizer_cfg["lr"]),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
            adjust_lr_fn="match_rms_adamw",
        )
        return CombinedOptimizer([muon, adamw])
    if bool(optimizer_cfg.get("no_decay_norms_and_gains", False)):
        # Gap-audit finding 10: a single AdamW group applies weight decay to
        # norm gains, AdaLN projections/gates, learned queries, q_global and
        # logit temperatures — the exact parameters that control attention
        # sharpness and conditioning strength. Standard practice: no decay
        # for 1-D params (biases, gains, gates, scales) and named embedding-
        # like params; decay for the rest.
        no_decay_names = ("q_global", "slot_queries", "registers",
                          "register_queries", "logit_scale", "log_tau",
                          "state_mixing")
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if p.ndim <= 1 or any(k in name for k in no_decay_names):
                no_decay.append(p)
            else:
                decay.append(p)
        params = [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    else:
        params = list(model.parameters())
    grouped = params and isinstance(params[0], dict)
    if extra_params:
        if grouped:
            params.append({"params": list(extra_params)})
        else:
            params.extend(extra_params)
    if grouped:
        # instantiate() would recursively wrap the param-group dicts in
        # DictConfig (which torch.optim rejects), so resolve the target
        # class and call it directly for the grouped path.
        kwargs = {k: v for k, v in dict(optimizer_cfg).items()
                  if k not in _RECIPE_KEYS}
        return get_class(str(optimizer_cfg["_target_"]))(params, **kwargs)
    # Plain path: strip recipe-only keys (keeping _target_ for instantiate).
    cfg = {k: v for k, v in dict(optimizer_cfg).items()
           if k not in ("muon_2d", "no_decay_norms_and_gains")}
    return instantiate(cfg, params=params)


def linear_warmup_weight(
    target_weight: float,
    warmup_epochs: float,
    current_epoch: float,
) -> float:
    r"""Linearly ramp a loss weight from ``0`` to ``target_weight``.

    The JEPA training loss combines several terms (reconstruction,
    latent MSE / cosine, SIGReg). The non-reconstruction terms benefit
    from a short linear warmup so the predictor first learns to match
    the target latents at small scale before the regulariser kicks in.

    Parameters
    ----------
    target_weight : float
        Final weight after warmup completes.
    warmup_epochs : float
        Length of the warmup ramp in epochs. ``<= 0`` means no warmup
        (returns ``target_weight`` immediately).
    current_epoch : float
        Current epoch counter (fractional epochs are fine).

    Returns
    -------
    float
        Weight at ``current_epoch``.
    """
    if float(warmup_epochs) <= 0.0:
        return float(target_weight)
    progress = max(0.0, min(1.0, float(current_epoch) / float(warmup_epochs)))
    return float(target_weight) * progress
