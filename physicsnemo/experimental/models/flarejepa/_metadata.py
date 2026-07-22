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

r"""Shared :class:`physicsnemo.core.meta.ModelMetaData` for the FlareJEPA stack.

Every FlareJEPA submodule inherits from
:class:`physicsnemo.core.module.Module` so the top-level model serialises
cleanly via ``Module.save()``. They all share a single ``FlareJEPAMetaData``
because the capability flags are identical across the stack. Unlike AeroJEPA,
FlareJEPA targets fixed input shapes, so CUDA graphs / compile support is a
goal (Phase 4); the flags stay conservative until that phase lands.
"""

from __future__ import annotations

from dataclasses import dataclass

from physicsnemo.core.meta import ModelMetaData


@dataclass
class FlareJEPAMetaData(ModelMetaData):
    r"""Meta-data for the :class:`FlareJEPA` model and its submodules."""

    # Optimization
    jit: bool = False
    cuda_graphs: bool = False  # Phase 4 target; flip once verified
    amp: bool = True
    bf16: bool = True
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False
