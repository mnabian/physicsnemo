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
"""Time-conditional DeFormer (latent-preserving refinement) and its GeoFLARE baseline."""

import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data

from physicsnemo.experimental.models.meshtransolver.conditional_norm import (
    ConditionalLayerNorm,
    conditional_layer_norms,
)

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))
CONF = CRASH_DIR / "conf"

from datapipe import SimSample  # noqa: E402
from local_neighborhoods import (  # noqa: E402
    LocalNeighborhoodCache,
    LocalNeighborhoodData,
    knn_edge_features,
    with_self_slot,
)
from rollout import GeoTransolverTimeConditional, TimeEmbedding  # noqa: E402
from surface_topology import (  # noqa: E402
    SurfaceContactData,
    surface_one_ring_exclusions,
)
from time_conditional import (  # noqa: E402
    MeshGeoFLARETimeConditional,
    TimeConditionalOutput,
)


def _compose(name):
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose(
            config_name=name,
            overrides=[
                "training.raw_data_dir=/fixtures/train",
                "training.raw_data_dir_validation=/fixtures/validation",
                "inference.raw_data_dir_test=/fixtures/test",
            ],
        )


def _sample(n=9, k=3, time=0.0, with_graph=True):
    torch.manual_seed(5)
    coords = torch.randn(n, 3) * 0.3
    source = torch.arange(n - 1)
    edges = torch.stack(
        (torch.cat((source, source + 1)), torch.cat((source + 1, source)))
    )
    graph = None
    if with_graph:
        graph = LocalNeighborhoodData(
            **Data(
                edge_index=edges, edge_attr=torch.randn(2 * (n - 1), 4), num_nodes=n
            ).to_dict()
        )
        neighborhood = LocalNeighborhoodCache().get(coords, edges, k, "geodesic")
        graph.knn_index, graph.knn_mask = with_self_slot(neighborhood)
        graph.knn_edge_attr = knn_edge_features(coords, graph.knn_index)
        graph.knn_metric = "geodesic"
    features = {
        "coords": coords,
        "features": torch.randn(n, 1),
        "time": torch.tensor(time),
    }
    target = coords + 0.05 * torch.randn(n, 3)
    sample = SimSample(features, target, graph=graph)
    stats = {"node": {}}
    return sample, stats


def _tc_model(**overrides):
    kwargs = dict(
        num_time_steps=5,
        functional_dim=3 + 1 + 8,
        out_dim=3,
        geometry_dim=3,
        global_dim=8,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        n_layers=1,
        use_te=False,
        attention_type="GALE_FA",
        time_embedding="fourier",
        time_fourier_bands=3,
        time_dim=8,
        time_as_global=True,
        input_dim_edges=4,
        mesh_hidden_dim=16,
        num_pre_processor_layers=1,
        num_post_processor_layers=1,
        mesh_context_fusion="pre_post",
        mesh_pre_residual_gate_init=0.0,
        mesh_post_residual_gate_init=1.0,
        use_contact=False,
        local_processor="knn_attention",
        local_knn_k=3,
        local_neighborhood="geodesic",
        local_num_heads=4,
        local_edge_dim=8,
        local_attention_backend="torch",
        local_value_norm=False,
        local_difference_values=True,
        local_update="concat_mlp",
        local_split_key_value=True,
    )
    kwargs.update(overrides)
    torch.manual_seed(1)
    return MeshGeoFLARETimeConditional(**kwargs)


def test_time_embedding_shapes_and_smoothness():
    scalar = TimeEmbedding("scalar")
    assert scalar.dim == 1 and scalar(torch.tensor(0.25)).tolist() == [[0.25]]
    fourier = TimeEmbedding("fourier", bands=4, dim=12)
    codes = torch.stack([fourier(torch.tensor(q)).squeeze(0) for q in (0.0, 0.01, 0.5)])
    assert codes.shape == (3, 12)
    assert (codes[0] - codes[1]).norm() < (codes[0] - codes[2]).norm()
    with pytest.raises(ValueError):
        TimeEmbedding("linear")


def test_recipes_compose_and_match_each_other():
    base = _compose("crash_geoflare_time_conditional")
    tc = _compose("crash_deformer_knn_time_conditional")
    for cfg in (base, tc):
        assert cfg.datapipe.sample_type == "one_time_step"
        assert cfg.datapipe.initial_history_steps == 1
        assert cfg.training.num_time_steps == 26 and cfg.training.epochs == 100
        assert cfg.training.validation_freq == 5 and cfg.training.optimizer == "muon"
        assert (
            cfg.model.time_embedding == "fourier" and cfg.model.time_as_global is True
        )
        assert cfg.model.functional_dim == 36 and cfg.model.global_dim == 32
        assert cfg.model.attention_type == "GALE_FA" and cfg.model.n_hidden == 256
    assert base.model._target_ == "rollout.GeoTransolverTimeConditional"
    assert base.datapipe._target_ == "datapipe.CrashPointCloudDataset"
    assert base.training.coarse_loss_weight == 0.0
    assert tc.model._target_ == "time_conditional.MeshGeoFLARETimeConditional"
    assert tc.datapipe._target_ == "datapipe.CrashGraphDataset"
    assert tc.training.coarse_loss_weight == 0.1
    assert (
        tc.model.local_processor == "knn_attention"
        and tc.model.local_update == "concat_mlp"
    )
    assert tc.model.activation_checkpointing is False
    assert (
        tc.model.num_post_processor_layers == 1
        and tc.model.mesh_post_residual_gate_init == 1.0
    )
    for key in ("epochs", "start_lr", "end_lr", "validation_freq", "seed", "amp_dtype"):
        assert base.training[key] == tc.training[key], key
    resolved = OmegaConf.to_container(tc, resolve=True)
    assert resolved["datapipe"]["local_knn_k"] == 32
    assert resolved["datapipe"]["local_knn_metric"] == "geodesic"


def test_tc_model_starts_undeformed_trains_and_rolls_out():
    sample, stats = _sample(time=0.5)
    assert "Din=5" in repr(sample)  # coords + one feature + the scalar time
    model = _tc_model().train()
    assert model.rollout_steps == 4
    # Zero-initialized heads: both outputs equal the initial geometry.
    with torch.no_grad():
        output = model(sample=sample, data_stats=stats, return_auxiliary=True)
    assert isinstance(output, TimeConditionalOutput)
    torch.testing.assert_close(output.prediction, sample.node_features["coords"])
    torch.testing.assert_close(output.coarse_prediction, sample.node_features["coords"])
    # Zero-initialized heads mean the first update only moves the heads; the
    # processors, backbone and time code receive gradients once the heads and
    # the pre gate are nonzero (third step).
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(3):
        optimizer.zero_grad()
        output = model(sample=sample, data_stats=stats, return_auxiliary=True)
        loss = torch.nn.functional.mse_loss(
            output.prediction, sample.node_target
        ) + 0.1 * (
            torch.nn.functional.mse_loss(output.coarse_prediction, sample.node_target)
        )
        loss.backward()
        grads = {
            n
            for n, p in model.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        }
        if step == 0:
            assert any(n.startswith("coarse_decoder") for n in grads)
            assert not any(n.startswith("pre_processor") for n in grads)
        optimizer.step()
    for prefix in (
        "coarse_decoder",
        "refinement_processor",
        "post_processor",
        "time_embed",
        "pre_processor",
        "blocks",
    ):
        assert any(n.startswith(prefix) for n in grads), prefix
    # Stage B has independent parameters from stage A.
    a = dict(model.post_processor.named_parameters())
    b = dict(model.refinement_processor.named_parameters())
    assert a.keys() == b.keys() and not all(torch.equal(a[k], b[k]) for k in a)
    # Evaluation queries every future frame independently and restores the sample.
    model.eval()
    original_time = sample.node_features["time"].clone()
    with torch.no_grad():
        rollout = model(sample=sample, data_stats=stats)
        singles = []
        for k in range(4):
            sample.node_features["time"] = torch.tensor(k / 4)
            singles.append(model._at_time(sample, stats).prediction)
        sample.node_features["time"] = original_time
    assert rollout.shape == (9, 4, 3)
    torch.testing.assert_close(rollout, torch.stack(singles, dim=1))
    torch.testing.assert_close(sample.node_features["time"], original_time)
    with pytest.raises(ValueError, match="future-frame range"):
        sample.node_features["time"] = torch.tensor(1.5)
        model.train()(sample=sample, data_stats=stats)


def test_tc_model_rejects_wrong_output_width_and_te():
    with pytest.raises(ValueError, match="positions only"):
        _tc_model(out_dim=5)
    with pytest.raises(ValueError, match="use_te=False"):
        _tc_model(use_te=True)


@pytest.mark.parametrize("processor", ["knn_mpnn", "mpnn"])
def test_tc_model_works_with_message_passing_processors(processor):
    sample, stats = _sample(time=0.25)
    model = _tc_model(local_processor=processor).train()
    output = model(sample=sample, data_stats=stats, return_auxiliary=True)
    assert output.prediction.shape == (9, 3) and torch.isfinite(output.prediction).all()
    output.prediction.square().sum().backward()
    assert model.edge_encoder is not None


def test_geoflare_time_conditional_with_fourier_time_and_global_token():
    sample, stats = _sample(time=0.75, with_graph=False)
    torch.manual_seed(2)
    model = GeoTransolverTimeConditional(
        num_time_steps=5,
        functional_dim=3 + 1 + 8,
        out_dim=3,
        geometry_dim=3,
        global_dim=8,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        n_layers=1,
        use_te=False,
        attention_type="GALE_FA",
        time_embedding="fourier",
        time_fourier_bands=3,
        time_dim=8,
        time_as_global=True,
    ).train()
    prediction = model(sample=sample, data_stats=stats)
    assert prediction.shape == (9, 3)
    prediction.square().sum().backward()
    assert model.time_embed.mlp[0].weight.grad is not None
    model.eval()
    with torch.no_grad():
        rollout = model(sample=sample, data_stats=stats)
    assert rollout.shape == (9, 4, 3)
    # The legacy scalar path still composes: one time channel, no global token.
    legacy = GeoTransolverTimeConditional(
        num_time_steps=5,
        functional_dim=5,
        out_dim=3,
        geometry_dim=3,
        global_dim=None,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        n_layers=1,
        use_te=False,
        attention_type="GALE_FA",
    ).train()
    assert legacy(sample=sample, data_stats=stats).shape == (9, 3)


TINY = dict(
    n_hidden=16,
    n_head=4,
    slice_num=4,
    n_layers=1,
    functional_dim=12,
    global_dim=8,
    time_dim=8,
    time_fourier_bands=3,
    mesh_hidden_dim=16,
    local_num_heads=4,
    local_edge_dim=8,
    local_knn_k=3,
    local_attention_backend="torch",
    contact_search_implementation="torch",
)


@pytest.mark.parametrize(
    "name",
    [
        "crash_geoflare_time_conditional",
        "crash_deformer_knn_time_conditional",
        "crash_deformer_knn_time_conditional_adaln",
        "crash_deformer_knn_time_conditional_contact",
        "crash_deformer_knn_time_conditional_single_stage",
        "crash_deformer_knn_time_conditional_contact_correction",
    ],
)
def test_time_conditional_model_configs_instantiate(name):
    """Every key of the model group must be accepted by the model class."""
    from hydra.utils import instantiate

    cfg = _compose(name)
    overrides = {k: v for k, v in TINY.items() if k in cfg.model}
    model = instantiate(cfg.model, **overrides)
    assert model.rollout_steps == 25
    if name.endswith("adaln"):
        assert model.norm_conditioning == "time" and model.num_conditional_norms > 0
    if name.endswith("contact"):
        assert model.provisional_contact and model.node_contact_builder is not None
        assert model.contact_block is None  # the backbone stays contact-free
    if name.endswith("single_stage"):
        assert model.two_stage is False and model.coarse_decoder is None
    if name.endswith("contact_correction"):
        assert len(model.refinement_processor.blocks) == 2
        assert len(model.post_processor.blocks) == 1
        assert model.coarse_feedback is not None and model.provisional_contact


# ----- GenCast-style norm conditioning (adaLN-zero on the time code) -----


def _excite(model):
    """Zero-initialised heads block every upstream gradient; give them weights."""
    nn.init.normal_(model.coarse_decoder[-1].weight, std=0.1)
    last = [m for m in model.post_node_decoder.modules() if isinstance(m, nn.Linear)][
        -1
    ]
    nn.init.normal_(last.weight, std=0.1)
    for module in conditional_layer_norms(model):
        nn.init.normal_(module.modulation.weight, std=0.05)


def test_adaln_model_is_function_preserving_and_trains_through_the_norms():
    sample, stats = _sample(time=0.5)
    plain = _tc_model().train()
    ada = _tc_model(norm_conditioning="time").train()
    # Every LayerNorm is replaced: FLARE block, output head, both post stages
    # and the refinement copy, context encoder, FiLM adapters, coarse head.
    assert ada.num_conditional_norms == 17
    assert not any(isinstance(m, nn.LayerNorm) for m in ada.modules())
    names = {n for n, m in ada.named_modules() if isinstance(m, ConditionalLayerNorm)}
    assert {
        "blocks.0.ln_1",
        "blocks.0.ln_mlp1.0",
        "coarse_decoder.0",
        "mesh_context_encoder.output_norm",
        "pre_context_film.latent_norm",
        "post_processor.blocks.0.attention_norm",
        "refinement_processor.blocks.0.update_mlp.3",
        "refinement_context_film.latent_norm",
    } <= names
    # Same function as the plain model at initialisation once the shared
    # parameters are copied: the modulation starts at zero.
    missing, unexpected = ada.load_state_dict(plain.state_dict(), strict=False)
    assert missing and all(".modulation." in key for key in missing)
    assert unexpected and all(key.rsplit(".", 1)[0] in names for key in unexpected)
    with torch.no_grad():
        reference = plain(sample=sample, data_stats=stats, return_auxiliary=True)
        output = ada(sample=sample, data_stats=stats, return_auxiliary=True)
    torch.testing.assert_close(output.prediction, reference.prediction)
    torch.testing.assert_close(output.coarse_prediction, reference.coarse_prediction)
    # With non-zero heads, gradients reach the modulation of the norms on the
    # active path and the time code through them.
    _excite(ada)
    output = ada(sample=sample, data_stats=stats, return_auxiliary=True)
    (
        output.prediction.square().sum() + output.coarse_prediction.square().sum()
    ).backward()
    for prefix in (
        "blocks.0.ln_1",
        "blocks.0.ln_mlp1.0",
        "coarse_decoder.0",
        "post_processor.blocks.0.attention_norm",
        "refinement_processor.blocks.0.update_mlp.3",
    ):
        grad = ada.get_submodule(prefix).modulation.weight.grad
        assert grad is not None and grad.abs().sum() > 0, prefix
    assert ada.time_embed.mlp[0].weight.grad.abs().sum() > 0
    # Evaluation sets one condition per queried frame and restores the time.
    ada.eval()
    original = sample.node_features["time"].clone()
    with torch.no_grad():
        rollout = ada(sample=sample, data_stats=stats)
        sample.node_features["time"] = torch.tensor(2 / 4)
        single = ada._at_time(sample, stats).prediction
        sample.node_features["time"] = original
    assert rollout.shape == (9, 4, 3)
    torch.testing.assert_close(rollout[:, 2], single)
    torch.testing.assert_close(sample.node_features["time"], original)
    with pytest.raises(ValueError, match="norm_conditioning"):
        _tc_model(norm_conditioning="film")


def test_adaln_gradients_match_under_activation_checkpointing():
    sample, stats = _sample(time=0.25)
    reference = _tc_model(norm_conditioning="time")
    _excite(reference)
    checkpointed = _tc_model(
        norm_conditioning="time",
        activation_checkpointing=True,
        checkpointing_ratio=1.0,
        activation_checkpointing_components=["blocks", "context"],
    )
    checkpointed.load_state_dict(reference.state_dict())
    assert checkpointed._activation_checkpointing_enabled
    grads = []
    for model in (reference, checkpointed):
        model.train()
        model.zero_grad()
        output = model(sample=sample, data_stats=stats, return_auxiliary=True)
        output.prediction.square().sum().backward()
        grads.append(
            {
                n: p.grad.clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        )
    assert grads[0].keys() == grads[1].keys()
    for key, value in grads[0].items():
        torch.testing.assert_close(grads[1][key], value, msg=key)


def test_adaln_recipe_differs_from_its_control_only_by_norm_conditioning():
    base = _compose("crash_deformer_knn_time_conditional")
    ada = _compose("crash_deformer_knn_time_conditional_adaln")
    assert (
        ada.model.norm_conditioning == "time" and "norm_conditioning" not in base.model
    )
    assert list(ada.training.muon_exclude_patterns) == ["modulation"]
    assert set(ada.training) - set(base.training) == {"muon_exclude_patterns"}
    for key in base.training:
        assert base.training[key] == ada.training[key], key
    ada_model = OmegaConf.to_container(ada.model, resolve=True)
    ada_model.pop("norm_conditioning")
    assert ada_model == OmegaConf.to_container(base.model, resolve=True)
    assert OmegaConf.to_container(ada.datapipe, resolve=True) == OmegaConf.to_container(
        base.datapipe, resolve=True
    )
    assert ada.experiment_name != base.experiment_name


@pytest.mark.skipif(
    not hasattr(torch.optim, "Muon"), reason="Muon needs PyTorch >= 2.9"
)
def test_muon_exclude_patterns_route_modulation_to_adamw():
    from utils import build_muon_optimizer

    def partition(patterns, min_dim):
        model = _tc_model(norm_conditioning="time")
        cfg = OmegaConf.create(
            {
                "training": {
                    "start_lr": 1e-3,
                    "muon_min_dim": min_dim,
                    "muon_exclude_patterns": patterns,
                }
            }
        )
        optimizer = build_muon_optimizer(model, cfg)
        parts = getattr(optimizer, "optimizers", [optimizer])
        by_type = {}
        for part in parts:
            for group in part.param_groups:
                by_type.setdefault(type(part).__name__, set()).update(
                    id(p) for p in group["params"]
                )
        modulation = {id(m.modulation.weight) for m in conditional_layer_norms(model)}
        regular = next(
            id(p)
            for n, p in model.named_parameters()
            if p.ndim == 2 and min(p.shape) >= 9 and "modulation" not in n
        )
        return by_type, modulation, regular

    by_type, modulation, regular = partition(["modulation"], 9)
    assert modulation <= by_type["AdamW"] and not (modulation & by_type["Muon"])
    assert regular in by_type["Muon"]
    # Without the pattern the modulation matrices ([2D, time_dim]) follow the
    # shape rule like any other 2-D parameter (the tiny model's time_dim is 8).
    by_type, modulation, regular = partition([], 0)
    assert modulation <= by_type["Muon"]


# ----- surface contact on the provisional shape -----


def _contact_sample(gap, time=0.5):
    """Two parallel unit quads ``gap`` mm apart (stats: physical == normalized)."""
    torch.manual_seed(11)
    sheet = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    coords = torch.cat(
        (
            torch.cat((sheet, torch.zeros(4, 1)), dim=-1),
            torch.cat((sheet, torch.full((4, 1), float(gap))), dim=-1),
        )
    )
    n = coords.shape[0]
    ring = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    edges = torch.cat((ring, ring + 4))
    edges = torch.cat((edges, edges.flip(-1))).t().contiguous()
    faces = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    graph = SurfaceContactData(
        **Data(
            edge_index=edges, edge_attr=torch.randn(edges.shape[1], 4), num_nodes=n
        ).to_dict()
    )
    neighborhood = LocalNeighborhoodCache().get(coords, edges, 3, "geodesic")
    graph.knn_index, graph.knn_mask = with_self_slot(neighborhood)
    graph.knn_edge_attr = knn_edge_features(coords, graph.knn_index)
    graph.knn_metric = "geodesic"
    graph.shell_thickness = torch.full((n,), 0.2)
    graph.contact_faces = faces
    graph.contact_surface_exclusions = surface_one_ring_exclusions(faces, n)
    features = {
        "coords": coords,
        "features": torch.randn(n, 1),
        "time": torch.tensor(time),
    }
    target = coords + 0.05 * torch.randn(n, 3)
    stats = {"node": {"pos_mean": torch.zeros(1, 3), "pos_std": torch.ones(1, 3)}}
    return SimSample(features, target, graph=graph), stats


def _contact_model(**overrides):
    return _tc_model(
        use_contact=True, contact_search_implementation="torch", **overrides
    )


def test_provisional_contact_shares_init_with_control_and_is_geometry_only():
    plain = _tc_model()
    contact = _contact_model()
    assert (
        contact.contact_block is None and contact.refinement_contact_block is not None
    )
    shared = dict(contact.named_parameters())
    for name, value in plain.named_parameters():
        assert torch.equal(shared[name], value), name
    extra = set(shared) - {n for n, _ in plain.named_parameters()}
    assert extra and all(n.startswith("refinement_contact_block.") for n in extra)
    with pytest.raises(ValueError, match="geometry-only"):
        _contact_model(contact_include_velocity=True)
    with pytest.raises(ValueError, match="geometry-only"):
        _contact_model(contact_surface_predictive=True)
    # Zero heads: the contact model also starts from the undeformed mesh.
    sample, stats = _contact_sample(gap=0.5)
    with torch.no_grad():
        output = contact.train()(sample=sample, data_stats=stats, return_auxiliary=True)
    torch.testing.assert_close(output.prediction, sample.node_features["coords"])
    torch.testing.assert_close(output.coarse_prediction, sample.node_features["coords"])


def test_provisional_contact_is_discovered_on_the_coarse_shape_and_trains():
    model = _contact_model().train()
    _excite(model)
    captured = []
    hook = model.node_contact_builder.register_forward_hook(
        lambda module, args, output: captured.append(output)
    )
    try:
        close, stats = _contact_sample(gap=0.5)
        far, _ = _contact_sample(gap=50.0)
        output = model(sample=close, data_stats=stats, return_auxiliary=True)
        assert captured[-1].edge_index.shape[1] > 0
        # The gate scales a contact-dependent correction of the final position only.
        with torch.no_grad():
            model.refinement_contact_block.gate.fill_(1.0)
            gated = model(sample=close, data_stats=stats, return_auxiliary=True)
            model.refinement_contact_block.gate.fill_(0.0)
            ungated = model(sample=close, data_stats=stats, return_auxiliary=True)
            model.refinement_contact_block.gate.fill_(1.0)
        assert not torch.allclose(gated.prediction, ungated.prediction)
        torch.testing.assert_close(gated.coarse_prediction, ungated.coarse_prediction)
        # Gradients reach the contact block, and the coarse head through the
        # contact geometry as well as through the final position.
        output = model(sample=close, data_stats=stats, return_auxiliary=True)
        output.prediction.square().sum().backward()
        grad = model.refinement_contact_block.edge_mlp[0].weight.grad
        assert grad is not None and grad.abs().sum() > 0
        assert model.coarse_decoder[-1].weight.grad.abs().sum() > 0
        # Distant sheets: no pairs, finite output, zero but defined contact grads.
        model.zero_grad()
        output = model(sample=far, data_stats=stats, return_auxiliary=True)
        assert captured[-1].edge_index.shape[1] == 0
        output.prediction.square().sum().backward()
        assert torch.isfinite(output.prediction).all()
        assert model.refinement_contact_block.edge_mlp[0].weight.grad is not None
        # Evaluation queries every frame with its own contact graph.
        model.eval()
        with torch.no_grad():
            rollout = model(sample=close, data_stats=stats)
        assert rollout.shape == (8, 4, 3) and len(captured) >= 4 + 5
    finally:
        hook.remove()
    # Without topology the model refuses rather than silently skipping contact.
    bare, stats = _sample(time=0.5)
    with pytest.raises(ValueError, match="shell_thickness|contact_faces"):
        model.train()(sample=bare, data_stats=stats)


def test_provisional_contact_recipe_differs_from_control_only_by_contact():
    base = _compose("crash_deformer_knn_time_conditional")
    contact = _compose("crash_deformer_knn_time_conditional_contact")
    assert contact.reader.include_contact_topology is True
    assert contact.datapipe.contact_surface is True
    assert contact.datapipe.contact_surface_exclusion == "reference_geodesic"
    assert contact.model.use_contact is True and contact.model.contact_dim == 8
    assert contact.model.contact_include_velocity is False
    assert contact.model.contact_surface_predictive is False
    for key in base.training:
        assert base.training[key] == contact.training[key], key
    assert set(base.training) == set(contact.training)
    contact_model = OmegaConf.to_container(contact.model, resolve=True)
    base_model = OmegaConf.to_container(base.model, resolve=True)
    assert base_model.pop("use_contact") is False
    assert {k: v for k, v in contact_model.items() if k in base_model} == base_model
    assert all(
        k.startswith(("contact_", "use_contact", "node_contact"))
        for k in set(contact_model) - set(base_model)
    )
    base_pipe = OmegaConf.to_container(base.datapipe, resolve=True)
    contact_pipe = OmegaConf.to_container(contact.datapipe, resolve=True)
    assert {
        k: v
        for k, v in contact_pipe.items()
        if k in base_pipe and not k.startswith("contact_")
    } == {k: v for k, v in base_pipe.items() if not k.startswith("contact_")}
    assert contact.experiment_name != base.experiment_name


# ----- single-stage ablation (no provisional head, no refinement copy) -----


def test_single_stage_model_is_one_post_stack_with_one_head():
    sample, stats = _sample(time=0.5)
    model = _tc_model(two_stage=False, num_post_processor_layers=2).train()
    assert model.two_stage is False
    assert model.coarse_decoder is None and model.refinement_processor is None
    assert model.refinement_context_film is None
    two_stage = _tc_model(num_post_processor_layers=2)
    assert not any(n.startswith("coarse_decoder") for n, _ in model.named_parameters())
    assert sum(p.numel() for p in model.parameters()) < sum(
        p.numel() for p in two_stage.parameters()
    )
    # Zero head: starts from the undeformed mesh, coarse output mirrors the prediction.
    with torch.no_grad():
        output = model(sample=sample, data_stats=stats, return_auxiliary=True)
    torch.testing.assert_close(output.prediction, sample.node_features["coords"])
    torch.testing.assert_close(output.coarse_prediction, output.prediction)
    # Trains through the post stack and the backbone once the head is non-zero.
    last = [m for m in model.post_node_decoder.modules() if isinstance(m, nn.Linear)][
        -1
    ]
    nn.init.normal_(last.weight, std=0.1)
    output = model(sample=sample, data_stats=stats)
    output.square().sum().backward()
    for prefix in ("post_processor", "blocks", "time_embed", "post_context_film"):
        assert any(
            n.startswith(prefix) and p.grad is not None and p.grad.abs().sum() > 0
            for n, p in model.named_parameters()
        ), prefix
    model.eval()
    with torch.no_grad():
        rollout = model(sample=sample, data_stats=stats)
    assert rollout.shape == (9, 4, 3)
    with pytest.raises(ValueError, match="two_stage"):
        _tc_model(
            two_stage=False, use_contact=True, contact_search_implementation="torch"
        )


def test_single_stage_recipe_differs_from_control_only_by_the_stage_split():
    base = _compose("crash_deformer_knn_time_conditional")
    single = _compose("crash_deformer_knn_time_conditional_single_stage")
    assert single.model.two_stage is False and "two_stage" not in base.model
    assert single.model.num_post_processor_layers == 2
    assert base.model.num_post_processor_layers == 1
    assert single.training.coarse_loss_weight == 0.0
    for key in base.training:
        if key != "coarse_loss_weight":
            assert base.training[key] == single.training[key], key
    assert set(base.training) == set(single.training)
    single_model = OmegaConf.to_container(single.model, resolve=True)
    base_model = OmegaConf.to_container(base.model, resolve=True)
    single_model.pop("two_stage")
    single_model.pop("num_post_processor_layers")
    base_model.pop("num_post_processor_layers")
    assert single_model == base_model
    assert OmegaConf.to_container(
        single.datapipe, resolve=True
    ) == OmegaConf.to_container(base.datapipe, resolve=True)


# ----- contact correction: deeper stage B fed with the provisional displacement -----


def test_refinement_depth_and_displacement_adapter():
    sample, stats = _sample(time=0.5)
    control = _tc_model()
    model = _tc_model(num_refinement_layers=2, refinement_input_displacement=True)
    assert model.num_refinement_layers == 2
    assert len(model.refinement_processor.blocks) == 2
    assert len(model.post_processor.blocks) == 1
    # Everything built before stage B is initialised exactly as in the control.
    shared = dict(model.named_parameters())
    for name, value in control.named_parameters():
        if not name.startswith(("refinement_", "post_node_decoder")):
            assert torch.equal(shared[name], value), name
    # Zero adapter and zero heads: starts from the undeformed mesh.
    model.train()
    with torch.no_grad():
        output = model(sample=sample, data_stats=stats, return_auxiliary=True)
    torch.testing.assert_close(output.prediction, sample.node_features["coords"])
    # With non-zero heads the correction depends on the provisional displacement
    # through the adapter, and the adapter receives gradients.
    _excite(model)
    with torch.no_grad():
        plain = model(sample=sample, data_stats=stats, return_auxiliary=True)
        nn.init.normal_(model.coarse_feedback.weight, std=0.5)
        fed = model(sample=sample, data_stats=stats, return_auxiliary=True)
    torch.testing.assert_close(fed.coarse_prediction, plain.coarse_prediction)
    assert not torch.allclose(
        fed.prediction - fed.coarse_prediction,
        plain.prediction - plain.coarse_prediction,
    )
    output = model(sample=sample, data_stats=stats, return_auxiliary=True)
    output.prediction.square().sum().backward()
    assert model.coarse_feedback.weight.grad.abs().sum() > 0
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if n.startswith("refinement_processor.blocks.1.update_mlp")
    )
    model.eval()
    with torch.no_grad():
        assert model(sample=sample, data_stats=stats).shape == (9, 4, 3)
    # The message-passing operator builds the deeper stage the same way.
    mpnn = _tc_model(local_processor="knn_mpnn", num_refinement_layers=2)
    assert mpnn.refinement_processor.processor_size == 2
    assert mpnn.post_processor.processor_size == 1
    out = mpnn.train()(sample=sample, data_stats=stats)
    assert out.shape == (9, 3) and torch.isfinite(out).all()
    with pytest.raises(ValueError, match="two_stage"):
        _tc_model(two_stage=False, refinement_input_displacement=True)
    with pytest.raises(ValueError, match="two_stage"):
        _tc_model(two_stage=False, num_refinement_layers=2)
    with pytest.raises(ValueError, match="at least 1"):
        _tc_model(num_refinement_layers=0)


def test_contact_correction_combines_contact_depth_and_adapter():
    close, stats = _contact_sample(gap=0.5)
    model = _contact_model(num_refinement_layers=2, refinement_input_displacement=True)
    model.train()
    _excite(model)
    nn.init.normal_(model.coarse_feedback.weight, std=0.1)
    captured = []
    hook = model.node_contact_builder.register_forward_hook(
        lambda module, args, output: captured.append(output)
    )
    try:
        output = model(sample=close, data_stats=stats, return_auxiliary=True)
        assert captured[-1].edge_index.shape[1] > 0
        output.prediction.square().sum().backward()
    finally:
        hook.remove()
    assert model.coarse_feedback.weight.grad.abs().sum() > 0
    assert model.refinement_contact_block.edge_mlp[0].weight.grad.abs().sum() > 0
    assert torch.isfinite(output.prediction).all()


def test_contact_correction_recipe_differs_from_contact_recipe_only_by_stage_b():
    contact = _compose("crash_deformer_knn_time_conditional_contact")
    correction = _compose("crash_deformer_knn_time_conditional_contact_correction")
    assert correction.model.num_refinement_layers == 2
    assert correction.model.refinement_input_displacement is True
    assert "num_refinement_layers" not in contact.model
    for key in contact.training:
        assert contact.training[key] == correction.training[key], key
    assert set(contact.training) == set(correction.training)
    a = OmegaConf.to_container(correction.model, resolve=True)
    a.pop("num_refinement_layers")
    a.pop("refinement_input_displacement")
    assert a == OmegaConf.to_container(contact.model, resolve=True)
    assert OmegaConf.to_container(correction.datapipe, resolve=True) == (
        OmegaConf.to_container(contact.datapipe, resolve=True)
    )
    assert OmegaConf.to_container(correction.reader, resolve=True) == (
        OmegaConf.to_container(contact.reader, resolve=True)
    )
    assert correction.experiment_name != contact.experiment_name


def test_penetration_penalty_sees_crossings_and_is_zero_when_separated():
    """Crossing-aware penalty on a predicted frame; unsigned search misses it."""
    sided = _contact_model(
        contact_side_from_reference=True, contact_activation_distance=2.0
    ).train()
    sample, stats = _contact_sample(gap=3.0)
    coords = sample.node_features["coords"]
    # The predicted upper sheet (reference 3 above) sits 0.5 below the lower one.
    crossed = coords.clone()
    crossed[4:, 2] = -0.5
    crossed.requires_grad_()
    loss, diagnostics = sided.penetration_penalty(
        crossed, sample, stats, tolerance_mm=0.0, length_scale_mm=1.0
    )
    # Every node of either sheet crosses the other: depth 0.5 + 0.2 half thicknesses.
    torch.testing.assert_close(loss, torch.tensor(0.7**2))
    assert diagnostics["penetration_crossed_pairs"] > 0
    assert diagnostics["penetration_active_nodes"] == 8
    loss.backward()
    assert torch.isfinite(crossed.grad).all() and crossed.grad.abs().sum() > 0
    # Separated beyond the band: no pairs, exactly zero, still differentiable path.
    apart = coords.clone()
    apart[4:, 2] = 2.5
    zero, diagnostics = sided.penetration_penalty(
        apart, sample, stats, tolerance_mm=0.0, length_scale_mm=1.0
    )
    assert zero == 0 and diagnostics["penetration_active_nodes"] == 0
    # Without the side test the crossed sheet looks 0.3 clear of the envelope.
    plain = _contact_model(contact_activation_distance=2.0).train()
    unsigned, _ = plain.penetration_penalty(
        crossed.detach(), sample, stats, tolerance_mm=0.0, length_scale_mm=1.0
    )
    assert unsigned == 0
    # A tolerance above the depth and the length scale behave as documented.
    tolerant, _ = sided.penetration_penalty(
        crossed.detach(), sample, stats, tolerance_mm=1.0, length_scale_mm=1.0
    )
    assert tolerant == 0
    scaled, _ = sided.penetration_penalty(
        crossed.detach(), sample, stats, tolerance_mm=0.0, length_scale_mm=10.0
    )
    torch.testing.assert_close(scaled, torch.tensor(0.07**2))
    with pytest.raises(ValueError, match="surface contact"):
        _tc_model().penetration_penalty(crossed.detach(), sample, stats)
    with pytest.raises(ValueError, match="tolerance_mm"):
        sided.penetration_penalty(crossed.detach(), sample, stats, tolerance_mm=-1.0)


def test_contact_iterations_rebuild_contact_on_the_corrected_shape():
    once = _contact_model(contact_side_from_reference=True)
    twice = _contact_model(contact_side_from_reference=True, contact_iterations=2)
    twice.load_state_dict(once.state_dict())
    for model in (once, twice):
        _excite(model)
        model.train()
    twice.load_state_dict(once.state_dict())
    seen = []
    hook = twice.node_contact_builder.register_forward_hook(
        lambda module, args, output: seen.append(args[0].detach().clone())
    )
    try:
        sample, stats = _contact_sample(gap=0.5)
        with torch.no_grad():
            # Gate zero: contact changes nothing, so both passes agree exactly.
            for model in (once, twice):
                model.refinement_contact_block.gate.fill_(0.0)
            same = twice(sample=sample, data_stats=stats)
            torch.testing.assert_close(same, once(sample=sample, data_stats=stats))
            assert len(seen) == 2 and not torch.equal(seen[0], seen[1])
            # The second pass searched on the first pass's corrected shape.
            first_shape = once(sample=sample, data_stats=stats)
            scale = stats["node"]["pos_std"].reshape(1, 3)
            mean = stats["node"]["pos_mean"].reshape(1, 3)
            torch.testing.assert_close(seen[1], first_shape * scale + mean)
            # Gate on: the second pass changes the prediction.
            for model in (once, twice):
                model.refinement_contact_block.gate.fill_(1.0)
            seen.clear()
            assert not torch.allclose(
                twice(sample=sample, data_stats=stats),
                once(sample=sample, data_stats=stats),
            )
        # Differentiable through both passes.
        twice(sample=sample, data_stats=stats).square().sum().backward()
        assert twice.refinement_contact_block.edge_mlp[0].weight.grad.abs().sum() > 0
    finally:
        hook.remove()
    with pytest.raises(ValueError, match="contact_iterations"):
        _tc_model(contact_iterations=2)
    with pytest.raises(ValueError, match="positive integer"):
        _contact_model(contact_iterations=0)


def test_load_warm_start_keeps_shared_weights_and_rejects_strangers(tmp_path):
    from utils import load_warm_start

    plain = _tc_model()
    _excite(plain)
    control = tmp_path / "control.mdlus"
    plain.save(str(control))
    contact = _contact_model(contact_side_from_reference=True)
    fresh = {
        n: p.detach().clone()
        for n, p in contact.named_parameters()
        if n.startswith("refinement_contact_block.")
    }
    report = load_warm_start(contact, control)
    after = dict(contact.named_parameters())
    for name, value in plain.named_parameters():
        assert torch.equal(after[name], value), name
    for name, value in fresh.items():
        assert torch.equal(after[name], value), name
    assert report["missing"] and all(
        k.startswith("refinement_contact_block.") for k in report["missing"]
    )
    assert len(report["loaded"]) == len(plain.state_dict())
    # Near function-preserving: the gate starts at 1e-3, exactly preserving at 0.
    sample, stats = _contact_sample(gap=0.5)
    with torch.no_grad():
        reference = plain.eval()(sample=sample, data_stats=stats)
        torch.testing.assert_close(
            contact.eval()(sample=sample, data_stats=stats),
            reference,
            atol=2e-3,
            rtol=0.0,
        )
        contact.refinement_contact_block.gate.zero_()
        torch.testing.assert_close(contact(sample=sample, data_stats=stats), reference)
    # A contact checkpoint carries tensors a control model does not have.
    contact.save(str(tmp_path / "contact.mdlus"))
    with pytest.raises(ValueError, match="unexpected"):
        load_warm_start(_tc_model(), tmp_path / "contact.mdlus")
    with pytest.raises(ValueError, match="missing"):
        load_warm_start(
            _contact_model(contact_side_from_reference=True),
            control,
            allowed_missing_prefixes=(),
        )


def test_global_motion_head_is_rigid_and_function_preserving():
    plain = _tc_model()
    rigid = _tc_model(global_motion=True)
    rigid.load_state_dict(plain.state_dict(), strict=False)
    for model in (plain, rigid):
        _excite(model)
        model.train()
    rigid.load_state_dict({**rigid.state_dict(), **plain.state_dict()})
    sample, stats = _sample(time=0.4)
    # Anisotropic statistics so the rigid motion is checked in physical units.
    stats = {
        "node": {
            "pos_mean": torch.tensor([[3.0, -1.0, 2.0]]),
            "pos_std": torch.tensor([[4.0, 2.0, 1.0]]),
        }
    }
    with torch.no_grad():
        base = plain(sample=sample, data_stats=stats)
        torch.testing.assert_close(rigid(sample=sample, data_stats=stats), base)
        # A pure translation of 1 unit output = scale mm along x, same for every node.
        rigid.global_motion_head[-1].bias[0] = 1.0
        shifted = rigid(sample=sample, data_stats=stats)
        delta = (shifted - base) * stats["node"]["pos_std"]
        torch.testing.assert_close(
            delta, torch.tensor([[10.0, 0.0, 0.0]]).expand_as(delta)
        )
        # A rotation about z moves nodes by theta x (x - centroid), theta = scale / lever.
        rigid.global_motion_head[-1].bias.zero_()
        rigid.global_motion_head[-1].bias[5] = 1.0
        turned = rigid(sample=sample, data_stats=stats)
        coords = sample.node_features["coords"]
        physical = coords * stats["node"]["pos_std"] + stats["node"]["pos_mean"]
        relative = physical - physical.mean(0, keepdim=True)
        theta = torch.tensor([0.0, 0.0, 10.0 / 2000.0])
        expected = torch.cross(theta.expand_as(relative), relative, dim=-1)
        torch.testing.assert_close((turned - base) * stats["node"]["pos_std"], expected)
    # The head trains through the final loss.
    rigid.global_motion_head[-1].bias.data.zero_()
    rigid(sample=sample, data_stats=stats).square().sum().backward()
    assert rigid.global_motion_head[-1].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="positive"):
        _tc_model(global_motion=True, global_motion_scale_mm=0.0)


def test_prediction_ensemble_averages_member_outputs(tmp_path):
    from evaluate_yaris_tc import PredictionEnsemble

    members = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        model = _tc_model()
        _excite(model)
        model.save(str(tmp_path / f"m{seed}.mdlus"))
        members.append(model.eval())
    sample, stats = _sample(time=0.3)
    ensemble = PredictionEnsemble(
        _tc_model(), [str(tmp_path / f"m{s}.mdlus") for s in (1, 2, 3)]
    )
    ensemble.eval()
    with torch.no_grad():
        expected = torch.stack(
            [m(sample=sample, data_stats=stats) for m in members]
        ).mean(0)
        torch.testing.assert_close(ensemble(sample=sample, data_stats=stats), expected)
    with pytest.raises(ValueError, match="missing"):
        PredictionEnsemble(_contact_model(), [str(tmp_path / "m1.mdlus")])


def test_concrete_dropout_is_stochastic_in_training_and_silent_in_eval():
    from physicsnemo.nn import ConcreteDropout, collect_concrete_dropout_losses

    plain = _tc_model()
    _excite(plain)
    dropped = _tc_model(concrete_dropout=True, dropout=0.1)
    dropped.load_state_dict(plain.state_dict(), strict=False)
    layers = [m for m in dropped.modules() if isinstance(m, ConcreteDropout)]
    assert layers and all(abs(float(m.p) - 0.1) < 1e-5 for m in layers)
    sample, stats = _sample(time=0.4)
    dropped.train()
    with torch.no_grad():
        first = dropped(sample=sample, data_stats=stats)
        second = dropped(sample=sample, data_stats=stats)
    assert not torch.allclose(first, second)
    dropped.eval()
    plain.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            dropped(sample=sample, data_stats=stats),
            plain(sample=sample, data_stats=stats),
        )
    term = collect_concrete_dropout_losses(dropped)
    assert term.requires_grad and torch.isfinite(term)
    assert float(term) < 0  # negative Bernoulli entropy at p = 0.1


def test_local_block_dropout_sites():
    from physicsnemo.nn import ConcreteDropout

    plain = _tc_model()
    _excite(plain)
    learned = _tc_model(local_dropout=0.1, local_concrete_dropout=True)
    learned.load_state_dict(plain.state_dict(), strict=False)
    blocks = sum(
        1
        for name, m in learned.named_modules()
        if isinstance(m, ConcreteDropout) and name.endswith(".dropout")
    )
    assert blocks == sum(
        len(p.blocks)
        for p in (
            learned.pre_processor,
            learned.post_processor,
            learned.refinement_processor,
        )
    )
    sample, stats = _sample(time=0.4)
    learned.train()
    with torch.no_grad():
        a = learned(sample=sample, data_stats=stats)
        b = learned(sample=sample, data_stats=stats)
    assert not torch.allclose(a, b)
    learned.eval()
    plain.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            learned(sample=sample, data_stats=stats),
            plain(sample=sample, data_stats=stats),
        )
    standard = _tc_model(local_dropout=0.2)
    assert any(isinstance(m, nn.Dropout) for m in standard.modules())
    assert not any(isinstance(m, ConcreteDropout) for m in standard.modules())
    with pytest.raises(ValueError, match="dropout"):
        _tc_model(local_dropout=1.0)


def test_part_token_design_encoder_is_permutation_invariant_and_feeds_the_head():
    model = _tc_model(
        part_tokens=True,
        part_token_dim=5,
        design_dim=8,
        global_dim=16,
        global_motion=True,
    )
    _excite(model)
    nn.init.normal_(model.design_out.weight, std=0.1)
    nn.init.normal_(model.design_pool_query, std=0.1)
    sample, stats = _sample(time=0.4)
    stats = {"node": {"pos_mean": torch.zeros(1, 3), "pos_std": torch.ones(1, 3)}}
    torch.manual_seed(3)
    parts = torch.randn(7, 5)
    sample.global_features = {"parts": parts}
    model.eval()
    with torch.no_grad():
        out = model(sample=sample, data_stats=stats)
        sample.global_features = {"parts": parts[torch.randperm(7)]}
        torch.testing.assert_close(model(sample=sample, data_stats=stats), out)
        sample.global_features = {"parts": parts + 1.0}
        assert not torch.allclose(model(sample=sample, data_stats=stats), out)
    assert model.global_motion_head[0].in_features == model.mesh_hidden_dim + 8 + 8
    sample.global_features = None
    with pytest.raises(ValueError, match="parts"):
        model(sample=sample, data_stats=stats)
    with pytest.raises(ValueError, match="positive"):
        _tc_model(part_tokens=True, part_token_dim=0)
