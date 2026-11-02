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
"""Time-conditional DeFormer (time_conditional.py) and its GeoTransolver baseline."""

import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data

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
    drop_retired_options,
)

FIXTURE_OVERRIDES = [
    "training.raw_data_dir=/fixtures/train",
    "training.raw_data_dir_validation=/fixtures/validation",
    "inference.raw_data_dir_test=/fixtures/test",
    "++yaris.data_root=/fixtures/yaris",
    "++yaris.split_manifest=/fixtures/splits.csv",
    "++yaris.outlier_mask=/fixtures/outliers.npz",
    "++yaris.work_dir=/fixtures/work",
]


def _compose(name, overrides=()):
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose(config_name=name, overrides=[*FIXTURE_OVERRIDES, *overrides])


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
    penetration_projection_implementation="torch",
)


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
        penetration_projection_implementation="torch",
    )
    kwargs.update(overrides)
    torch.manual_seed(1)
    return MeshGeoFLARETimeConditional(**kwargs)


def _excite(model, std=0.1):
    """Zero-initialised heads block every upstream gradient; give them weights."""
    heads = [model.coarse_decoder[-1]]
    heads.append(
        [m for m in model.post_node_decoder.modules() if isinstance(m, nn.Linear)][-1]
    )
    for head in heads:
        nn.init.normal_(head.weight, std=std)


def _contact_sample(gap, time=0.5):
    """Two parallel unit quads ``gap`` mm apart (stats: physical == normalized),
    with the shell facets, thickness and one-ring exclusions of the contact search."""
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


def test_time_embedding_shapes_and_smoothness():
    scalar = TimeEmbedding("scalar")
    assert scalar.dim == 1 and scalar(torch.tensor(0.25)).tolist() == [[0.25]]
    fourier = TimeEmbedding("fourier", bands=4, dim=12)
    codes = torch.stack([fourier(torch.tensor(q)).squeeze(0) for q in (0.0, 0.01, 0.5)])
    assert codes.shape == (3, 12)
    assert (codes[0] - codes[1]).norm() < (codes[0] - codes[2]).norm()
    with pytest.raises(ValueError):
        TimeEmbedding("linear")


def test_gm_recipes_compose_and_match_each_other():
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
    # body-in-white meshes: geodesic neighbourhoods, no activation checkpointing, no frame interval.
    assert tc.model.local_neighborhood == "geodesic" and tc.model.dt is None
    assert tc.model.activation_checkpointing is False
    assert tc.model.num_pre_processor_checkpoint_segments == 0
    assert (
        tc.model.num_post_processor_layers == 1
        and tc.model.mesh_post_residual_gate_init == 1.0
    )
    for key in ("epochs", "start_lr", "end_lr", "validation_freq", "seed", "amp_dtype"):
        assert base.training[key] == tc.training[key], key
    resolved = OmegaConf.to_container(tc, resolve=True)
    assert resolved["datapipe"]["local_knn_k"] == 32
    assert resolved["datapipe"]["local_knn_metric"] == "geodesic"


def test_yaris_recipes_compose():
    network = _compose("yaris_deformer")
    assert network.model._target_ == "time_conditional.MeshGeoFLARETimeConditional"
    assert network.datapipe._target_ == "yaris_full.YarisFullGraphDataset"
    assert network.reader._target_ == "yaris_full.YarisFullReader"
    assert network.training.trainer_class == "yaris_full_trainer.YarisFullTrainer"
    assert network.training.epochs == 60 and network.training.num_time_steps == 31
    assert network.training.loss == "mse" and network.training.loss_space == "physical"
    assert network.training.loss_scale_mm == 10.0
    assert (
        network.training.coarse_loss_weight == 0.1
        and network.training.optimizer == "muon"
    )
    assert (
        network.model.local_neighborhood == "euclidean"
        and network.model.local_knn_k == 32
    )
    assert network.model.local_attention_fused is True
    assert network.model.activation_checkpointing is True
    assert network.model.penetration_projection_iterations == 0
    assert network.model.clamp_to_wall is False
    resolved = OmegaConf.to_container(network, resolve=True)
    assert (
        resolved["datapipe"]["dt"] == 0.005
        and resolved["datapipe"]["local_knn_k"] == 32
    )
    assert resolved["datapipe"]["sample_type"] == "one_time_step"
    assert (
        resolved["datapipe"]["local_knn_cache_dir"]
        == "/fixtures/work/stats/local_knn_cache"
    )
    baseline = _compose("yaris_geotransolver")
    assert baseline.model._target_ == "rollout.GeoTransolverTimeConditional"
    assert baseline.training.coarse_loss_weight == 0.0
    assert "local_knn_k" not in baseline.model
    corrector = _compose("yaris_deformer_corrector")
    assert corrector.model._target_ == "residual_model.ResidualTimeConditional"
    assert corrector.model.final_clamp_to_wall is True
    assert corrector.datapipe.rigid_wall_position == 100.0
    assert (
        corrector.datapipe.contact_surface is True
        and corrector.datapipe.mesh_edges is True
    )
    assert OmegaConf.to_container(corrector.model.teacher) == OmegaConf.to_container(
        network.model
    )


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
            singles.append(model.predict_frame(sample, stats).prediction)
        sample.node_features["time"] = original_time
    assert rollout.shape == (9, 4, 3)
    torch.testing.assert_close(rollout, torch.stack(singles, dim=1))
    torch.testing.assert_close(sample.node_features["time"], original_time)
    # A shorter target queries exactly its horizon.
    short = SimSample(sample.node_features, torch.zeros(9, 2, 3), graph=sample.graph)
    with torch.no_grad():
        assert model(sample=short, data_stats=stats).shape == (9, 2, 3)
    with pytest.raises(ValueError, match="future-frame range"):
        sample.node_features["time"] = torch.tensor(1.5)
        model.train()(sample=sample, data_stats=stats)


def test_tc_model_rejects_wrong_output_width_and_te():
    with pytest.raises(ValueError, match="positions only"):
        _tc_model(out_dim=5)
    with pytest.raises(ValueError, match="use_te=False"):
        _tc_model(use_te=True)
    with pytest.raises(ValueError, match="num_time_steps"):
        _tc_model(num_time_steps=1)
    with pytest.raises(ValueError, match="dt"):
        _tc_model(dt=0.0)


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
        dt=0.005,
    ).train()
    assert model.dt == 0.005 and model.rollout_steps == 4
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


@pytest.mark.parametrize(
    "name, steps",
    [
        ("crash_geoflare_time_conditional", 25),
        ("crash_deformer_knn_time_conditional", 25),
        ("yaris_deformer", 30),
        ("yaris_geotransolver", 30),
    ],
)
def test_time_conditional_model_configs_instantiate(name, steps):
    """Every key of the model group must be accepted by the model class."""
    from hydra.utils import instantiate

    cfg = _compose(name)
    overrides = {k: v for k, v in TINY.items() if k in cfg.model}
    model = instantiate(cfg.model, **overrides)
    assert model.rollout_steps == steps


def test_retired_research_options_are_accepted_at_their_released_values():
    model = _tc_model(
        norm_conditioning="none",
        use_contact=False,
        two_stage=True,
        num_refinement_layers=None,
        refinement_input_displacement=False,
        global_motion=False,
        rigid_parts=False,
        head_init_std=0.0,
        refiner=False,
        contact_search_implementation="torch",
        contact_surface_max_pairs=1234,
    )
    assert model.projection.implementation == "torch"
    assert model.projection.max_pairs == 1234
    both = _tc_model(
        contact_search_implementation="torch",
        penetration_projection_implementation="warp",
        contact_surface_max_pairs=111,
        penetration_projection_max_pairs=222,
    )
    assert both.projection.implementation == "warp"
    assert both.projection.max_pairs == 222
    assert not hasattr(model, "global_motion")
    for bad in (
        dict(global_motion=True),
        dict(use_contact=True),
        dict(two_stage=False),
        dict(norm_conditioning="time"),
        dict(head_init_std=0.02),
    ):
        with pytest.raises(ValueError, match="research option"):
            _tc_model(**bad)
    with pytest.raises(TypeError):
        _tc_model(not_an_option=1)
    assert drop_retired_options(
        {"a": 1, "old": 2, "keep": 3}, {"a": 1}, {"old": "new"}
    ) == {
        "new": 2,
        "keep": 3,
    }
    assert drop_retired_options({"old": 2, "new": 5}, {}, {"old": "new"}) == {"new": 5}


def test_evaluation_applies_the_wall_clamp_but_training_does_not():
    sample, stats = _contact_sample(gap=0.5)
    plain = _tc_model()
    _excite(plain)
    clamped = _tc_model(clamp_to_wall=True)
    clamped.load_state_dict(plain.state_dict())
    plain.eval()
    with torch.no_grad():
        base = plain(sample=sample, data_stats=stats)
    wall = float(base[..., 0].median())
    sample.graph.rigid_wall = torch.tensor([wall])
    clamped.eval()
    with torch.no_grad():
        out = clamped(sample=sample, data_stats=stats)
    assert bool((base[..., 0] > wall + 1e-6).any())
    assert bool((out[..., 0] <= wall + 1e-6).all())
    torch.testing.assert_close(out[..., 1:], base[..., 1:])
    torch.testing.assert_close(out[..., 0], base[..., 0].clamp(max=wall))
    # Training predictions are never clamped.
    plain.train()
    clamped.train()
    with torch.no_grad():
        torch.testing.assert_close(
            clamped(sample=sample, data_stats=stats),
            plain(sample=sample, data_stats=stats),
        )
    # The projection runs in evaluation on the shell facets; nothing penetrates
    # here, so the output is unchanged, and invalid settings are refused.
    projecting = _tc_model(penetration_projection_iterations=2)
    projecting.load_state_dict(plain.state_dict())
    assert projecting.projection.enabled
    projecting.eval()
    with torch.no_grad():
        torch.testing.assert_close(projecting(sample=sample, data_stats=stats), base)
    for bad in (
        dict(penetration_projection_iterations=-1),
        dict(penetration_projection_damping=0.0),
        dict(penetration_projection_tolerance_mm=-0.1),
        dict(wall_axis=3),
    ):
        with pytest.raises(ValueError):
            _tc_model(**bad)
    del sample.graph.rigid_wall
    with pytest.raises(ValueError, match="rigid_wall"):
        with torch.no_grad():
            clamped.eval()(sample=sample, data_stats=stats)


def test_load_warm_start_keeps_weights_and_rejects_strangers(tmp_path):
    from utils import load_warm_start

    source = _tc_model()
    _excite(source)
    path = tmp_path / "source.mdlus"
    source.save(str(path))
    target = _tc_model()
    report = load_warm_start(target, path)
    for name, value in source.named_parameters():
        assert torch.equal(dict(target.named_parameters())[name], value), name
    assert not report["missing"] and len(report["loaded"]) == len(source.state_dict())
    # Missing tensors must belong to an allowed prefix; unexpected ones are refused.
    partial = {
        k: v
        for k, v in source.state_dict().items()
        if not k.startswith("coarse_decoder.")
    }
    torch.save(partial, tmp_path / "partial.pt")
    with pytest.raises(ValueError, match="missing"):
        load_warm_start(_tc_model(), tmp_path / "partial.pt")
    report = load_warm_start(
        _tc_model(),
        tmp_path / "partial.pt",
        allowed_missing_prefixes=("coarse_decoder.",),
    )
    assert report["missing"] and all(
        k.startswith("coarse_decoder.") for k in report["missing"]
    )
    extra = dict(source.state_dict())
    extra["stranger.weight"] = torch.zeros(1)
    torch.save(extra, tmp_path / "extra.pt")
    with pytest.raises(ValueError, match="unexpected"):
        load_warm_start(_tc_model(), tmp_path / "extra.pt")
