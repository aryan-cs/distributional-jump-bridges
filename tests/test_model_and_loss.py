from __future__ import annotations

import torch

from cebt.models.cebt import ModelConfig, build_model
from cebt.training.losses import (
    LossWeights,
    cebt_loss,
    gaussian_nll_loss,
    pairwise_rank_loss,
    supervised_response_loss,
)


def test_cebt_forward_shapes_and_finite_loss() -> None:
    config = ModelConfig(
        price_features=8, metadata_features=6, event_embedding_dim=16, hidden_dim=32, latent_dim=4
    )
    model = build_model("cebt", config)
    outputs = model(
        torch.randn(3, 5, 8),
        torch.randn(3, 16),
        torch.randn(3, 6),
    )
    assert outputs["prediction"].shape == (3, 3)
    assert outputs["event_delta"].shape == (3, 3)
    assert outputs["mu"].shape == (3, 4)
    loss, metrics = cebt_loss(
        outputs, torch.randn(3, 3), torch.tensor([1.0, 0.0, 1.0]), LossWeights()
    )
    assert torch.isfinite(loss)
    assert metrics["kl"] >= 0.0
    assert "rank" in metrics


def test_bottleneck_dimension_changes_latent_shape() -> None:
    small = build_model("cebt", ModelConfig(event_embedding_dim=16, hidden_dim=32, latent_dim=2))
    large = build_model("cebt", ModelConfig(event_embedding_dim=16, hidden_dim=32, latent_dim=8))
    x = torch.randn(2, 5, 8)
    e = torch.randn(2, 16)
    m = torch.randn(2, 6)
    assert small(x, e, m)["mu"].shape[-1] == 2
    assert large(x, e, m)["mu"].shape[-1] == 8


def test_disclosure_operator_forward_shapes() -> None:
    config = ModelConfig(
        price_features=8,
        metadata_features=6,
        event_embedding_dim=16,
        hidden_dim=32,
        latent_dim=4,
        operator_rank=3,
    )
    model = build_model("dot", config)
    outputs = model(
        torch.randn(4, 5, 8),
        torch.randn(4, 16),
        torch.randn(4, 6),
    )
    assert outputs["prediction"].shape == (4, 3)
    assert outputs["event_delta"].shape == (4, 3)
    assert outputs["z_event"].shape == (4, 32)
    assert outputs["operator_norm"].shape == (4,)
    assert torch.all(torch.isfinite(outputs["operator_norm"]))


def test_event_jump_state_space_forward_shapes_and_nll() -> None:
    config = ModelConfig(
        price_features=8,
        metadata_features=6,
        event_embedding_dim=16,
        hidden_dim=32,
        latent_dim=4,
        jump_scale=0.2,
    )
    model = build_model("ejssm", config)
    outputs = model(
        torch.randn(4, 5, 8),
        torch.randn(4, 16),
        torch.randn(4, 6),
    )
    assert outputs["prediction"].shape == (4, 3)
    assert outputs["event_delta"].shape == (4, 3)
    assert outputs["z_event"].shape == (4, 32)
    assert outputs["outcome_logvar"].shape == (4, 3)
    assert outputs["jump_norm"].shape == (4,)
    targets = torch.randn(4, 3)
    nll = gaussian_nll_loss(outputs, targets)
    loss, metrics = cebt_loss(
        outputs, targets, torch.tensor([1.0, 1.0, 0.0, 0.0]), LossWeights(nll_weight=0.1)
    )
    assert torch.isfinite(nll)
    assert torch.isfinite(loss)
    assert metrics["nll"] == float(nll.detach().cpu())


def test_event_jump_no_jump_intervention_zeroes_latent_jump() -> None:
    config = ModelConfig(
        price_features=8,
        metadata_features=6,
        event_embedding_dim=16,
        hidden_dim=32,
        latent_dim=4,
        jump_scale=0.2,
    )
    model = build_model("ejssm", config)
    x_pre = torch.randn(4, 5, 8)
    event_embedding = torch.randn(4, 16)
    metadata = torch.randn(4, 6)
    outputs = model(x_pre, event_embedding, metadata, intervention="no_jump")
    assert torch.allclose(outputs["z_event"], torch.zeros_like(outputs["z_event"]))
    assert torch.allclose(outputs["jump_norm"], torch.zeros_like(outputs["jump_norm"]))


def test_distributional_jump_bridge_forward_shapes_and_intervention() -> None:
    config = ModelConfig(
        price_features=8,
        metadata_features=6,
        event_embedding_dim=16,
        hidden_dim=32,
        latent_dim=4,
        jump_scale=0.2,
        jump_uses_metadata=False,
    )
    model = build_model("djb", config)
    x_pre = torch.randn(4, 5, 8)
    event_embedding = torch.randn(4, 16)
    metadata = torch.randn(4, 6)
    outputs = model(x_pre, event_embedding, metadata)
    assert outputs["prediction"].shape == (4, 3)
    assert outputs["event_delta"].shape == (4, 3)
    assert outputs["outcome_logvar"].shape == (4, 3)
    assert outputs["base_logvar"].shape == (4, 3)
    assert outputs["logvar_delta"].shape == (4, 3)
    assert outputs["z_event"].shape == (4, 38)
    targets = torch.randn(4, 3)
    loss, metrics = cebt_loss(
        outputs, targets, torch.tensor([1.0, 1.0, 0.0, 0.0]), LossWeights(nll_weight=0.1)
    )
    assert torch.isfinite(loss)
    assert metrics["nll"] == float(gaussian_nll_loss(outputs, targets).detach().cpu())
    no_jump = model(x_pre, event_embedding, metadata, intervention="no_jump")
    assert torch.allclose(no_jump["event_delta"], torch.zeros_like(no_jump["event_delta"]))
    assert torch.allclose(no_jump["logvar_delta"], torch.zeros_like(no_jump["logvar_delta"]))


def test_return_conservative_djb_preserves_abnormal_return_mean_delta() -> None:
    config = ModelConfig(
        price_features=8,
        metadata_features=6,
        event_embedding_dim=16,
        hidden_dim=32,
        jump_scale=0.2,
        jump_uses_metadata=False,
    )
    model = build_model("rc_djb", config)
    outputs = model(
        torch.randn(4, 5, 8),
        torch.randn(4, 16),
        torch.randn(4, 6),
    )
    assert torch.allclose(outputs["event_delta"][:, 0], torch.zeros(4))
    assert torch.allclose(outputs["prediction"][:, 0], outputs["base_prediction"][:, 0])
    assert torch.any(torch.abs(outputs["event_delta"][:, 1:]) > 0.0)


def test_event_jump_can_exclude_control_metadata_from_jump_generator() -> None:
    config = ModelConfig(
        price_features=8,
        metadata_features=6,
        event_embedding_dim=16,
        hidden_dim=32,
        jump_uses_metadata=False,
    )
    model = build_model("ejssm", config)
    first_layer = model.jump_generator[0]
    assert first_layer.in_features == 16
    embedding = torch.randn(2, 16)
    x_pre = torch.randn(2, 5, 8)
    metadata = torch.zeros(2, 6)
    metadata[1, 1] = 1.0
    outputs = model(x_pre, embedding, metadata)
    assert outputs["z_event"].shape == (2, 32)


def test_pairwise_rank_loss_rewards_correct_ordering_on_events() -> None:
    targets = torch.tensor([0.20, -0.10, 0.05, 0.50])
    is_event = torch.tensor([1.0, 1.0, 1.0, 0.0])
    good_scores = torch.tensor([0.30, -0.20, 0.10, -1.00])
    bad_scores = torch.tensor([-0.30, 0.20, -0.10, 1.00])
    good = pairwise_rank_loss(good_scores, targets, is_event)
    bad = pairwise_rank_loss(bad_scores, targets, is_event)
    assert good < bad


def test_rank_weight_changes_total_loss() -> None:
    outputs = {
        "prediction": torch.tensor(
            [
                [-0.30, 0.0, 0.0],
                [0.20, 0.0, 0.0],
                [-0.10, 0.0, 0.0],
                [0.00, 0.0, 0.0],
                [0.00, 0.0, 0.0],
            ]
        ),
        "event_delta": torch.zeros(5, 3),
        "z_event": torch.zeros(5, 4),
        "mu": torch.zeros(5, 4),
        "logvar": torch.zeros(5, 4),
    }
    targets = torch.tensor(
        [
            [0.30, 0.0, 0.0],
            [-0.20, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.00, 0.0, 0.0],
            [0.00, 0.0, 0.0],
        ]
    )
    is_event = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    base_loss, _ = cebt_loss(outputs, targets, is_event, LossWeights(rank_weight=0.0))
    rank_loss, metrics = cebt_loss(outputs, targets, is_event, LossWeights(rank_weight=0.5))
    assert metrics["rank"] >= 0.0
    assert rank_loss != base_loss


def test_standardized_huber_loss_uses_target_weights() -> None:
    predictions = torch.tensor([[0.10, 0.02, 0.50], [0.00, -0.01, -0.20]])
    targets = torch.tensor([[0.00, 0.00, 0.00], [0.10, -0.02, 0.10]])
    target_mean = torch.mean(targets, dim=0)
    target_std = torch.std(targets, dim=0, unbiased=False).clamp_min(1e-6)
    balanced = supervised_response_loss(
        predictions,
        targets,
        LossWeights(
            supervised_loss="huber",
            standardize_targets=True,
            target_weights=(1.0, 1.0, 1.0),
        ),
        target_mean=target_mean,
        target_std=target_std,
    )
    abnormal_weighted = supervised_response_loss(
        predictions,
        targets,
        LossWeights(
            supervised_loss="huber",
            standardize_targets=True,
            target_weights=(3.0, 1.0, 0.25),
        ),
        target_mean=target_mean,
        target_std=target_std,
    )
    assert torch.isfinite(balanced)
    assert torch.isfinite(abnormal_weighted)
    assert balanced != abnormal_weighted
