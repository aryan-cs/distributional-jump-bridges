from __future__ import annotations

import torch

from cebt.models.cebt import CEBTConfig, build_model
from cebt.training.losses import LossWeights, cebt_loss


def test_cebt_forward_shapes_and_finite_loss() -> None:
    config = CEBTConfig(
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


def test_bottleneck_dimension_changes_latent_shape() -> None:
    small = build_model("cebt", CEBTConfig(event_embedding_dim=16, hidden_dim=32, latent_dim=2))
    large = build_model("cebt", CEBTConfig(event_embedding_dim=16, hidden_dim=32, latent_dim=8))
    x = torch.randn(2, 5, 8)
    e = torch.randn(2, 16)
    m = torch.randn(2, 6)
    assert small(x, e, m)["mu"].shape[-1] == 2
    assert large(x, e, m)["mu"].shape[-1] == 8
