"""Counterfactual Event Bottleneck Transformer model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CEBTConfig:
    price_features: int = 8
    metadata_features: int = 6
    event_embedding_dim: int = 256
    hidden_dim: int = 96
    latent_dim: int = 8
    output_dim: int = 3
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, row: dict) -> CEBTConfig:
        return cls(**{key: row[key] for key in cls.__dataclass_fields__ if key in row})


class NoEventDynamics(nn.Module):
    """Predict ordinary future movement from pre-event numeric windows."""

    def __init__(self, config: CEBTConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(config.price_features, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=4,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.metadata_projection = nn.Linear(config.metadata_features, config.hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(
        self, x_pre: torch.Tensor, metadata: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_projection(x_pre)
        encoded = self.encoder(hidden)
        pooled = encoded.mean(dim=1) + self.metadata_projection(metadata)
        return self.head(pooled), pooled


class EventBottleneck(nn.Module):
    """Compact stochastic latent representation of disclosure-induced change."""

    def __init__(self, config: CEBTConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.event_embedding_dim + config.metadata_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(
        self,
        event_embedding: torch.Tensor,
        metadata: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(torch.cat([event_embedding, metadata], dim=-1))
        mu = self.mu(hidden)
        logvar = torch.clamp(self.logvar(hidden), min=-8.0, max=6.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_event = mu + eps * std
        else:
            z_event = mu
        return z_event, mu, logvar


class ResidualOutcomeHead(nn.Module):
    """Predict event-induced residual added to no-event dynamics."""

    def __init__(self, config: CEBTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                config.hidden_dim + config.latent_dim + config.metadata_features, config.hidden_dim
            ),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(
        self,
        no_event_state: torch.Tensor,
        z_event: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([no_event_state, z_event, metadata], dim=-1))


class CounterfactualEventBottleneckTransformer(nn.Module):
    def __init__(self, config: CEBTConfig) -> None:
        super().__init__()
        self.config = config
        self.no_event = NoEventDynamics(config)
        self.event_bottleneck = EventBottleneck(config)
        self.residual_head = ResidualOutcomeHead(config)

    def forward(
        self,
        x_pre: torch.Tensor,
        event_embedding: torch.Tensor,
        metadata: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base_prediction, no_event_state = self.no_event(x_pre, metadata)
        z_event, mu, logvar = self.event_bottleneck(event_embedding, metadata)
        event_delta = self.residual_head(no_event_state, z_event, metadata)
        prediction = base_prediction + event_delta
        return {
            "prediction": prediction,
            "base_prediction": base_prediction,
            "event_delta": event_delta,
            "z_event": z_event,
            "mu": mu,
            "logvar": logvar,
        }


class NoEventOnlyModel(nn.Module):
    def __init__(self, config: CEBTConfig) -> None:
        super().__init__()
        self.no_event = NoEventDynamics(config)

    def forward(
        self,
        x_pre: torch.Tensor,
        event_embedding: torch.Tensor,
        metadata: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prediction, state = self.no_event(x_pre, metadata)
        return {
            "prediction": prediction,
            "base_prediction": prediction,
            "event_delta": torch.zeros_like(prediction),
            "z_event": state.new_zeros((state.shape[0], 1)),
            "mu": state.new_zeros((state.shape[0], 1)),
            "logvar": state.new_zeros((state.shape[0], 1)),
        }


class TextOnlyMLP(nn.Module):
    def __init__(self, config: CEBTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.event_embedding_dim + config.metadata_features, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(
        self,
        x_pre: torch.Tensor,
        event_embedding: torch.Tensor,
        metadata: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prediction = self.net(torch.cat([event_embedding, metadata], dim=-1))
        return {
            "prediction": prediction,
            "base_prediction": torch.zeros_like(prediction),
            "event_delta": prediction,
            "z_event": prediction.new_zeros((prediction.shape[0], 1)),
            "mu": prediction.new_zeros((prediction.shape[0], 1)),
            "logvar": prediction.new_zeros((prediction.shape[0], 1)),
        }


class ConcatFusionModel(nn.Module):
    def __init__(self, config: CEBTConfig) -> None:
        super().__init__()
        self.no_event = NoEventDynamics(config)
        self.head = nn.Sequential(
            nn.Linear(
                config.hidden_dim + config.event_embedding_dim + config.metadata_features,
                config.hidden_dim,
            ),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(
        self,
        x_pre: torch.Tensor,
        event_embedding: torch.Tensor,
        metadata: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base, state = self.no_event(x_pre, metadata)
        prediction = self.head(torch.cat([state, event_embedding, metadata], dim=-1))
        return {
            "prediction": prediction,
            "base_prediction": base,
            "event_delta": prediction - base,
            "z_event": prediction.new_zeros((prediction.shape[0], 1)),
            "mu": prediction.new_zeros((prediction.shape[0], 1)),
            "logvar": prediction.new_zeros((prediction.shape[0], 1)),
        }


def build_model(model_name: str, config: CEBTConfig) -> nn.Module:
    if model_name in {"cebt", "cebt_no_controls"}:
        return CounterfactualEventBottleneckTransformer(config)
    if model_name == "no_event":
        return NoEventOnlyModel(config)
    if model_name == "text_only":
        return TextOnlyMLP(config)
    if model_name == "concat":
        return ConcatFusionModel(config)
    raise ValueError(f"Unknown model_name: {model_name}")
