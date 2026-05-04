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
    operator_rank: int = 4
    operator_scale: float = 0.10
    jump_scale: float = 0.20
    jump_uses_metadata: bool = True
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
        intervention: str = "full",
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
        intervention: str = "full",
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
        intervention: str = "full",
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


class DisclosureOperatorTransformer(nn.Module):
    """Map disclosures to low-rank latent state transition operators.

    The event text does not enter the prediction head directly. Instead, it generates a
    sample-specific low-rank operator that perturbs the no-event latent state. This makes
    event influence observable as a state intervention.
    """

    def __init__(self, config: CEBTConfig) -> None:
        super().__init__()
        self.config = config
        self.no_event = NoEventDynamics(config)
        self.base_transition = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.operator_rank = config.operator_rank
        self.operator_scale = config.operator_scale
        self.operator_generator = nn.Sequential(
            nn.Linear(config.event_embedding_dim + config.metadata_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.hidden_dim,
                (2 * config.hidden_dim * config.operator_rank) + config.operator_rank,
            ),
        )
        self.operator_norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim + config.metadata_features),
            nn.Linear(config.hidden_dim + config.metadata_features, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        nn.init.eye_(self.base_transition.weight)

    def forward(
        self,
        x_pre: torch.Tensor,
        event_embedding: torch.Tensor,
        metadata: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base_prediction, state = self.no_event(x_pre, metadata)
        generated = self.operator_generator(torch.cat([event_embedding, metadata], dim=-1))
        batch = generated.shape[0]
        rank = self.operator_rank
        hidden = self.config.hidden_dim
        left_end = hidden * rank
        right_end = left_end + hidden * rank
        left = generated[:, :left_end].reshape(batch, hidden, rank)
        right = generated[:, left_end:right_end].reshape(batch, hidden, rank)
        gate = torch.tanh(generated[:, right_end:])
        left = torch.tanh(left)
        right = torch.tanh(right)
        projected = torch.bmm(state.unsqueeze(1), right).squeeze(1)
        low_rank_update = torch.bmm(left, (projected * gate).unsqueeze(-1)).squeeze(-1)
        low_rank_update = low_rank_update * (self.operator_scale / max(rank, 1) ** 0.5)
        base_state = self.base_transition(state)
        event_state = self.operator_norm(base_state + low_rank_update)
        prediction = self.head(torch.cat([event_state, metadata], dim=-1))
        return {
            "prediction": prediction,
            "base_prediction": base_prediction,
            "event_delta": prediction - base_prediction,
            "z_event": low_rank_update,
            "mu": state.new_zeros((state.shape[0], 1)),
            "logvar": state.new_zeros((state.shape[0], 1)),
            "operator_norm": torch.linalg.vector_norm(low_rank_update, dim=-1),
        }


class EventJumpStateSpaceModel(nn.Module):
    """Event-driven state-space model with disclosure-conditioned latent jumps.

    The model first estimates a no-event latent transition from pre-event market history. The
    disclosure then generates a sparse jump in that latent state before the outcome distribution is
    predicted. This makes event impact explicit as a learned state discontinuity.
    """

    def __init__(self, config: CEBTConfig) -> None:
        super().__init__()
        self.config = config
        self.jump_uses_metadata = config.jump_uses_metadata
        jump_input_dim = (
            config.event_embedding_dim + config.metadata_features
            if self.jump_uses_metadata
            else config.event_embedding_dim
        )
        self.input_projection = nn.Linear(config.price_features, config.hidden_dim)
        self.encoder = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
            batch_first=True,
        )
        self.metadata_projection = nn.Linear(config.metadata_features, config.hidden_dim)
        self.base_gate = nn.Sequential(
            nn.Linear(config.hidden_dim + config.metadata_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.base_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        self.jump_generator = nn.Sequential(
            nn.Linear(jump_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
        )
        self.jump_norm = nn.LayerNorm(config.hidden_dim)
        self.outcome_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim + config.metadata_features),
            nn.Linear(config.hidden_dim + config.metadata_features, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim + config.metadata_features),
            nn.Linear(config.hidden_dim + config.metadata_features, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(
        self,
        x_pre: torch.Tensor,
        event_embedding: torch.Tensor,
        metadata: torch.Tensor,
        intervention: str = "full",
    ) -> dict[str, torch.Tensor]:
        encoded_inputs = self.input_projection(x_pre)
        _, hidden = self.encoder(encoded_inputs)
        state = hidden[-1] + self.metadata_projection(metadata)
        gate = self.base_gate(torch.cat([state, metadata], dim=-1))
        no_event_state = state * (1.0 + 0.1 * gate)
        base_prediction = self.base_head(no_event_state)

        jump_input = (
            torch.cat([event_embedding, metadata], dim=-1)
            if self.jump_uses_metadata
            else event_embedding
        )
        jump_params = self.jump_generator(jump_input)
        jump_direction, jump_gate = torch.chunk(jump_params, chunks=2, dim=-1)
        jump = torch.tanh(jump_direction) * torch.sigmoid(jump_gate) * self.config.jump_scale
        if intervention == "no_jump":
            jump = torch.zeros_like(jump)
        jumped_state = self.jump_norm(no_event_state + jump)
        outcome_input = torch.cat([jumped_state, metadata], dim=-1)
        prediction = self.outcome_head(outcome_input)
        outcome_logvar = torch.clamp(self.logvar_head(outcome_input), min=-7.0, max=3.0)
        return {
            "prediction": prediction,
            "base_prediction": base_prediction,
            "event_delta": prediction - base_prediction,
            "z_event": jump,
            "mu": jump.new_zeros((jump.shape[0], 1)),
            "logvar": jump.new_zeros((jump.shape[0], 1)),
            "outcome_logvar": outcome_logvar,
            "jump_norm": torch.linalg.vector_norm(jump, dim=-1),
        }


class DistributionalJumpBridgeModel(nn.Module):
    """Disclosure-conditioned transport from a no-event distribution to an event distribution.

    EJSSM applies a latent state jump and then predicts a distribution from the jumped state.
    DJB makes the distributional intervention explicit: the market history first defines a
    no-event Gaussian forecast, and the disclosure bridge then applies bounded shifts to both the
    predictive mean and log-variance. This exposes whether text is changing the point forecast,
    the uncertainty estimate, or both.
    """

    def __init__(self, config: CEBTConfig) -> None:
        super().__init__()
        self.config = config
        self.jump_uses_metadata = config.jump_uses_metadata
        self.transport_return_mean = True
        jump_input_dim = (
            config.event_embedding_dim + config.metadata_features
            if self.jump_uses_metadata
            else config.event_embedding_dim
        )
        self.input_projection = nn.Linear(config.price_features, config.hidden_dim)
        self.encoder = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
            batch_first=True,
        )
        self.metadata_projection = nn.Linear(config.metadata_features, config.hidden_dim)
        self.base_gate = nn.Sequential(
            nn.Linear(config.hidden_dim + config.metadata_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.base_distribution = nn.Sequential(
            nn.LayerNorm(config.hidden_dim + config.metadata_features),
            nn.Linear(config.hidden_dim + config.metadata_features, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim * 2),
        )
        self.bridge_encoder = nn.Sequential(
            nn.Linear(jump_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.state_bridge = nn.Linear(config.hidden_dim, config.hidden_dim * 2)
        self.distribution_bridge = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2 + config.metadata_features),
            nn.Linear(config.hidden_dim * 2 + config.metadata_features, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim * 2),
        )
        self.bridge_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        x_pre: torch.Tensor,
        event_embedding: torch.Tensor,
        metadata: torch.Tensor,
        intervention: str = "full",
    ) -> dict[str, torch.Tensor]:
        encoded_inputs = self.input_projection(x_pre)
        _, hidden = self.encoder(encoded_inputs)
        state = hidden[-1] + self.metadata_projection(metadata)
        gate = self.base_gate(torch.cat([state, metadata], dim=-1))
        no_event_state = state * (1.0 + 0.1 * gate)
        base_params = self.base_distribution(torch.cat([no_event_state, metadata], dim=-1))
        base_prediction, base_logvar = torch.chunk(base_params, chunks=2, dim=-1)
        base_logvar = torch.clamp(base_logvar, min=-7.0, max=3.0)

        bridge_input = (
            torch.cat([event_embedding, metadata], dim=-1)
            if self.jump_uses_metadata
            else event_embedding
        )
        bridge_hidden = self.bridge_encoder(bridge_input)
        state_params = self.state_bridge(bridge_hidden)
        state_direction, state_gate = torch.chunk(state_params, chunks=2, dim=-1)
        state_jump = (
            torch.tanh(state_direction) * torch.sigmoid(state_gate) * self.config.jump_scale
        )
        if intervention == "no_jump":
            state_jump = torch.zeros_like(state_jump)
            bridge_hidden = torch.zeros_like(bridge_hidden)
        bridged_state = self.bridge_norm(no_event_state + state_jump)

        distribution_delta = self.distribution_bridge(
            torch.cat([bridged_state, bridge_hidden, metadata], dim=-1)
        )
        mean_delta, logvar_delta = torch.chunk(distribution_delta, chunks=2, dim=-1)
        mean_delta = torch.tanh(mean_delta) * self.config.jump_scale
        logvar_delta = torch.tanh(logvar_delta) * self.config.jump_scale
        if not self.transport_return_mean:
            mean_delta = mean_delta.clone()
            mean_delta[:, 0] = 0.0
        if intervention == "no_jump":
            mean_delta = torch.zeros_like(mean_delta)
            logvar_delta = torch.zeros_like(logvar_delta)
        prediction = base_prediction + mean_delta
        outcome_logvar = torch.clamp(base_logvar + logvar_delta, min=-7.0, max=3.0)
        return {
            "prediction": prediction,
            "base_prediction": base_prediction,
            "event_delta": mean_delta,
            "z_event": torch.cat([state_jump, mean_delta, logvar_delta], dim=-1),
            "mu": state_jump.new_zeros((state_jump.shape[0], 1)),
            "logvar": state_jump.new_zeros((state_jump.shape[0], 1)),
            "outcome_logvar": outcome_logvar,
            "base_logvar": base_logvar,
            "logvar_delta": logvar_delta,
            "jump_norm": torch.linalg.vector_norm(state_jump, dim=-1),
        }


class ReturnConservativeDistributionalJumpBridgeModel(DistributionalJumpBridgeModel):
    """DJB variant that preserves the no-event abnormal-return mean.

    The disclosure bridge may still alter volatility, volume, and all predictive variances, but it
    cannot directly shift the abnormal-return point forecast. This tests whether disclosure text is
    more useful as a response-distribution operator than as a cross-sectional return-rank signal.
    """

    def __init__(self, config: CEBTConfig) -> None:
        super().__init__(config)
        self.transport_return_mean = False


def build_model(model_name: str, config: CEBTConfig) -> nn.Module:
    if model_name in {"cebt", "cebt_no_controls"}:
        return CounterfactualEventBottleneckTransformer(config)
    if model_name == "no_event":
        return NoEventOnlyModel(config)
    if model_name == "text_only":
        return TextOnlyMLP(config)
    if model_name == "concat":
        return ConcatFusionModel(config)
    if model_name in {"dot", "disclosure_operator"}:
        return DisclosureOperatorTransformer(config)
    if model_name in {"ejssm", "event_jump_state_space"}:
        return EventJumpStateSpaceModel(config)
    if model_name in {"djb", "distributional_jump_bridge"}:
        return DistributionalJumpBridgeModel(config)
    if model_name in {"rc_djb", "return_conservative_djb"}:
        return ReturnConservativeDistributionalJumpBridgeModel(config)
    raise ValueError(f"Unknown model_name: {model_name}")
