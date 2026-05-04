"""CEBT losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    supervised_weight: float = 1.0
    supervised_loss: str = "mse"
    standardize_targets: bool = False
    target_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    huber_delta: float = 1.0
    nll_weight: float = 0.0
    control_delta_weight: float = 0.25
    kl_weight: float = 0.01
    sparsity_weight: float = 0.001
    consistency_weight: float = 0.05
    rank_weight: float = 0.0
    rank_temperature: float = 0.02

    @classmethod
    def from_dict(cls, row: dict) -> LossWeights:
        values = {key: row[key] for key in cls.__dataclass_fields__ if key in row}
        if "target_weights" in values:
            values["target_weights"] = tuple(float(value) for value in values["target_weights"])
        return cls(**values)


def cebt_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    is_event: torch.Tensor,
    weights: LossWeights,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    supervised = supervised_response_loss(
        outputs["prediction"],
        targets,
        weights,
        target_mean=target_mean,
        target_std=target_std,
    )
    nll = gaussian_nll_loss(outputs, targets)
    control_mask = (is_event <= 0.5).float().unsqueeze(-1)
    if torch.sum(control_mask) > 0:
        control_delta = torch.sum((outputs["event_delta"] * control_mask) ** 2) / torch.sum(
            control_mask
        )
    else:
        control_delta = outputs["event_delta"].new_tensor(0.0)
    mu = outputs["mu"]
    logvar = outputs["logvar"]
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    sparsity = torch.mean(torch.abs(outputs["z_event"]))
    event_delta = outputs["event_delta"]
    if event_delta.shape[0] > 1:
        consistency = torch.mean(torch.var(event_delta, dim=0))
    else:
        consistency = event_delta.new_tensor(0.0)
    rank = pairwise_rank_loss(
        outputs["prediction"][:, 0],
        targets[:, 0],
        is_event,
        temperature=weights.rank_temperature,
    )
    total = (
        weights.supervised_weight * supervised
        + weights.nll_weight * nll
        + weights.control_delta_weight * control_delta
        + weights.kl_weight * kl
        + weights.sparsity_weight * sparsity
        + weights.consistency_weight * consistency
        + weights.rank_weight * rank
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "supervised": float(supervised.detach().cpu()),
        "nll": float(nll.detach().cpu()),
        "control_delta": float(control_delta.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "sparsity": float(sparsity.detach().cpu()),
        "consistency": float(consistency.detach().cpu()),
        "rank": float(rank.detach().cpu()),
    }
    return total, metrics


def supervised_response_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: LossWeights,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights.standardize_targets:
        if target_mean is None or target_std is None:
            raise ValueError("Target standardization requested without target_mean/target_std.")
        predictions = (predictions - target_mean) / target_std
        targets = (targets - target_mean) / target_std
    target_weights = predictions.new_tensor(weights.target_weights)
    if target_weights.numel() != predictions.shape[-1]:
        raise ValueError(
            f"target_weights has {target_weights.numel()} entries, "
            f"but predictions have {predictions.shape[-1]} targets."
        )
    if weights.supervised_loss == "mse":
        loss = (predictions - targets) ** 2
    elif weights.supervised_loss in {"huber", "smooth_l1"}:
        loss = F.smooth_l1_loss(
            predictions,
            targets,
            reduction="none",
            beta=max(float(weights.huber_delta), 1e-6),
        )
    else:
        raise ValueError(f"Unknown supervised_loss: {weights.supervised_loss}")
    return torch.mean(loss * target_weights.view(1, -1))


def gaussian_nll_loss(outputs: dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    if "outcome_logvar" not in outputs:
        return targets.new_tensor(0.0)
    logvar = torch.clamp(outputs["outcome_logvar"], min=-7.0, max=3.0)
    squared_error = (targets - outputs["prediction"]) ** 2
    return 0.5 * torch.mean(torch.exp(-logvar) * squared_error + logvar)


def pairwise_rank_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    is_event: torch.Tensor,
    temperature: float = 0.02,
) -> torch.Tensor:
    """Pairwise logistic ranking loss over real disclosure events in a batch."""

    event_mask = is_event >= 0.5
    event_scores = scores[event_mask]
    event_targets = targets[event_mask]
    if event_scores.shape[0] < 2:
        return scores.new_tensor(0.0)
    target_diff = event_targets.unsqueeze(0) - event_targets.unsqueeze(1)
    score_diff = event_scores.unsqueeze(0) - event_scores.unsqueeze(1)
    pair_mask = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
    pair_mask = pair_mask & (torch.abs(target_diff) > 1e-8)
    if not torch.any(pair_mask):
        return scores.new_tensor(0.0)
    direction = torch.sign(target_diff[pair_mask])
    scaled_margin = direction * score_diff[pair_mask] / max(float(temperature), 1e-6)
    return F.softplus(-scaled_margin).mean()
