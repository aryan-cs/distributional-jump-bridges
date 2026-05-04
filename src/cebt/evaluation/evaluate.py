"""Model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from cebt.evaluation.bootstrap import bootstrap_ci, clustered_bootstrap_ci, leave_one_group_out
from cebt.evaluation.leakage import validate_feature_metadata
from cebt.evaluation.metrics import (
    abnormal_return_spread,
    balanced_accuracy_from_scores,
    calibration_error,
    mse,
    rank_ic,
)
from cebt.models.cebt import CEBTConfig, build_model
from cebt.training.dataset import CEBTTensorDataset
from cebt.training.train import auto_device
from cebt.utils.io import read_jsonl, write_json, write_jsonl


def evaluate_model(
    config: dict[str, Any],
    feature_path: str | Path,
    metadata_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    split: int = 2,
    intervention: str = "full",
) -> dict[str, Any]:
    if intervention not in {"full", "no_jump", "shuffle_event", "zero_event"}:
        raise ValueError(f"Unknown intervention: {intervention}")
    validate_feature_metadata(metadata_path)
    device = auto_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = CEBTConfig.from_dict(checkpoint.get("model_config", config.get("model", {})))
    model = build_model(checkpoint.get("model_name", "cebt"), model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    dataset = CEBTTensorDataset(feature_path, split=split)
    if len(dataset) == 0:
        dataset = CEBTTensorDataset(feature_path, split=1)
    loader = DataLoader(dataset, batch_size=int(config.get("training", {}).get("batch_size", 32)))
    shuffled_embeddings = (
        _shuffled_embeddings(dataset, config) if intervention == "shuffle_event" else None
    )
    (
        predictions,
        targets,
        deltas,
        latents,
        outcome_logvars,
        base_logvars,
        logvar_deltas,
        is_event,
        dataset_indices,
    ) = _predict(
        model,
        loader,
        device,
        intervention=intervention,
        shuffled_embeddings=shuffled_embeddings,
    )
    metadata_rows = read_jsonl(metadata_path)
    source_rows = [
        metadata_rows[source_idx] if source_idx < len(metadata_rows) else {}
        for source_idx in dataset_indices.tolist()
    ]
    paired_values = np.column_stack([predictions.reshape(predictions.shape[0], -1), targets])
    rank_values = np.column_stack([predictions[:, 0], targets[:, 0]])
    ticker_groups = np.asarray([row.get("ticker", "UNKNOWN") for row in source_rows], dtype=object)
    month_groups = np.asarray(
        [str(row.get("label_start_date", "UNKNOWN"))[:7] for row in source_rows],
        dtype=object,
    )
    metrics = {
        "rows": int(targets.shape[0]),
        "intervention": intervention,
        "mse": mse(predictions, targets),
        "abnormal_return_balanced_accuracy": balanced_accuracy_from_scores(
            predictions[:, 0], targets[:, 0]
        ),
        "abnormal_return_rank_ic": rank_ic(predictions[:, 0], targets[:, 0]),
        "abnormal_return_spread": abnormal_return_spread(predictions[:, 0], targets[:, 0]),
        "calibration_error": calibration_error(predictions[:, 0], targets[:, 0]),
        "mean_abs_event_delta_true_events": _masked_abs_mean(deltas, is_event >= 0.5),
        "mean_abs_event_delta_controls": _masked_abs_mean(deltas, is_event < 0.5),
        "mean_abs_latent_event_true_events": _masked_abs_mean(latents, is_event >= 0.5),
        "mean_abs_latent_event_controls": _masked_abs_mean(latents, is_event < 0.5),
        "event_mse": _masked_mse(predictions, targets, is_event >= 0.5),
        "control_mse": _masked_mse(predictions, targets, is_event < 0.5),
        "event_rank_ic": _masked_rank_ic(predictions[:, 0], targets[:, 0], is_event >= 0.5),
        "control_rank_ic": _masked_rank_ic(predictions[:, 0], targets[:, 0], is_event < 0.5),
        "latent_jump_auc": _binary_auc(_row_abs_mean(latents), is_event),
        "latent_jump_paired_gap": _paired_event_control_gap(source_rows, _row_abs_mean(latents)),
        "mse_ci": bootstrap_ci(
            paired_values,
            _mse_from_paired_rows(targets.shape[1]),
            n_boot=1000,
            seed=int(config.get("seed", 7)),
        ),
        "rank_ic_ci": bootstrap_ci(
            rank_values,
            lambda rows: rank_ic(rows[:, 0], rows[:, 1]) or 0.0,
            n_boot=1000,
            seed=int(config.get("seed", 7)),
        ),
        "mse_ci_ticker_cluster": clustered_bootstrap_ci(
            paired_values,
            ticker_groups,
            _mse_from_paired_rows(targets.shape[1]),
            n_boot=1000,
            seed=int(config.get("seed", 7)) + 101,
        ),
        "rank_ic_ci_ticker_cluster": clustered_bootstrap_ci(
            rank_values,
            ticker_groups,
            lambda rows: rank_ic(rows[:, 0], rows[:, 1]) or 0.0,
            n_boot=1000,
            seed=int(config.get("seed", 7)) + 102,
        ),
        "rank_ic_leave_one_ticker_out": leave_one_group_out(
            rank_values,
            ticker_groups,
            lambda rows: rank_ic(rows[:, 0], rows[:, 1]) or 0.0,
        ),
        "mse_ci_month_cluster": clustered_bootstrap_ci(
            paired_values,
            month_groups,
            _mse_from_paired_rows(targets.shape[1]),
            n_boot=1000,
            seed=int(config.get("seed", 7)) + 103,
        ),
        "rank_ic_ci_month_cluster": clustered_bootstrap_ci(
            rank_values,
            month_groups,
            lambda rows: rank_ic(rows[:, 0], rows[:, 1]) or 0.0,
            n_boot=1000,
            seed=int(config.get("seed", 7)) + 104,
        ),
    }
    if outcome_logvars is not None:
        metrics.update(_probabilistic_metrics(predictions, targets, outcome_logvars, is_event))
        metrics.update(
            _return_logvar_diagnostics(
                predictions,
                targets,
                outcome_logvars,
                base_logvars,
                logvar_deltas,
                is_event,
            )
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_stem = _output_stem(checkpoint.get("model_name", "model"), intervention)
    write_json(output / f"{output_stem}_eval_metrics.json", metrics)
    prediction_rows = []
    for local_idx, source_row in enumerate(source_rows):
        prediction_rows.append(
            {
                **source_row,
                "model": checkpoint.get("model_name", "model"),
                "intervention": intervention,
                "prediction_abnormal_return": float(predictions[local_idx, 0]),
                "prediction_volatility_jump": float(predictions[local_idx, 1]),
                "prediction_volume_jump": float(predictions[local_idx, 2]),
                "target_abnormal_return": float(targets[local_idx, 0]),
                "target_volatility_jump": float(targets[local_idx, 1]),
                "target_volume_jump": float(targets[local_idx, 2]),
                "event_delta_abs_mean": float(np.mean(np.abs(deltas[local_idx]))),
                "latent_event_abs_mean": float(np.mean(np.abs(latents[local_idx]))),
            }
        )
        if outcome_logvars is not None:
            row = prediction_rows[-1]
            row["prediction_logvar_abnormal_return"] = float(outcome_logvars[local_idx, 0])
            row["prediction_logvar_volatility_jump"] = float(outcome_logvars[local_idx, 1])
            row["prediction_logvar_volume_jump"] = float(outcome_logvars[local_idx, 2])
            row["prediction_gaussian_nll"] = float(
                _row_gaussian_nll(
                    predictions[local_idx],
                    targets[local_idx],
                    outcome_logvars[local_idx],
                )
            )
            if base_logvars is not None:
                row["base_logvar_abnormal_return"] = float(base_logvars[local_idx, 0])
            if logvar_deltas is not None:
                row["delta_logvar_abnormal_return"] = float(logvar_deltas[local_idx, 0])
    prediction_path = output / f"{output_stem}_predictions.jsonl"
    write_jsonl(prediction_path, prediction_rows)
    return metrics


def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    intervention: str = "full",
    shuffled_embeddings: torch.Tensor | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
]:
    model.eval()
    predictions = []
    targets = []
    deltas = []
    latents = []
    outcome_logvars = []
    base_logvars = []
    logvar_deltas = []
    is_event = []
    indices = []
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            x_pre = batch["x_pre"].to(device)
            event_embedding = batch["event_embedding"].to(device)
            batch_size = int(event_embedding.shape[0])
            if shuffled_embeddings is not None:
                event_embedding = shuffled_embeddings[cursor : cursor + batch_size].to(device)
            if intervention == "zero_event":
                event_embedding = torch.zeros_like(event_embedding)
            cursor += batch_size
            metadata = batch["metadata"].to(device)
            outputs = _forward_model(model, x_pre, event_embedding, metadata, intervention)
            predictions.append(outputs["prediction"].detach().cpu().numpy())
            deltas.append(outputs["event_delta"].detach().cpu().numpy())
            latents.append(outputs["z_event"].detach().cpu().numpy())
            if "outcome_logvar" in outputs:
                outcome_logvars.append(outputs["outcome_logvar"].detach().cpu().numpy())
            if "base_logvar" in outputs:
                base_logvars.append(outputs["base_logvar"].detach().cpu().numpy())
            if "logvar_delta" in outputs:
                logvar_deltas.append(outputs["logvar_delta"].detach().cpu().numpy())
            targets.append(batch["y"].numpy())
            is_event.append(batch["is_event"].numpy())
            indices.append(batch["index"].numpy())
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(deltas, axis=0),
        np.concatenate(latents, axis=0),
        np.concatenate(outcome_logvars, axis=0) if outcome_logvars else None,
        np.concatenate(base_logvars, axis=0) if base_logvars else None,
        np.concatenate(logvar_deltas, axis=0) if logvar_deltas else None,
        np.concatenate(is_event, axis=0),
        np.concatenate(indices, axis=0),
    )


def _forward_model(
    model: torch.nn.Module,
    x_pre: torch.Tensor,
    event_embedding: torch.Tensor,
    metadata: torch.Tensor,
    intervention: str,
) -> dict[str, torch.Tensor]:
    if intervention == "no_jump":
        try:
            return model(x_pre, event_embedding, metadata, intervention=intervention)
        except TypeError:
            pass
    return model(x_pre, event_embedding, metadata)


def _masked_abs_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(np.mean(np.abs(values[mask])))


def _row_abs_mean(values: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(values), axis=1)


def _masked_mse(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return mse(predictions[mask], targets[mask])


def _masked_rank_ic(scores: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return rank_ic(scores[mask], targets[mask])


def _mse_from_paired_rows(target_dim: int):
    def metric(rows: np.ndarray) -> float:
        return mse(rows[:, :target_dim], rows[:, target_dim:])

    return metric


def _row_gaussian_nll(
    prediction: np.ndarray,
    target: np.ndarray,
    logvar: np.ndarray,
) -> float:
    return float(0.5 * np.mean(np.exp(-logvar) * (target - prediction) ** 2 + logvar))


def _probabilistic_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    logvars: np.ndarray,
    is_event: np.ndarray,
) -> dict[str, float | None]:
    event_mask = is_event >= 0.5
    control_mask = is_event < 0.5
    return {
        "gaussian_nll": _masked_gaussian_nll(predictions, targets, logvars),
        "event_gaussian_nll": _masked_gaussian_nll(
            predictions,
            targets,
            logvars,
            event_mask,
        ),
        "control_gaussian_nll": _masked_gaussian_nll(
            predictions,
            targets,
            logvars,
            control_mask,
        ),
        "gaussian_80_coverage": _masked_gaussian_coverage(
            predictions,
            targets,
            logvars,
            z_score=1.281551565545,
        ),
        "event_gaussian_80_coverage": _masked_gaussian_coverage(
            predictions,
            targets,
            logvars,
            mask=event_mask,
            z_score=1.281551565545,
        ),
        "control_gaussian_80_coverage": _masked_gaussian_coverage(
            predictions,
            targets,
            logvars,
            mask=control_mask,
            z_score=1.281551565545,
        ),
        "gaussian_95_coverage": _masked_gaussian_coverage(
            predictions,
            targets,
            logvars,
            z_score=1.959963984540,
        ),
        "event_gaussian_95_coverage": _masked_gaussian_coverage(
            predictions,
            targets,
            logvars,
            mask=event_mask,
            z_score=1.959963984540,
        ),
        "control_gaussian_95_coverage": _masked_gaussian_coverage(
            predictions,
            targets,
            logvars,
            mask=control_mask,
            z_score=1.959963984540,
        ),
        "mean_predictive_std": _masked_predictive_std(logvars),
        "event_mean_predictive_std": _masked_predictive_std(logvars, event_mask),
        "control_mean_predictive_std": _masked_predictive_std(logvars, control_mask),
    }


def _return_logvar_diagnostics(
    predictions: np.ndarray,
    targets: np.ndarray,
    logvars: np.ndarray,
    base_logvars: np.ndarray | None,
    logvar_deltas: np.ndarray | None,
    is_event: np.ndarray,
) -> dict[str, float | None]:
    return_logvar = logvars[:, 0]
    abnormal_target = targets[:, 0]
    abnormal_error = np.abs(abnormal_target - predictions[:, 0])
    event_mask = is_event >= 0.5
    control_mask = is_event < 0.5
    metrics: dict[str, float | None] = {
        "return_logvar_signed_rank_ic": rank_ic(return_logvar, abnormal_target),
        "return_logvar_abs_error_rank_ic": rank_ic(return_logvar, abnormal_error),
    }
    if base_logvars is not None:
        metrics["base_return_logvar_signed_rank_ic"] = rank_ic(base_logvars[:, 0], abnormal_target)
    if logvar_deltas is not None:
        return_delta = logvar_deltas[:, 0]
        metrics.update(
            {
                "return_logvar_delta_signed_rank_ic": rank_ic(return_delta, abnormal_target),
                "return_logvar_delta_abs_error_rank_ic": rank_ic(return_delta, abnormal_error),
                "mean_abs_return_logvar_delta_true_events": _masked_abs_mean(
                    return_delta.reshape(-1, 1),
                    event_mask,
                ),
                "mean_abs_return_logvar_delta_controls": _masked_abs_mean(
                    return_delta.reshape(-1, 1),
                    control_mask,
                ),
            }
        )
    return metrics


def _masked_gaussian_nll(
    predictions: np.ndarray,
    targets: np.ndarray,
    logvars: np.ndarray,
    mask: np.ndarray | None = None,
) -> float | None:
    if mask is not None:
        if not np.any(mask):
            return None
        predictions = predictions[mask]
        targets = targets[mask]
        logvars = logvars[mask]
    if predictions.size == 0:
        return None
    return _row_gaussian_nll(predictions, targets, logvars)


def _masked_gaussian_coverage(
    predictions: np.ndarray,
    targets: np.ndarray,
    logvars: np.ndarray,
    z_score: float,
    mask: np.ndarray | None = None,
) -> float | None:
    if mask is not None:
        if not np.any(mask):
            return None
        predictions = predictions[mask]
        targets = targets[mask]
        logvars = logvars[mask]
    if predictions.size == 0:
        return None
    std = np.sqrt(np.exp(logvars))
    errors = np.abs(targets - predictions)
    return float(np.mean(errors <= z_score * std))


def _masked_predictive_std(
    logvars: np.ndarray,
    mask: np.ndarray | None = None,
) -> float | None:
    if mask is not None:
        if not np.any(mask):
            return None
        logvars = logvars[mask]
    if logvars.size == 0:
        return None
    return float(np.mean(np.sqrt(np.exp(logvars))))


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels >= 0.5
    positives = scores[labels]
    negatives = scores[~labels]
    if positives.size == 0 or negatives.size == 0:
        return None
    values = np.concatenate([positives, negatives])
    ranks = _average_ranks(values)
    pos_ranks = ranks[: positives.size]
    auc = (np.sum(pos_ranks) - positives.size * (positives.size + 1) / 2.0) / (
        positives.size * negatives.size
    )
    return float(auc)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    return ranks


def _paired_event_control_gap(
    source_rows: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[str, float] | None:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row, score in zip(source_rows, scores, strict=False):
        event_id = str(row.get("event_id", ""))
        if not event_id:
            continue
        key = "event" if row.get("control_type") == "real_event" else "control"
        grouped.setdefault(event_id, {"event": [], "control": []})[key].append(float(score))
    gaps = [
        float(np.mean(values["event"]) - np.mean(values["control"]))
        for values in grouped.values()
        if values["event"] and values["control"]
    ]
    if not gaps:
        return None
    gap_values = np.asarray(gaps, dtype=float).reshape(-1, 1)
    return bootstrap_ci(gap_values, lambda rows: float(np.mean(rows[:, 0])), n_boot=1000, seed=17)


def _shuffled_embeddings(dataset: CEBTTensorDataset, config: dict[str, Any]) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 7)) + 909)
    permutation = torch.randperm(len(dataset), generator=generator)
    return dataset.event_embedding[permutation]


def _output_stem(model_name: str, intervention: str) -> str:
    return model_name if intervention == "full" else f"{model_name}_{intervention}"
