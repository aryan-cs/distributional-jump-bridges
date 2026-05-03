"""Model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from cebt.evaluation.bootstrap import bootstrap_ci
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
) -> dict[str, Any]:
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
    predictions, targets, deltas, is_event, dataset_indices = _predict(model, loader, device)
    metadata_rows = read_jsonl(metadata_path)
    metrics = {
        "rows": int(targets.shape[0]),
        "mse": mse(predictions, targets),
        "abnormal_return_balanced_accuracy": balanced_accuracy_from_scores(
            predictions[:, 0], targets[:, 0]
        ),
        "abnormal_return_rank_ic": rank_ic(predictions[:, 0], targets[:, 0]),
        "abnormal_return_spread": abnormal_return_spread(predictions[:, 0], targets[:, 0]),
        "calibration_error": calibration_error(predictions[:, 0], targets[:, 0]),
        "mean_abs_event_delta_true_events": _masked_abs_mean(deltas, is_event >= 0.5),
        "mean_abs_event_delta_controls": _masked_abs_mean(deltas, is_event < 0.5),
        "mse_ci": bootstrap_ci(
            np.column_stack([predictions.reshape(predictions.shape[0], -1), targets]),
            lambda rows: mse(rows[:, : targets.shape[1]], rows[:, targets.shape[1] :]),
            n_boot=1000,
            seed=int(config.get("seed", 7)),
        ),
        "rank_ic_ci": bootstrap_ci(
            np.column_stack([predictions[:, 0], targets[:, 0]]),
            lambda rows: rank_ic(rows[:, 0], rows[:, 1]) or 0.0,
            n_boot=1000,
            seed=int(config.get("seed", 7)),
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"{checkpoint.get('model_name', 'model')}_eval_metrics.json", metrics)
    prediction_rows = []
    for local_idx, source_idx in enumerate(dataset_indices.tolist()):
        source_row = metadata_rows[source_idx] if source_idx < len(metadata_rows) else {}
        prediction_rows.append(
            {
                **source_row,
                "model": checkpoint.get("model_name", "model"),
                "prediction_abnormal_return": float(predictions[local_idx, 0]),
                "prediction_volatility_jump": float(predictions[local_idx, 1]),
                "prediction_volume_jump": float(predictions[local_idx, 2]),
                "target_abnormal_return": float(targets[local_idx, 0]),
                "target_volatility_jump": float(targets[local_idx, 1]),
                "target_volume_jump": float(targets[local_idx, 2]),
                "event_delta_abs_mean": float(np.mean(np.abs(deltas[local_idx]))),
            }
        )
    prediction_path = output / f"{checkpoint.get('model_name', 'model')}_predictions.jsonl"
    write_jsonl(prediction_path, prediction_rows)
    return metrics


def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    deltas = []
    is_event = []
    indices = []
    with torch.no_grad():
        for batch in loader:
            x_pre = batch["x_pre"].to(device)
            event_embedding = batch["event_embedding"].to(device)
            metadata = batch["metadata"].to(device)
            outputs = model(x_pre, event_embedding, metadata)
            predictions.append(outputs["prediction"].detach().cpu().numpy())
            deltas.append(outputs["event_delta"].detach().cpu().numpy())
            targets.append(batch["y"].numpy())
            is_event.append(batch["is_event"].numpy())
            indices.append(batch["index"].numpy())
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(deltas, axis=0),
        np.concatenate(is_event, axis=0),
        np.concatenate(indices, axis=0),
    )


def _masked_abs_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(np.mean(np.abs(values[mask])))
