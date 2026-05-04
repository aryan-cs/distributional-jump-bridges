"""Training loop."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from cebt.models.cebt import CEBTConfig, build_model
from cebt.training.dataset import CEBTTensorDataset
from cebt.training.losses import LossWeights, cebt_loss
from cebt.utils.config import ensure_dir
from cebt.utils.io import write_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    config: dict[str, Any],
    feature_path: str | Path,
    output_dir: str | Path,
    model_name: str = "cebt",
) -> dict[str, Any]:
    set_seed(int(config.get("seed", 7)))
    run_dir = ensure_dir(output_dir)
    model_config = CEBTConfig.from_dict(config.get("model", {}))
    loss_weights = LossWeights.from_dict(config.get("loss", {}))
    training = config.get("training", {})
    device = auto_device()
    train_ds = CEBTTensorDataset(feature_path, split=0)
    val_ds = CEBTTensorDataset(feature_path, split=1)
    if len(train_ds) == 0:
        raise ValueError("No training rows in feature bundle.")
    target_mean, target_std = _target_stats(train_ds, device, loss_weights)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(training.get("batch_size", 32)),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=int(training.get("batch_size", 32)), shuffle=False)
    model = build_model(model_name, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    for epoch in range(int(training.get("epochs", 3))):
        model.train()
        train_metrics = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(batch["x_pre"], batch["event_embedding"], batch["metadata"])
                loss, metrics = cebt_loss(
                    outputs,
                    batch["y"],
                    batch["is_event"],
                    loss_weights,
                    target_mean=target_mean,
                    target_std=target_std,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_metrics.append(metrics)
        val_metrics = (
            evaluate_loss(
                model,
                val_loader,
                loss_weights,
                device,
                target_mean=target_mean,
                target_std=target_std,
            )
            if len(val_ds)
            else {}
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train": _mean_metrics(train_metrics),
                "val": val_metrics,
            }
        )
    checkpoint = {
        "model_name": model_name,
        "model_config": model_config.__dict__,
        "loss_config": loss_weights.__dict__,
        "target_mean": target_mean.detach().cpu().tolist() if target_mean is not None else None,
        "target_std": target_std.detach().cpu().tolist() if target_std is not None else None,
        "state_dict": model.state_dict(),
        "history": history,
    }
    checkpoint_path = run_dir / f"{model_name}.pt"
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "epochs": len(history),
        "device": str(device),
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "history": history,
    }
    write_json(run_dir / f"{model_name}_train_summary.json", summary)
    return summary


def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_weights: LossWeights,
    device: torch.device,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    metrics = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = model(batch["x_pre"], batch["event_embedding"], batch["metadata"])
            _, row = cebt_loss(
                outputs,
                batch["y"],
                batch["is_event"],
                loss_weights,
                target_mean=target_mean,
                target_std=target_std,
            )
            metrics.append(row)
    return _mean_metrics(metrics)


def _target_stats(
    train_ds: CEBTTensorDataset,
    device: torch.device,
    loss_weights: LossWeights,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not loss_weights.standardize_targets:
        return None, None
    y = train_ds.y.to(device)
    target_mean = torch.mean(y, dim=0)
    target_std = torch.std(y, dim=0, unbiased=False).clamp_min(1e-6)
    return target_mean, target_std


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
