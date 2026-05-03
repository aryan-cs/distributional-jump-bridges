"""Bootstrap confidence intervals for forecast metrics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 7,
    alpha: float = 0.05,
) -> dict[str, float]:
    if values.shape[0] == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        indices = rng.integers(0, values.shape[0], size=values.shape[0])
        estimates.append(metric_fn(values[indices]))
    ordered = np.asarray(estimates, dtype=float)
    return {
        "mean": float(metric_fn(values)),
        "lo": float(np.quantile(ordered, alpha / 2.0)),
        "hi": float(np.quantile(ordered, 1.0 - alpha / 2.0)),
    }


def paired_bootstrap_ci(
    left: np.ndarray,
    right: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 7,
    alpha: float = 0.05,
) -> dict[str, float]:
    if left.shape[0] == 0 or left.shape[0] != right.shape[0]:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        indices = rng.integers(0, left.shape[0], size=left.shape[0])
        estimates.append(metric_fn(left[indices]) - metric_fn(right[indices]))
    observed = metric_fn(left) - metric_fn(right)
    ordered = np.asarray(estimates, dtype=float)
    return {
        "mean": float(observed),
        "lo": float(np.quantile(ordered, alpha / 2.0)),
        "hi": float(np.quantile(ordered, 1.0 - alpha / 2.0)),
    }
