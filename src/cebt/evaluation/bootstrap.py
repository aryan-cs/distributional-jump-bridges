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


def clustered_bootstrap_ci(
    values: np.ndarray,
    group_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 7,
    alpha: float = 0.05,
) -> dict[str, float]:
    if values.shape[0] == 0 or values.shape[0] != group_ids.shape[0]:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    groups = np.asarray(sorted(set(group_ids.tolist())), dtype=object)
    if groups.shape[0] == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    group_to_indices = {group: np.flatnonzero(group_ids == group) for group in groups}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(groups, size=groups.shape[0], replace=True)
        indices = np.concatenate([group_to_indices[group] for group in sampled_groups])
        estimates.append(metric_fn(values[indices]))
    ordered = np.asarray(estimates, dtype=float)
    return {
        "mean": float(metric_fn(values)),
        "lo": float(np.quantile(ordered, alpha / 2.0)),
        "hi": float(np.quantile(ordered, 1.0 - alpha / 2.0)),
        "groups": int(groups.shape[0]),
    }


def leave_one_group_out(
    values: np.ndarray,
    group_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
) -> dict[str, float | str]:
    if values.shape[0] == 0 or values.shape[0] != group_ids.shape[0]:
        return {
            "full": float("nan"),
            "mean": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "groups": 0,
            "min_group": "",
            "max_group": "",
        }
    groups = np.asarray(sorted(set(group_ids.tolist())), dtype=object)
    if groups.shape[0] <= 1:
        full = float(metric_fn(values))
        return {
            "full": full,
            "mean": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "groups": int(groups.shape[0]),
            "min_group": "",
            "max_group": "",
        }
    estimates = []
    kept_groups = []
    for group in groups:
        mask = group_ids != group
        if not np.any(mask):
            continue
        estimates.append(float(metric_fn(values[mask])))
        kept_groups.append(str(group))
    if not estimates:
        full = float(metric_fn(values))
        return {
            "full": full,
            "mean": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "groups": int(groups.shape[0]),
            "min_group": "",
            "max_group": "",
        }
    ordered = np.asarray(estimates, dtype=float)
    min_idx = int(np.argmin(ordered))
    max_idx = int(np.argmax(ordered))
    return {
        "full": float(metric_fn(values)),
        "mean": float(np.mean(ordered)),
        "lo": float(np.quantile(ordered, 0.025)),
        "hi": float(np.quantile(ordered, 0.975)),
        "min": float(ordered[min_idx]),
        "max": float(ordered[max_idx]),
        "std": float(np.std(ordered, ddof=0)),
        "groups": int(groups.shape[0]),
        "min_group": kept_groups[min_idx],
        "max_group": kept_groups[max_idx],
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
