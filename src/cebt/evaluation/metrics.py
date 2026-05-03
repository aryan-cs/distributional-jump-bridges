"""Metrics for event-driven forecasts."""

from __future__ import annotations

import math

import numpy as np


def balanced_accuracy_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    preds = scores >= 0
    positives = labels >= 0
    tp = np.sum(preds & positives)
    tn = np.sum(~preds & ~positives)
    fp = np.sum(preds & ~positives)
    fn = np.sum(~preds & positives)
    tpr = 0.0 if tp + fn == 0 else tp / (tp + fn)
    tnr = 0.0 if tn + fp == 0 else tn / (tn + fp)
    return float((tpr + tnr) / 2.0)


def rank_ic(scores: np.ndarray, targets: np.ndarray) -> float | None:
    if scores.shape[0] < 2:
        return None
    score_ranks = _ranks(scores)
    target_ranks = _ranks(targets)
    return _pearson(score_ranks, target_ranks)


def abnormal_return_spread(
    scores: np.ndarray, targets: np.ndarray, quantile: float = 0.2
) -> float | None:
    if scores.shape[0] < 5:
        return None
    low_cut = np.quantile(scores, quantile)
    high_cut = np.quantile(scores, 1.0 - quantile)
    high = targets[scores >= high_cut]
    low = targets[scores <= low_cut]
    if high.size == 0 or low.size == 0:
        return None
    return float(np.mean(high) - np.mean(low))


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def calibration_error(scores: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    if scores.size == 0:
        return 0.0
    probabilities = 1.0 / (1.0 + np.exp(-scores))
    labels = (targets >= 0).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (probabilities >= lo) & (probabilities < hi if hi < 1.0 else probabilities <= hi)
        if not np.any(mask):
            continue
        total += float(np.mean(mask)) * abs(
            float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
        )
    return total


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = left - np.mean(left)
    right = right - np.mean(right)
    denom = math.sqrt(float(np.sum(left**2) * np.sum(right**2)))
    if denom == 0:
        return None
    return float(np.sum(left * right) / denom)
