"""Generate compact result tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cebt.evaluation.bootstrap import paired_bootstrap_ci
from cebt.utils.io import read_json, read_jsonl, write_csv, write_json


def make_tables(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    metric_rows = []
    for path in sorted(root.glob("*_eval_metrics.json")):
        metrics = read_json(path)
        model = path.name.replace("_eval_metrics.json", "")
        for key, value in metrics.items():
            if isinstance(value, dict):
                metric_rows.append({"model": model, "metric": key, **value})
            else:
                metric_rows.append({"model": model, "metric": key, "value": value})
    write_csv(root / "table_eval_metrics.csv", metric_rows)
    paired_rows = make_paired_comparison_table(root)
    residual_rows = make_residual_table(root)
    summary = {
        "metric_rows": len(metric_rows),
        "paired_rows": len(paired_rows),
        "residual_rows": len(residual_rows),
        "source_dir": str(root),
    }
    write_json(root / "tables_summary.json", summary)
    return summary


def make_paired_comparison_table(root: Path, reference_model: str = "cebt") -> list[dict]:
    predictions = _load_predictions(root)
    if reference_model not in predictions:
        return []
    reference = predictions[reference_model]
    rows = []
    for model, model_rows in sorted(predictions.items()):
        if model == reference_model:
            continue
        joined = _join_prediction_rows(reference, model_rows)
        if not joined:
            continue
        ref_errors = np.asarray([item["reference_mse"] for item in joined], dtype=float)
        model_errors = np.asarray([item["model_mse"] for item in joined], dtype=float)
        improvement = paired_bootstrap_ci(
            model_errors,
            ref_errors,
            lambda values: float(np.mean(values)),
            n_boot=2000,
            seed=17,
        )
        rows.append(
            {
                "reference_model": reference_model,
                "baseline_model": model,
                "paired_rows": len(joined),
                "baseline_minus_reference_mse": improvement["mean"],
                "ci_lo": improvement["lo"],
                "ci_hi": improvement["hi"],
                "reference_better": improvement["lo"] > 0.0,
            }
        )
    write_csv(root / "table_paired_mse_comparisons.csv", rows)
    return rows


def make_residual_table(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*_eval_metrics.json")):
        metrics = read_json(path)
        model = path.name.replace("_eval_metrics.json", "")
        true_delta = metrics.get("mean_abs_event_delta_true_events")
        control_delta = metrics.get("mean_abs_event_delta_controls")
        ratio = None
        if true_delta is not None and control_delta not in (None, 0.0):
            ratio = true_delta / control_delta
        rows.append(
            {
                "model": model,
                "mean_abs_event_delta_true_events": true_delta,
                "mean_abs_event_delta_controls": control_delta,
                "event_to_control_delta_ratio": ratio,
            }
        )
    write_csv(root / "table_residual_diagnostics.csv", rows)
    return rows


def _load_predictions(root: Path) -> dict[str, list[dict]]:
    predictions = {}
    for path in sorted(root.glob("*_predictions.jsonl")):
        model = path.name.replace("_predictions.jsonl", "")
        predictions[model] = read_jsonl(path)
    return predictions


def _join_prediction_rows(reference: list[dict], candidate: list[dict]) -> list[dict]:
    reference_by_sample = {row["sample_id"]: row for row in reference if row.get("sample_id")}
    joined = []
    for row in candidate:
        sample_id = row.get("sample_id")
        if sample_id not in reference_by_sample:
            continue
        ref_row = reference_by_sample[sample_id]
        joined.append(
            {
                "sample_id": sample_id,
                "reference_mse": _row_mse(ref_row),
                "model_mse": _row_mse(row),
            }
        )
    return joined


def _row_mse(row: dict) -> float:
    pred = np.asarray(
        [
            row["prediction_abnormal_return"],
            row["prediction_volatility_jump"],
            row["prediction_volume_jump"],
        ],
        dtype=float,
    )
    target = np.asarray(
        [
            row["target_abnormal_return"],
            row["target_volatility_jump"],
            row["target_volume_jump"],
        ],
        dtype=float,
    )
    return float(np.mean((pred - target) ** 2))
