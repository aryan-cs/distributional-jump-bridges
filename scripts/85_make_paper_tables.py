"""Assemble paper tables from the corrected DJB/RC-DJB experiment artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cebt.evaluation.bootstrap import paired_bootstrap_ci
from cebt.evaluation.metrics import rank_ic
from cebt.utils.io import read_json, read_jsonl, write_csv, write_json

PAPER_TABLE_DIR = Path("paper/tables")
MODEL_SOURCES = {
    "no_event": Path("data/runs/paper_v3"),
    "concat": Path("data/runs/paper_v3"),
    "cebt": Path("data/runs/paper_v3"),
    "dot": Path("data/runs/paper_v3_dot_mse"),
    "ejssm_bge": Path("data/runs/paper_v3_bge_ejssm_balanced"),
    "djb": Path("data/runs/paper_v3_bge_djb_best"),
    "rc_djb": Path("data/runs/paper_v3_bge_rc_djb_best"),
}
MODEL_FILE_STEMS = {
    "no_event": "no_event",
    "concat": "concat",
    "cebt": "cebt",
    "dot": "dot",
    "ejssm_bge": "ejssm",
    "djb": "djb",
    "rc_djb": "rc_djb",
}
RCDJB_INTERVENTIONS = {
    "full": "rc_djb",
    "no_bridge": "rc_djb_no_jump",
    "zero_text": "rc_djb_zero_event",
    "shuffled_text": "rc_djb_shuffle_event",
}
TEMPERATURE_SOURCES = {
    ("DJB", "T=0.02"): Path("data/runs/paper_v3_bge_djb_best/djb_eval_metrics.json"),
    ("DJB", "T=0.10"): Path("data/runs/paper_v3_bge_djb_t010/djb_eval_metrics.json"),
    ("RC-DJB", "T=0.02"): Path("data/runs/paper_v3_bge_rc_djb_best/rc_djb_eval_metrics.json"),
    ("RC-DJB", "T=0.10"): Path("data/runs/paper_v3_bge_rc_djb_t010/rc_djb_eval_metrics.json"),
}
REFERENCE_MODEL = "djb"


def main() -> None:
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = _load_model_metrics()
    predictions = _load_model_predictions()
    write_csv(PAPER_TABLE_DIR / "table_eval_metrics.csv", _flatten_metrics(metrics))
    write_csv(PAPER_TABLE_DIR / "table_paired_mse_comparisons.csv", _paired_mse_rows(predictions))
    write_csv(
        PAPER_TABLE_DIR / "table_paired_rank_ic_comparisons.csv",
        _paired_rank_rows(predictions),
    )
    write_csv(PAPER_TABLE_DIR / "table_residual_diagnostics.csv", _residual_rows(metrics))
    write_csv(
        PAPER_TABLE_DIR / "table_rc_djb_interventions.csv",
        _rc_djb_intervention_rows(),
    )
    write_csv(
        PAPER_TABLE_DIR / "table_rc_djb_intervention_paired.csv",
        _rc_djb_intervention_paired_rows(),
    )
    write_csv(
        PAPER_TABLE_DIR / "table_rank_temperature_sensitivity.csv",
        _temperature_sensitivity_rows(),
    )
    write_json(
        PAPER_TABLE_DIR / "tables_summary.json",
        {
            "models": sorted(metrics),
            "paired_reference_model": REFERENCE_MODEL,
            "paired_models": sorted(model for model in predictions if model != REFERENCE_MODEL),
        },
    )


def _load_model_metrics() -> dict[str, dict[str, Any]]:
    rows = {}
    for model, root in MODEL_SOURCES.items():
        stem = MODEL_FILE_STEMS[model]
        path = root / f"{stem}_eval_metrics.json"
        if path.exists():
            rows[model] = read_json(path)
    return rows


def _load_model_predictions() -> dict[str, list[dict[str, Any]]]:
    rows = {}
    for model, root in MODEL_SOURCES.items():
        stem = MODEL_FILE_STEMS[model]
        path = root / f"{stem}_predictions.jsonl"
        if path.exists():
            rows[model] = read_jsonl(path)
    return rows


def _flatten_metrics(metrics_by_model: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model, metrics in metrics_by_model.items():
        for metric, value in metrics.items():
            if isinstance(value, dict):
                rows.append({"model": model, "metric": metric, **value})
            else:
                rows.append({"model": model, "metric": metric, "value": value})
    return rows


def _paired_mse_rows(predictions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    reference = predictions.get(REFERENCE_MODEL)
    if not reference:
        return []
    rows = []
    for model, candidate in predictions.items():
        if model == REFERENCE_MODEL:
            continue
        joined = _join_prediction_rows(reference, candidate)
        if not joined:
            continue
        reference_errors = np.asarray([item["reference_mse"] for item in joined], dtype=float)
        model_errors = np.asarray([item["model_mse"] for item in joined], dtype=float)
        interval = paired_bootstrap_ci(
            model_errors,
            reference_errors,
            lambda values: float(np.mean(values)),
            n_boot=2000,
            seed=17,
        )
        rows.append(
            {
                "reference_model": REFERENCE_MODEL,
                "baseline_model": model,
                "paired_rows": len(joined),
                "baseline_minus_reference_mse": interval["mean"],
                "ci_lo": interval["lo"],
                "ci_hi": interval["hi"],
                "reference_better": interval["lo"] > 0.0,
            }
        )
    return rows


def _paired_rank_rows(predictions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    reference = predictions.get(REFERENCE_MODEL)
    if not reference:
        return []
    rows = []
    for model, candidate in predictions.items():
        if model == REFERENCE_MODEL:
            continue
        joined = _join_prediction_rows(reference, candidate)
        if not joined:
            continue
        reference_pairs = np.asarray(
            [[item["reference_prediction"], item["target_abnormal_return"]] for item in joined],
            dtype=float,
        )
        model_pairs = np.asarray(
            [[item["model_prediction"], item["target_abnormal_return"]] for item in joined],
            dtype=float,
        )
        interval = paired_bootstrap_ci(
            reference_pairs,
            model_pairs,
            lambda values: rank_ic(values[:, 0], values[:, 1]) or 0.0,
            n_boot=2000,
            seed=29,
        )
        rows.append(
            {
                "reference_model": REFERENCE_MODEL,
                "baseline_model": model,
                "paired_rows": len(joined),
                "reference_minus_baseline_rank_ic": interval["mean"],
                "ci_lo": interval["lo"],
                "ci_hi": interval["hi"],
                "reference_better": interval["lo"] > 0.0,
            }
        )
    return rows


def _residual_rows(metrics_by_model: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model, metrics in metrics_by_model.items():
        true_delta = metrics.get("mean_abs_event_delta_true_events")
        control_delta = metrics.get("mean_abs_event_delta_controls")
        ratio = None
        if true_delta is not None and control_delta not in (None, 0.0):
            ratio = float(true_delta) / float(control_delta)
        rows.append(
            {
                "model": model,
                "mean_abs_event_delta_true_events": true_delta,
                "mean_abs_event_delta_controls": control_delta,
                "event_to_control_delta_ratio": ratio,
                "mean_abs_return_delta_true_events": metrics.get(
                    "mean_abs_return_delta_true_events"
                ),
                "mean_abs_return_delta_controls": metrics.get(
                    "mean_abs_return_delta_controls"
                ),
                "mean_abs_volatility_delta_true_events": metrics.get(
                    "mean_abs_volatility_delta_true_events"
                ),
                "mean_abs_volatility_delta_controls": metrics.get(
                    "mean_abs_volatility_delta_controls"
                ),
                "mean_abs_volume_delta_true_events": metrics.get(
                    "mean_abs_volume_delta_true_events"
                ),
                "mean_abs_volume_delta_controls": metrics.get(
                    "mean_abs_volume_delta_controls"
                ),
                "response_transport_auc": metrics.get("response_transport_auc"),
                "volatility_transport_auc": metrics.get("volatility_transport_auc"),
                "volume_transport_auc": metrics.get("volume_transport_auc"),
            }
        )
    return rows


def _rc_djb_intervention_rows() -> list[dict[str, Any]]:
    root = Path("data/runs/paper_v3_bge_rc_djb_best")
    rows = []
    for label, stem in RCDJB_INTERVENTIONS.items():
        path = root / f"{stem}_eval_metrics.json"
        if not path.exists():
            continue
        metrics = read_json(path)
        rows.append(
            {
                "intervention": label,
                "mse": metrics.get("mse"),
                "event_mse": metrics.get("event_mse"),
                "event_gaussian_nll": metrics.get("event_gaussian_nll"),
                "abnormal_return_rank_ic": metrics.get("abnormal_return_rank_ic"),
                "latent_jump_auc": metrics.get("latent_jump_auc"),
                "response_transport_auc": metrics.get("response_transport_auc"),
                "volatility_transport_auc": metrics.get("volatility_transport_auc"),
                "volume_transport_auc": metrics.get("volume_transport_auc"),
                "mean_abs_event_delta_true_events": metrics.get("mean_abs_event_delta_true_events"),
                "mean_abs_event_delta_controls": metrics.get("mean_abs_event_delta_controls"),
                "mean_abs_return_delta_true_events": metrics.get(
                    "mean_abs_return_delta_true_events"
                ),
                "mean_abs_return_delta_controls": metrics.get(
                    "mean_abs_return_delta_controls"
                ),
                "mean_abs_volatility_delta_true_events": metrics.get(
                    "mean_abs_volatility_delta_true_events"
                ),
                "mean_abs_volatility_delta_controls": metrics.get(
                    "mean_abs_volatility_delta_controls"
                ),
                "mean_abs_volume_delta_true_events": metrics.get(
                    "mean_abs_volume_delta_true_events"
                ),
                "mean_abs_volume_delta_controls": metrics.get(
                    "mean_abs_volume_delta_controls"
                ),
                "return_logvar_delta_signed_rank_ic": metrics.get(
                    "return_logvar_delta_signed_rank_ic"
                ),
            }
        )
    return rows


def _rc_djb_intervention_paired_rows() -> list[dict[str, Any]]:
    root = Path("data/runs/paper_v3_bge_rc_djb_best")
    full = read_jsonl(root / "rc_djb_predictions.jsonl")
    rows = []
    for label, stem in RCDJB_INTERVENTIONS.items():
        if label == "full":
            continue
        candidate = read_jsonl(root / f"{stem}_predictions.jsonl")
        joined = [
            item
            for item in _join_prediction_rows(full, candidate)
            if item["control_type"] == "real_event"
        ]
        if not joined:
            continue
        full_errors = np.asarray([item["reference_mse"] for item in joined], dtype=float)
        intervention_errors = np.asarray([item["model_mse"] for item in joined], dtype=float)
        interval = paired_bootstrap_ci(
            intervention_errors,
            full_errors,
            lambda values: float(np.mean(values)),
            n_boot=2000,
            seed=41,
        )
        rows.append(
            {
                "reference_model": "full_rc_djb",
                "intervention": label,
                "event_rows": len(joined),
                "intervention_minus_full_mse": interval["mean"],
                "ci_lo": interval["lo"],
                "ci_hi": interval["hi"],
                "full_better": interval["lo"] > 0.0,
            }
        )
    return rows


def _temperature_sensitivity_rows() -> list[dict[str, Any]]:
    rows = []
    for (model, label), path in TEMPERATURE_SOURCES.items():
        if not path.exists():
            continue
        metrics = read_json(path)
        rank_ci = metrics.get("rank_ic_ci_ticker_cluster", {})
        loto = metrics.get("rank_ic_leave_one_ticker_out", {})
        rows.append(
            {
                "model": model,
                "temperature": label,
                "mse": metrics.get("mse"),
                "rank_ic": metrics.get("abnormal_return_rank_ic"),
                "rank_ic_ci_lo": rank_ci.get("lo"),
                "rank_ic_ci_hi": rank_ci.get("hi"),
                "leave_one_ticker_min": loto.get("min"),
                "leave_one_ticker_min_group": loto.get("min_group"),
                "leave_one_ticker_max": loto.get("max"),
                "leave_one_ticker_max_group": loto.get("max_group"),
            }
        )
    return rows


def _join_prediction_rows(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
                "control_type": ref_row.get("control_type", ""),
                "reference_mse": _row_mse(ref_row),
                "model_mse": _row_mse(row),
                "reference_prediction": ref_row["prediction_abnormal_return"],
                "model_prediction": row["prediction_abnormal_return"],
                "target_abnormal_return": ref_row["target_abnormal_return"],
            }
        )
    return joined


def _row_mse(row: dict[str, Any]) -> float:
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


if __name__ == "__main__":
    main()
