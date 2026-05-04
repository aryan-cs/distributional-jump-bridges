"""Run DJB/RC-DJB replication and sensitivity sweeps."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from cebt.cli import load_run_config, processed_dir, resolve_path
from cebt.evaluation.evaluate import evaluate_model
from cebt.training.train import train_model
from cebt.utils.config import deep_merge, ensure_dir, load_yaml
from cebt.utils.io import write_json


def main() -> None:
    args = _parse_args()
    experiment = load_yaml(resolve_path(args.config))
    base_config = load_run_config(experiment.get("base_config", "configs/paper_v3_bge_rc_djb.yaml"))
    output_root = ensure_dir(resolve_path(args.output_dir or experiment["output_root"]))
    feature_dir = resolve_path(experiment.get("feature_dir", processed_dir(base_config)))
    feature_path = feature_dir / "features.npz"
    metadata_path = feature_dir / "features_metadata.jsonl"
    models = args.models or list(experiment.get("models", ["djb", "rc_djb"]))
    seeds = args.seeds or [int(seed) for seed in experiment.get("seeds", [29, 42, 7, 88, 13])]
    interventions = args.interventions or list(experiment.get("interventions", ["full"]))
    selected_sweeps = set(args.sweeps or [])
    sweep_rows = [
        row
        for row in experiment.get("sweeps", [{"id": "baseline", "overrides": {}}])
        if not selected_sweeps or str(row["id"]) in selected_sweeps
    ]
    if not sweep_rows:
        raise SystemExit("No sweeps selected.")

    manifest: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for sweep in sweep_rows:
        sweep_id = str(sweep["id"])
        overrides = sweep.get("overrides", {})
        for model_name in models:
            for seed in seeds:
                run_config = deep_merge(deepcopy(base_config), overrides)
                run_config["seed"] = int(seed)
                run_dir = ensure_dir(output_root / sweep_id / model_name / f"seed_{seed}")
                write_json(run_dir / "resolved_config.json", run_config)
                checkpoint_path = run_dir / f"{model_name}.pt"
                if not args.skip_train:
                    train_model(run_config, feature_path, run_dir, model_name=model_name)
                if not args.skip_eval:
                    if not checkpoint_path.exists():
                        raise SystemExit(f"Missing checkpoint for evaluation: {checkpoint_path}")
                    for intervention in interventions:
                        metrics = evaluate_model(
                            run_config,
                            feature_path,
                            metadata_path,
                            checkpoint_path,
                            run_dir,
                            intervention=intervention,
                        )
                        metric_rows.append(
                            _metric_row(
                                sweep_id=sweep_id,
                                model_name=model_name,
                                seed=seed,
                                intervention=intervention,
                                run_dir=run_dir,
                                metrics=metrics,
                            )
                        )
                manifest.append(
                    {
                        "sweep_id": sweep_id,
                        "model_name": model_name,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "checkpoint_path": str(checkpoint_path),
                        "overrides": overrides,
                    }
                )
    write_json(output_root / "manifest.json", {"runs": manifest})
    _write_metrics_csv(output_root / "metrics.csv", metric_rows)
    summary_rows = _summary_rows(metric_rows)
    _write_metrics_csv(output_root / "summary.csv", summary_rows)
    paired_rows = _paired_model_rows(metric_rows)
    _write_metrics_csv(output_root / "paired_model_differences.csv", paired_rows)
    print(f"Wrote {len(manifest)} run manifest rows to {output_root / 'manifest.json'}")
    if metric_rows:
        print(f"Wrote {len(metric_rows)} metric rows to {output_root / 'metrics.csv'}")
        print(f"Wrote {len(summary_rows)} summary rows to {output_root / 'summary.csv'}")
    if paired_rows:
        print(
            "Wrote "
            f"{len(paired_rows)} paired model rows to "
            f"{output_root / 'paired_model_differences.csv'}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/djb_replication_sweeps.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--sweeps", nargs="*", default=None)
    parser.add_argument("--interventions", nargs="*", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def _metric_row(
    sweep_id: str,
    model_name: str,
    seed: int,
    intervention: str,
    run_dir: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sweep_id": sweep_id,
        "model_name": model_name,
        "seed": seed,
        "intervention": intervention,
        "run_dir": str(run_dir),
        "rows": metrics.get("rows"),
        "mse": metrics.get("mse"),
        "rank_ic": metrics.get("abnormal_return_rank_ic"),
        "rank_ic_ticker_lo": _nested(metrics, "rank_ic_ci_ticker_cluster", "lo"),
        "rank_ic_ticker_hi": _nested(metrics, "rank_ic_ci_ticker_cluster", "hi"),
        "mse_ticker_lo": _nested(metrics, "mse_ci_ticker_cluster", "lo"),
        "mse_ticker_hi": _nested(metrics, "mse_ci_ticker_cluster", "hi"),
        "balanced_accuracy": metrics.get("abnormal_return_balanced_accuracy"),
        "spread": metrics.get("abnormal_return_spread"),
        "latent_jump_auc": metrics.get("latent_jump_auc"),
        "response_transport_auc": metrics.get("response_transport_auc"),
        "volatility_transport_auc": metrics.get("volatility_transport_auc"),
        "volume_transport_auc": metrics.get("volume_transport_auc"),
    }


def _nested(metrics: dict[str, Any], key: str, subkey: str) -> Any:
    value = metrics.get(key)
    return value.get(subkey) if isinstance(value, dict) else None


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["sweep_id"]), str(row["model_name"]), str(row["intervention"]))
        grouped.setdefault(key, []).append(row)
    summary = []
    for (sweep_id, model_name, intervention), group_rows in sorted(grouped.items()):
        base = {
            "sweep_id": sweep_id,
            "model_name": model_name,
            "intervention": intervention,
            "runs": len(group_rows),
            "seeds": ",".join(str(row["seed"]) for row in group_rows),
        }
        for metric in (
            "mse",
            "rank_ic",
            "balanced_accuracy",
            "spread",
            "latent_jump_auc",
            "response_transport_auc",
            "volatility_transport_auc",
            "volume_transport_auc",
        ):
            base.update(_metric_distribution(group_rows, metric))
        summary.append(base)
    return summary


def _paired_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["sweep_id"]), str(row["intervention"]), int(row["seed"]))
        by_cell.setdefault(key, {})[str(row["model_name"])] = row
    pair_rows = []
    for (sweep_id, intervention, seed), models in sorted(by_cell.items()):
        if "djb" not in models or "rc_djb" not in models:
            continue
        djb = models["djb"]
        rc_djb = models["rc_djb"]
        pair_rows.append(
            {
                "sweep_id": sweep_id,
                "intervention": intervention,
                "seed": seed,
                "rc_djb_minus_djb_rank_ic": _diff(rc_djb, djb, "rank_ic"),
                "rc_djb_minus_djb_mse": _diff(rc_djb, djb, "mse"),
                "rc_djb_minus_djb_response_transport_auc": _diff(
                    rc_djb,
                    djb,
                    "response_transport_auc",
                ),
            }
        )
    return pair_rows


def _metric_distribution(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = np.asarray(
        [float(row[metric]) for row in rows if row.get(metric) is not None],
        dtype=float,
    )
    if values.size == 0:
        return {
            f"{metric}_median": None,
            f"{metric}_q25": None,
            f"{metric}_q75": None,
        }
    return {
        f"{metric}_median": float(np.median(values)),
        f"{metric}_q25": float(np.quantile(values, 0.25)),
        f"{metric}_q75": float(np.quantile(values, 0.75)),
    }


def _diff(left: dict[str, Any], right: dict[str, Any], metric: str) -> float | None:
    if left.get(metric) is None or right.get(metric) is None:
        return None
    return float(left[metric]) - float(right[metric])


if __name__ == "__main__":
    main()
