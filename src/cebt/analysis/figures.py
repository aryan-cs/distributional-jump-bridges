"""Figure generation for DJB paper artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cebt.evaluation.metrics import rank_ic
from cebt.utils.io import read_json, read_jsonl, write_json


def make_figures(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics(root)
    predictions = _load_predictions(root)
    outputs = {}
    outputs["architecture"] = str(_plot_architecture(figure_dir))
    if metrics:
        outputs["metric_comparison"] = str(_plot_metric_comparison(metrics, figure_dir))
        outputs["rank_ic_comparison"] = str(_plot_rank_ic(metrics, figure_dir))
        outputs["event_delta_controls"] = str(_plot_event_delta(metrics, figure_dir))
    if predictions:
        outputs["prediction_scatter"] = str(_plot_prediction_scatter(predictions, figure_dir))
        outputs["prediction_phase_portrait"] = str(
            _plot_prediction_phase_portrait(predictions, figure_dir)
        )
        outputs["temporal_rank_ic_heatmap"] = str(
            _plot_temporal_rank_ic_heatmap(predictions, figure_dir)
        )
        outputs["event_residual_mosaic"] = str(_plot_event_residual_mosaic(predictions, figure_dir))
        outputs["residual_distribution"] = str(
            _plot_residual_distribution(predictions, figure_dir)
        )
    write_json(root / "figures_summary.json", outputs)
    return outputs


def _load_metrics(root: Path) -> dict[str, dict]:
    metrics = {}
    for path in sorted(root.glob("*_eval_metrics.json")):
        metrics[path.name.replace("_eval_metrics.json", "")] = read_json(path)
    return metrics


def _model_order(metrics: dict[str, dict]) -> list[str]:
    preferred = ["no_event", "text_only", "concat", "cebt_no_controls", "cebt"]
    ordered = [name for name in preferred if name in metrics]
    ordered.extend(name for name in sorted(metrics) if name not in ordered)
    return ordered


def _display_name(model: str) -> str:
    return {
        "no_event": "No-event",
        "text_only": "Text-only",
        "concat": "Concat fusion",
        "cebt_no_controls": "Bottleneck\n(no controls)",
        "cebt": "Bottleneck",
    }.get(model, model)


def _load_predictions(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*_predictions.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def _plot_metric_comparison(metrics: dict[str, dict], figure_dir: Path) -> Path:
    names = _model_order(metrics)
    values = [metrics[name]["mse"] for name in names]
    lows = [
        metrics[name].get("mse_ci", {}).get("lo", values[idx])
        for idx, name in enumerate(names)
    ]
    highs = [
        metrics[name].get("mse_ci", {}).get("hi", values[idx])
        for idx, name in enumerate(names)
    ]
    yerr = [
        [max(value - lo, 0.0) for value, lo in zip(values, lows, strict=False)],
        [max(hi - value, 0.0) for value, hi in zip(values, highs, strict=False)],
    ]
    path = figure_dir / "metric_comparison.png"
    plt.figure(figsize=(8, 4.5))
    plt.bar(names, values, yerr=yerr, capsize=4, color="#4f82d6")
    plt.ylabel("MSE, lower is better")
    plt.title("Forecast Error With Bootstrap 95% CI")
    plt.xticks(range(len(names)), [_display_name(name) for name in names], rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _plot_rank_ic(metrics: dict[str, dict], figure_dir: Path) -> Path:
    names = _model_order(metrics)
    values = [metrics[name]["abnormal_return_rank_ic"] for name in names]
    lows = [
        metrics[name].get("rank_ic_ci", {}).get("lo", values[idx])
        for idx, name in enumerate(names)
    ]
    highs = [
        metrics[name].get("rank_ic_ci", {}).get("hi", values[idx])
        for idx, name in enumerate(names)
    ]
    yerr = [
        [max(value - lo, 0.0) for value, lo in zip(values, lows, strict=False)],
        [max(hi - value, 0.0) for value, hi in zip(values, highs, strict=False)],
    ]
    colors = ["#7a8898" if name != "cebt" else "#4f82d6" for name in names]
    path = figure_dir / "rank_ic_comparison.png"
    plt.figure(figsize=(8, 4.5))
    plt.bar(names, values, yerr=yerr, capsize=4, color=colors)
    plt.axhline(0.0, color="black", linewidth=0.9)
    plt.ylabel("Abnormal-return rank IC")
    plt.title("Event Ranking Signal With Bootstrap 95% CI")
    plt.xticks(range(len(names)), [_display_name(name) for name in names], rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _plot_event_delta(metrics: dict[str, dict], figure_dir: Path) -> Path:
    names = _model_order(metrics)
    true_values = [
        metrics[name].get("mean_abs_event_delta_true_events") or 0.0 for name in names
    ]
    control_values = [
        metrics[name].get("mean_abs_event_delta_controls") or 0.0 for name in names
    ]
    x = range(len(names))
    path = figure_dir / "event_delta_controls.png"
    plt.figure(figsize=(8, 4.5))
    plt.bar([value - 0.18 for value in x], true_values, width=0.36, label="True events")
    plt.bar([value + 0.18 for value in x], control_values, width=0.36, label="Controls")
    plt.ylabel("Mean absolute event delta")
    plt.title("Event Residual Magnitude Separates Events From Controls")
    plt.xticks(list(x), [_display_name(name) for name in names], rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _plot_architecture(figure_dir: Path) -> Path:
    path = figure_dir / "cebt_architecture.png"
    boxes = [
        (0.07, 0.64, "Pre-event price, volume,\nmarket-relative history"),
        (0.07, 0.24, "Disclosure text\nand filing metadata"),
        (0.38, 0.64, "NoEventDynamics\nTransformer encoder"),
        (0.38, 0.24, "EventBottleneck\ncompact stochastic latent"),
        (0.68, 0.64, "No-event\nforecast"),
        (0.68, 0.24, "ResidualOutcomeHead\nevent delta"),
        (0.77, 0.44, "Final prediction\nbaseline + delta"),
    ]
    arrows = [
        ((0.28, 0.74), (0.38, 0.74)),
        ((0.28, 0.34), (0.38, 0.34)),
        ((0.59, 0.74), (0.68, 0.74)),
        ((0.59, 0.34), (0.68, 0.34)),
        ((0.78, 0.70), (0.79, 0.56)),
        ((0.78, 0.34), (0.79, 0.48)),
        ((0.50, 0.62), (0.70, 0.39)),
    ]
    plt.figure(figsize=(11, 5.2))
    ax = plt.gca()
    ax.set_axis_off()
    colors = ["#b9c5d2", "#b9c5d2", "#8fb8e9", "#e2c75c", "#6f7f90", "#5f6f80", "#4f82d6"]
    text_colors = ["#111827", "#111827", "#111827", "#111827", "white", "white", "white"]
    for (x, y, text), color, text_color in zip(boxes, colors, text_colors, strict=False):
        ax.add_patch(
            plt.Rectangle((x, y), 0.20, 0.16, facecolor=color, edgecolor="black", linewidth=1.2)
        )
        ax.text(x + 0.10, y + 0.08, text, ha="center", va="center", fontsize=11, color=text_color)
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "linewidth": 1.6, "color": "black"},
        )
    ax.text(
        0.39,
        0.08,
        "Control loss suppresses event delta on matched no-event dates",
        ha="center",
        fontsize=11,
        color="#242424",
    )
    ax.annotate(
        "",
        xy=(0.70, 0.31),
        xytext=(0.44, 0.12),
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#242424"},
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _plot_prediction_scatter(predictions: list[dict], figure_dir: Path) -> Path:
    cebt_rows = [row for row in predictions if row.get("model") == "cebt"]
    rows = cebt_rows or predictions
    x = [row["prediction_abnormal_return"] for row in rows]
    y = [row["target_abnormal_return"] for row in rows]
    path = figure_dir / "prediction_scatter.png"
    plt.figure(figsize=(5.5, 5.0))
    plt.scatter(x, y, s=24, alpha=0.75, color="#4fbf78")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Predicted abnormal return")
    plt.ylabel("Realized abnormal return")
    plt.title("DJB Event-Return Predictions")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _plot_prediction_phase_portrait(predictions: list[dict], figure_dir: Path) -> Path:
    cebt_rows = _event_rows([row for row in predictions if row.get("model") == "cebt"])
    rows = cebt_rows or _event_rows(predictions) or predictions
    x = np.asarray([row["prediction_abnormal_return"] for row in rows], dtype=float)
    y = np.asarray([row["target_abnormal_return"] for row in rows], dtype=float)
    path = figure_dir / "prediction_phase_portrait.png"
    plt.figure(figsize=(6.0, 5.2))
    plt.hexbin(x, y, gridsize=30, mincnt=1, cmap="Greys")
    plt.colorbar(label="Held-out events per cell")
    plt.axhline(0.0, color="white", linewidth=1.0, alpha=0.9)
    plt.axvline(0.0, color="white", linewidth=1.0, alpha=0.9)
    plt.xlabel("Predicted abnormal return")
    plt.ylabel("Realized abnormal return")
    plt.title("DJB Event-Return Phase Portrait")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def _plot_temporal_rank_ic_heatmap(predictions: list[dict], figure_dir: Path) -> Path:
    rows = _event_rows(predictions)
    models = _model_order({row["model"]: {} for row in rows if row.get("model")})
    months = sorted({str(row.get("label_start_date", ""))[:7] for row in rows})
    months = [month for month in months if len(month) == 7]
    matrix = np.full((len(models), len(months)), np.nan, dtype=float)
    for row_idx, model in enumerate(models):
        model_rows = [row for row in rows if row.get("model") == model]
        for col_idx, month in enumerate(months):
            bucket = [
                row
                for row in model_rows
                if str(row.get("label_start_date", "")).startswith(month)
            ]
            if len(bucket) >= 8:
                matrix[row_idx, col_idx] = rank_ic(
                    np.asarray([row["prediction_abnormal_return"] for row in bucket], dtype=float),
                    np.asarray([row["target_abnormal_return"] for row in bucket], dtype=float),
                ) or 0.0
    path = figure_dir / "temporal_rank_ic_heatmap.png"
    width = max(8.0, 0.42 * len(months))
    plt.figure(figsize=(width, 3.8))
    im = plt.imshow(matrix, aspect="auto", cmap="Greys", vmin=-0.35, vmax=0.35)
    plt.colorbar(im, label="Monthly event rank IC")
    plt.yticks(range(len(models)), [_display_name(model).replace("\n", " ") for model in models])
    plt.xticks(range(len(months)), months, rotation=45, ha="right")
    plt.title("Where Event Ranking Signal Appears Over Time")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def _plot_event_residual_mosaic(predictions: list[dict], figure_dir: Path) -> Path:
    rows = _event_rows([row for row in predictions if row.get("model") == "cebt"])
    tickers = _top_values([row.get("ticker", "UNKNOWN") for row in rows], limit=28)
    months = sorted({str(row.get("label_start_date", ""))[:7] for row in rows})
    months = [month for month in months if len(month) == 7]
    matrix = np.full((len(tickers), len(months)), np.nan, dtype=float)
    for row_idx, ticker in enumerate(tickers):
        ticker_rows = [row for row in rows if row.get("ticker", "UNKNOWN") == ticker]
        for col_idx, month in enumerate(months):
            values = [
                float(row["event_delta_abs_mean"])
                for row in ticker_rows
                if str(row.get("label_start_date", "")).startswith(month)
            ]
            if values:
                matrix[row_idx, col_idx] = float(np.mean(values))
    path = figure_dir / "event_residual_mosaic.png"
    width = max(8.0, 0.42 * len(months))
    height = max(5.0, 0.18 * len(tickers))
    plt.figure(figsize=(width, height))
    masked = np.ma.masked_invalid(matrix)
    im = plt.imshow(masked, aspect="auto", cmap="Greys")
    plt.colorbar(im, label="Mean absolute DJB event residual")
    plt.yticks(range(len(tickers)), tickers)
    plt.xticks(range(len(months)), months, rotation=45, ha="right")
    plt.title("Disclosure Residual Mosaic by Company and Month")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def _plot_residual_distribution(predictions: list[dict], figure_dir: Path) -> Path:
    rows = [row for row in predictions if row.get("model") == "cebt"]
    true_values = np.asarray(
        [
            row["event_delta_abs_mean"]
            for row in rows
            if row.get("control_type") == "real_event"
        ],
        dtype=float,
    )
    control_values = np.asarray(
        [
            row["event_delta_abs_mean"]
            for row in rows
            if row.get("control_type") != "real_event"
        ],
        dtype=float,
    )
    path = figure_dir / "residual_distribution.png"
    plt.figure(figsize=(7.0, 4.2))
    bins = np.linspace(0.0, max(_safe_max(true_values), _safe_max(control_values), 1e-6), 36)
    plt.hist(
        true_values,
        bins=bins,
        density=True,
        alpha=0.65,
        label="True 8-K events",
        color="#4fbf78",
    )
    plt.hist(
        control_values,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Matched no-event controls",
        color="#d1b84e",
    )
    plt.xlabel("Absolute event residual")
    plt.ylabel("Density")
    plt.title("DJB Residual Mass Shifts Away From No-Event Controls")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def _event_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("control_type") == "real_event"]


def _top_values(values: list[str], limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [value for value, _ in ranked[:limit]]


def _safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmax(values))
