"""Figure generation for CEBT paper artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

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
        outputs["event_delta_controls"] = str(_plot_event_delta(metrics, figure_dir))
    if predictions:
        outputs["prediction_scatter"] = str(_plot_prediction_scatter(predictions, figure_dir))
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
    plt.bar(names, values, yerr=yerr, capsize=4, color="#4c78a8")
    plt.ylabel("MSE, lower is better")
    plt.title("Forecast Error With Bootstrap 95% CI")
    plt.xticks(rotation=25, ha="right")
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
    plt.xticks(list(x), names, rotation=25, ha="right")
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
    colors = ["#f4f1de", "#f4f1de", "#81b29a", "#e07a5f", "#3d405b", "#3d405b", "#2a9d8f"]
    text_colors = ["black", "black", "black", "white", "white", "white", "white"]
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
        color="#7f1d1d",
    )
    ax.annotate(
        "",
        xy=(0.70, 0.31),
        xytext=(0.44, 0.12),
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#7f1d1d"},
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
    plt.scatter(x, y, s=24, alpha=0.75, color="#2a9d8f")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Predicted abnormal return")
    plt.ylabel("Realized abnormal return")
    plt.title("CEBT Event-Return Predictions")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path
