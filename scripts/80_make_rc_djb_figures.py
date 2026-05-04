"""Generate paper-ready RC-DJB figures from recorded experiment artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

METRICS_PATH = Path("paper/tables/table_eval_metrics.csv")
INTERVENTIONS_PATH = Path("paper/tables/table_rc_djb_interventions.csv")
INTERVENTION_PAIRED_PATH = Path("paper/tables/table_rc_djb_intervention_paired.csv")

MONTHLY_PREDICTION_SOURCES = {
    "Concat": Path("data/runs/paper_v3/concat_predictions.jsonl"),
    "CEBT": Path("data/runs/paper_v3/cebt_predictions.jsonl"),
    "EJSSM-BGE": Path("data/runs/paper_v3_bge_ejssm_balanced/ejssm_predictions.jsonl"),
    "DJB": Path("data/runs/paper_v3_bge_djb_best/djb_predictions.jsonl"),
    "RC-DJB": Path("data/runs/paper_v3_bge_rc_djb_best/rc_djb_predictions.jsonl"),
}

MODEL_ORDER = ["no_event", "concat", "cebt", "dot", "ejssm_bge", "djb", "rc_djb"]
DISPLAY_NAMES = {
    "no_event": "No-event",
    "concat": "Concat fusion",
    "cebt": "CEBT",
    "dot": "DOT",
    "ejssm_bge": "EJSSM-BGE",
    "djb": "DJB",
    "rc_djb": "RC-DJB",
}
PASTEL_MODELS = {
    "no_event": "#d8dee9",
    "concat": "#cfd8dc",
    "cebt": "#d7c5ea",
    "dot": "#c7d9f2",
    "ejssm_bge": "#b7d7f2",
    "djb": "#f7c8a9",
    "rc_djb": "#9fd8cb",
}


def main() -> None:
    output_dir = Path("paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_style()
    make_architecture_figure(output_dir / "rc_djb_architecture.png")
    make_pareto_frontier(output_dir / "rc_djb_pareto_frontier.png")
    make_intervention_story(output_dir / "rc_djb_intervention_story.png")
    make_monthly_rank_heatmap(output_dir / "rc_djb_monthly_rank_heatmap.png")


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#1f2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path, *, dpi: int = 240) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_architecture_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    palette = {
        "ink": "#1f2937",
        "muted": "#6b7280",
        "input": "#edf2f7",
        "state": "#dbeafe",
        "bridge": "#fff7ed",
        "constraint": "#dcfce7",
        "transport": "#f5f3ff",
        "output": "#d1fae5",
        "line": "#374151",
        "protected": "#5a9b77",
    }

    def box(
        xy: tuple[float, float],
        wh: tuple[float, float],
        title: str,
        subtitle: str,
        face: str,
        edge: str | None = None,
        title_size: float = 10.2,
        subtitle_size: float = 9.8,
    ) -> None:
        x, y = xy
        w, h = wh
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            facecolor=face,
            edgecolor=edge or palette["ink"],
            linewidth=1.15,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="semibold",
            color=palette["ink"],
        )
        ax.text(
            x + w / 2,
            y + h * 0.32,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=palette["ink"],
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = palette["line"],
        lw: float = 1.45,
    ) -> None:
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color,
            shrinkA=4,
            shrinkB=4,
            connectionstyle="arc3,rad=0",
        )
        ax.add_patch(patch)

    ax.text(0.155, 0.90, "Inputs", fontsize=9.0, color=palette["muted"], ha="center")
    ax.text(0.435, 0.91, "Encoders", fontsize=9.0, color=palette["muted"], ha="center")
    ax.text(
        0.725,
        0.91,
        "Transport constraints",
        fontsize=9.0,
        color=palette["muted"],
        ha="center",
    )
    ax.text(0.935, 0.91, "Output", fontsize=9.0, color=palette["muted"], ha="center")

    box((0.055, 0.690), (0.205, 0.130), "Pre-event market state", r"$X_{<t}$", palette["input"])
    box((0.055, 0.295), (0.205, 0.130), "SEC 8-K disclosure", r"$e_t$", "#fffbe6")
    box(
        (0.340, 0.690),
        (0.190, 0.130),
        "No-event distribution",
        r"$(\mu_0,\sigma_0)$",
        palette["state"],
        title_size=9.4,
        subtitle_size=9.2,
    )
    box(
        (0.340, 0.295),
        (0.190, 0.130),
        "Disclosure bridge",
        r"$B(e_t)$",
        palette["bridge"],
        title_size=9.4,
        subtitle_size=9.2,
    )

    box(
        (0.625, 0.690),
        (0.200, 0.130),
        "Protected mean",
        r"$\Delta\mu_r=0$",
        palette["constraint"],
        title_size=9.2,
        subtitle_size=9.0,
    )
    box(
        (0.625, 0.295),
        (0.200, 0.130),
        "Allowed shifts",
        r"$\Delta\mu_{\sigma,V},\ \Delta\log\sigma^2$",
        palette["transport"],
        title_size=9.0,
        subtitle_size=8.2,
    )
    box(
        (0.890, 0.455),
        (0.095, 0.190),
        "Event response",
        r"$(\mu_1,\sigma_1)$",
        palette["output"],
        title_size=7.8,
        subtitle_size=8.6,
    )

    rail_x = 0.850
    rail_top = 0.755
    rail_bottom = 0.360
    rail_mid = 0.550
    ax.plot(
        [rail_x, rail_x],
        [rail_bottom, rail_top],
        color=palette["muted"],
        linewidth=1.25,
        solid_capstyle="round",
    )

    arrow((0.260, 0.755), (0.340, 0.755))
    arrow((0.260, 0.360), (0.340, 0.360))
    arrow((0.530, 0.755), (0.625, 0.755), color=palette["protected"], lw=1.25)
    arrow((0.530, 0.360), (0.625, 0.360), color=palette["muted"], lw=1.20)
    arrow((0.825, 0.755), (rail_x, 0.755), color=palette["protected"], lw=1.20)
    arrow((0.825, 0.360), (rail_x, 0.360), color=palette["line"], lw=1.20)
    arrow((rail_x, rail_mid), (0.890, rail_mid), color=palette["line"], lw=1.20)

    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    _save(fig, path, dpi=240)


def make_pareto_frontier(path: Path) -> None:
    metrics = _read_metric_table(METRICS_PATH)
    models = [model for model in MODEL_ORDER if model in metrics]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.set_facecolor("#fbfdff")
    ax.grid(True, color="#e5edf4", linewidth=0.7)
    ax.axhline(0.0, color="#64748b", linewidth=0.9)

    xs, ys, xlos, xhis, ylos, yhis = [], [], [], [], [], []
    for model in models:
        row = metrics[model]
        mse = row["mse"]
        rank = row["abnormal_return_rank_ic"]
        mse_ci = row.get("mse_ci_ticker_cluster", {})
        rank_ci = row.get("rank_ic_ci_ticker_cluster", {})
        xs.append(mse)
        ys.append(rank)
        xlos.append(mse_ci.get("lo", mse))
        xhis.append(mse_ci.get("hi", mse))
        ylos.append(rank_ci.get("lo", rank))
        yhis.append(rank_ci.get("hi", rank))

    ax.axvspan(
        min(xlos),
        metrics["concat"]["mse"],
        ymin=0.50,
        ymax=1.00,
        facecolor="#e5f6ef",
        alpha=0.55,
        label="Low MSE + positive rank region",
    )
    for model, x, y, lo_x, hi_x, lo_y, hi_y in zip(
        models, xs, ys, xlos, xhis, ylos, yhis, strict=True
    ):
        color = PASTEL_MODELS[model]
        ax.errorbar(
            x,
            y,
            xerr=[[max(x - lo_x, 0.0)], [max(hi_x - x, 0.0)]],
            yerr=[[max(y - lo_y, 0.0)], [max(hi_y - y, 0.0)]],
            fmt="o",
            markersize=7,
            markeredgecolor="#1f2937",
            markeredgewidth=0.8,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            zorder=3,
        )

    ax.set_xlabel("Held-out multi-target MSE (lower is better)")
    ax.set_ylabel("Abnormal-return rank IC (higher is better)")
    ax.set_title("Response error vs. return-ranking signal", pad=10, fontweight="semibold")
    x_margin = (max(xhis) - min(xlos)) * 0.08
    y_margin = (max(yhis) - min(ylos)) * 0.10
    ax.set_xlim(min(xlos) - x_margin, max(xhis) + x_margin)
    ax.set_ylim(min(ylos) - y_margin, max(yhis) + y_margin)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PASTEL_MODELS[model],
            markeredgecolor="#1f2937",
            markersize=7,
            label=DISPLAY_NAMES[model],
        )
        for model in models
    ]
    handles.append(
        Line2D([0], [0], color="#9fb9d5", linewidth=1.5, label="95% ticker-clustered CI")
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.30, top=0.88, left=0.12, right=0.98)
    _save(fig, path, dpi=260)


def make_intervention_story(path: Path) -> None:
    interventions = _read_rows(INTERVENTIONS_PATH)
    paired = _read_rows(INTERVENTION_PAIRED_PATH)
    intervention_names = {
        "no_bridge": "No bridge",
        "zero_text": "Zero text",
        "shuffled_text": "Shuffled text",
    }
    pastel = {
        "full": "#9fd8cb",
        "no_bridge": "#cfd8dc",
        "zero_text": "#f7c8a9",
        "shuffled_text": "#d7c5ea",
    }
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.2),
        gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.62},
    )

    ax = axes[0]
    ax.set_facecolor("#fbfdff")
    ax.grid(True, axis="x", color="#e5edf4", linewidth=0.7)
    y_positions = np.arange(len(paired))
    values = np.asarray([float(row["intervention_minus_full_mse"]) for row in paired])
    lows = np.asarray([float(row["ci_lo"]) for row in paired])
    highs = np.asarray([float(row["ci_hi"]) for row in paired])
    labels = [intervention_names[row["intervention"]] for row in paired]
    colors = [pastel[row["intervention"]] for row in paired]
    ax.axvline(0.0, color="#64748b", linewidth=0.9)
    for y, value, lo, hi, color in zip(y_positions, values, lows, highs, colors, strict=True):
        ax.errorbar(
            value,
            y,
            xerr=[[max(value - lo, 0.0)], [max(hi - value, 0.0)]],
            fmt="o",
            markersize=7,
            markeredgecolor="#1f2937",
            markeredgewidth=0.8,
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=4,
            zorder=3,
        )
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Event-row MSE increase relative to full RC-DJB")
    ax.set_title(
        "A. Text bridge interventions raise event error",
        loc="left",
        fontweight="semibold",
    )
    ax.set_xlim(0.0, max(highs) * 1.18)
    ax.legend(
        handles=[Line2D([0], [0], color="#9fb9d5", linewidth=2.0, label="95% paired bootstrap CI")],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        frameon=False,
        fontsize=8,
    )

    ax = axes[1]
    ax.set_facecolor("#fbfdff")
    ax.grid(True, axis="y", color="#e5edf4", linewidth=0.7)
    order = ["full", "no_bridge", "zero_text", "shuffled_text"]
    rows_by_name = {row["intervention"]: row for row in interventions}
    x = np.arange(len(order))
    width = 0.34
    real = [float(rows_by_name[name]["mean_abs_event_delta_true_events"]) for name in order]
    controls = [float(rows_by_name[name]["mean_abs_event_delta_controls"]) for name in order]
    ax.bar(
        x - width / 2,
        real,
        width,
        color="#9fd8cb",
        edgecolor="#1f2937",
        linewidth=0.6,
        label="Real 8-K rows",
    )
    ax.bar(
        x + width / 2,
        controls,
        width,
        color="#b7d7f2",
        edgecolor="#1f2937",
        linewidth=0.6,
        label="Matched controls",
    )
    ax.set_ylabel("Mean absolute response transport")
    ax.set_title(
        "B. Bridge transport remains larger on disclosures",
        loc="left",
        fontweight="semibold",
    )
    ax.set_xticks(x, ["Full", "No bridge", "Zero text", "Shuffled text"])
    ax.set_ylim(0.0, max(real + controls) * 1.20 if real or controls else 1.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.18, top=0.94, left=0.18, right=0.98)
    _save(fig, path, dpi=260)


def make_monthly_rank_heatmap(path: Path) -> None:
    series = {
        model: _read_prediction_rows(source)
        for model, source in MONTHLY_PREDICTION_SOURCES.items()
        if source.exists()
    }
    months = sorted(
        {
            str(row.get("label_start_date", ""))[:7]
            for rows in series.values()
            for row in rows
            if row.get("control_type") == "real_event"
        }
    )
    months = [month for month in months if len(month) == 7]
    models = list(series)
    matrix = np.full((len(models), len(months)), np.nan, dtype=float)
    counts = np.zeros((len(models), len(months)), dtype=int)
    for row_idx, model in enumerate(models):
        rows = [row for row in series[model] if row.get("control_type") == "real_event"]
        for col_idx, month in enumerate(months):
            bucket = [
                row
                for row in rows
                if str(row.get("label_start_date", "")).startswith(month)
            ]
            counts[row_idx, col_idx] = len(bucket)
            if len(bucket) >= 8:
                matrix[row_idx, col_idx] = _rank_ic(
                    np.asarray([row["prediction_abnormal_return"] for row in bucket], dtype=float),
                    np.asarray([row["target_abnormal_return"] for row in bucket], dtype=float),
                )

    cmap = LinearSegmentedColormap.from_list(
        "pastel_rank_ic",
        ["#f3b6b6", "#fff7ed", "#e9eef4", "#e6f4ee", "#9fd8cb"],
    )
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-0.35, vmax=0.35)
    ax.set_facecolor("#f8fafc")
    ax.set_title("Monthly event-rank IC pattern", pad=10, fontweight="semibold")
    ax.set_yticks(np.arange(len(models)), models)
    ax.set_xticks(np.arange(len(months)), months, rotation=35, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#64748b")
        spine.set_linewidth(0.8)

    for row_idx in range(counts.shape[0]):
        for col_idx in range(counts.shape[1]):
            if counts[row_idx, col_idx] and counts[row_idx, col_idx] < 8:
                ax.text(
                    col_idx,
                    row_idx,
                    ".",
                    ha="center",
                    va="center",
                    color="#64748b",
                    fontsize=8,
                )

    colorbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.28, fraction=0.08)
    colorbar.set_label("Monthly rank IC; blank cells have fewer than 8 real 8-K rows", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.subplots_adjust(bottom=0.34, top=0.82, left=0.12, right=0.99)
    _save(fig, path, dpi=260)


def _read_metric_table(path: Path) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for row in _read_rows(path):
        model = row["model"]
        metric = row["metric"]
        metrics.setdefault(model, {})
        if row.get("value"):
            try:
                metrics[model][metric] = float(row["value"])
            except ValueError:
                metrics[model][metric] = row["value"]
        elif row.get("mean"):
            metrics[model][metric] = {
                "mean": float(row["mean"]),
                "lo": float(row["lo"]),
                "hi": float(row["hi"]),
            }
    return metrics


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_prediction_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rank_ic(scores: np.ndarray, targets: np.ndarray) -> float:
    if scores.size < 2:
        return float("nan")
    score_ranks = _ranks(scores)
    target_ranks = _ranks(targets)
    score_ranks = score_ranks - np.mean(score_ranks)
    target_ranks = target_ranks - np.mean(target_ranks)
    denom = np.sqrt(float(np.sum(score_ranks**2) * np.sum(target_ranks**2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(score_ranks * target_ranks) / denom)


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


if __name__ == "__main__":
    main()
