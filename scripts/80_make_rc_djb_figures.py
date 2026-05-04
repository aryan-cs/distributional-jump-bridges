"""Generate paper-ready RC-DJB figures from recorded experiment artifacts."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter

METRICS_PATH = Path("paper/tables/table_eval_metrics.csv")
INTERVENTIONS_PATH = Path("paper/tables/table_rc_djb_interventions.csv")
INTERVENTION_PAIRED_PATH = Path("paper/tables/table_rc_djb_intervention_paired.csv")
FEATURE_METADATA_PATH = Path("data/processed/paper_v3_bge/features_metadata.jsonl")
RCDJB_FULL_PATH = Path("data/runs/paper_v3_bge_rc_djb_best/rc_djb_predictions.jsonl")
RCDJB_NO_JUMP_PATH = Path("data/runs/paper_v3_bge_rc_djb_best/rc_djb_no_jump_predictions.jsonl")
RCDJB_ZERO_TEXT_PATH = Path(
    "data/runs/paper_v3_bge_rc_djb_best/rc_djb_zero_event_predictions.jsonl"
)
RCDJB_SHUFFLED_TEXT_PATH = Path(
    "data/runs/paper_v3_bge_rc_djb_best/rc_djb_shuffle_event_predictions.jsonl"
)

MONTHLY_PREDICTION_SOURCES = {
    "Concat": Path("data/runs/paper_v3/concat_predictions.jsonl"),
    "Event bottleneck": Path("data/runs/paper_v3/cebt_predictions.jsonl"),
    "Event-jump SSM": Path("data/runs/paper_v3_bge_ejssm_balanced/ejssm_predictions.jsonl"),
    "DJB": Path("data/runs/paper_v3_bge_djb_best/djb_predictions.jsonl"),
    "RC-DJB": Path("data/runs/paper_v3_bge_rc_djb_best/rc_djb_predictions.jsonl"),
}

MODEL_ORDER = ["no_event", "concat", "cebt", "dot", "ejssm_bge", "djb", "rc_djb"]
DISPLAY_NAMES = {
    "no_event": "No-event",
    "concat": "Concat fusion",
    "cebt": "Event bottleneck",
    "dot": "Disclosure operator",
    "ejssm_bge": "Event-jump SSM",
    "djb": "DJB",
    "rc_djb": "RC-DJB",
}
COLOR_MODELS = {
    "no_event": "#7a8898",
    "concat": "#5f6f80",
    "cebt": "#8a6fd1",
    "dot": "#4aa7bf",
    "ejssm_bge": "#4f82d6",
    "djb": "#d1b84e",
    "rc_djb": "#4fbf78",
}


def main() -> None:
    output_dir = Path("paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_style()
    make_architecture_figure(output_dir / "rc_djb_architecture.png")
    make_pareto_frontier(output_dir / "rc_djb_pareto_frontier.png")
    make_data_temporal_audit(output_dir / "rc_djb_data_temporal_audit.png")
    make_bridge_vector_field(output_dir / "rc_djb_bridge_vector_field.png")
    make_rank_decile_ribbons(output_dir / "rc_djb_rank_decile_ribbons.png")
    make_bridge_delta_ridges(output_dir / "rc_djb_bridge_delta_ridges.png")
    make_firewall_audit(output_dir / "rc_djb_firewall_audit.png")
    make_bridge_activation_ladder(output_dir / "rc_djb_bridge_activation_ladder.png")
    make_monthly_rank_heatmap(output_dir / "rc_djb_monthly_rank_heatmap.png")


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#242424",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, path: Path, *, dpi: int = 240) -> None:
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_architecture_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    palette = {
        "ink": "#111827",
        "muted": "#6b7280",
        "input": "#b9c5d2",
        "state": "#8fb8e9",
        "bridge": "#e2c75c",
        "constraint": "#78d79a",
        "transport": "#b39ee5",
        "output": "#71d7ae",
        "line": "#374151",
        "protected": "#242424",
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
            boxstyle="round,pad=0.006,rounding_size=0.012",
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
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0",
            joinstyle="miter",
            capstyle="butt",
        )
        ax.add_patch(patch)

    def elbow_arrow(
        points: list[tuple[float, float]],
        *,
        color: str = palette["line"],
        lw: float = 1.35,
    ) -> None:
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        patch = FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            joinstyle="miter",
            capstyle="butt",
        )
        ax.add_patch(patch)

    ax.text(0.275, 0.940, "Market stream", fontsize=9.0, color=palette["muted"], ha="center")
    ax.text(
        0.725,
        0.940,
        "Disclosure stream",
        fontsize=9.0,
        color=palette["muted"],
        ha="center",
    )
    stage_labels = [
        ("Inputs", 0.835),
        ("Encoders", 0.625),
        ("Transport", 0.415),
        ("Output", 0.190),
    ]
    for label, y in stage_labels:
        ax.text(
            0.075,
            y,
            label,
            fontsize=8.6,
            color=palette["muted"],
            ha="right",
            va="center",
        )

    left_x = 0.135
    right_x = 0.550
    box_w = 0.315
    box_h = 0.092
    y_input = 0.795
    y_encoder = 0.590
    y_transport = 0.385
    y_output = 0.135
    output_x = 0.340
    output_w = 0.320
    output_h = 0.112

    box(
        (left_x, y_input),
        (box_w, box_h),
        "Pre-event market state",
        r"$X_{<t}$",
        palette["input"],
    )
    box((right_x, y_input), (box_w, box_h), "SEC 8-K disclosure", r"$e_t$", "#e2c75c")
    box(
        (left_x, y_encoder),
        (box_w, box_h),
        "No-event distribution",
        r"$(\mu_0,\sigma_0)$",
        palette["state"],
        title_size=9.6,
        subtitle_size=9.2,
    )
    box(
        (right_x, y_encoder),
        (box_w, box_h),
        "Disclosure bridge",
        r"$B(e_t)$",
        palette["bridge"],
        title_size=9.6,
        subtitle_size=9.2,
    )

    box(
        (left_x, y_transport),
        (box_w, box_h),
        "Protected mean",
        r"$\Delta\mu_r=0$",
        palette["constraint"],
        title_size=9.6,
        subtitle_size=9.0,
    )
    box(
        (right_x, y_transport),
        (box_w, box_h),
        "Allowed shifts",
        r"$\Delta\mu_{\sigma,V},\ \Delta\log\sigma^2$",
        palette["transport"],
        title_size=9.4,
        subtitle_size=8.2,
    )
    box(
        (output_x, y_output),
        (output_w, output_h),
        "Event response",
        r"$(\mu_1,\sigma_1)$",
        palette["output"],
        title_size=9.6,
        subtitle_size=9.2,
    )

    edge_gap = 0.012
    left_center = left_x + box_w / 2
    right_center = right_x + box_w / 2
    output_top = y_output + output_h

    arrow((left_center, y_input - edge_gap), (left_center, y_encoder + box_h + edge_gap))
    arrow((right_center, y_input - edge_gap), (right_center, y_encoder + box_h + edge_gap))
    arrow(
        (left_center, y_encoder - edge_gap),
        (left_center, y_transport + box_h + edge_gap),
        color=palette["protected"],
        lw=1.25,
    )
    arrow(
        (right_center, y_encoder - edge_gap),
        (right_center, y_transport + box_h + edge_gap),
        color=palette["muted"],
        lw=1.20,
    )
    elbow_arrow(
        [
            (left_center, y_transport - edge_gap),
            (left_center, 0.300),
            (0.430, 0.300),
            (0.430, output_top + edge_gap),
        ],
        color=palette["protected"],
        lw=1.20,
    )
    elbow_arrow(
        [
            (right_center, y_transport - edge_gap),
            (right_center, 0.300),
            (0.570, 0.300),
            (0.570, output_top + edge_gap),
        ],
        color=palette["line"],
        lw=1.20,
    )

    _save(fig, path, dpi=240)


def make_pareto_frontier(path: Path) -> None:
    metrics = _read_metric_table(METRICS_PATH)
    models = [model for model in MODEL_ORDER if model in metrics]
    fig, ax = plt.subplots(figsize=(8.2, 4.55))
    ax.set_facecolor("#fbfdff")
    ax.grid(True, color="#e5edf4", linewidth=0.7)
    ax.axhline(0.0, color="#8091a5", linewidth=0.9)

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
        facecolor="#a7d8c0",
        alpha=0.72,
        label="Lower error than concat, positive rank IC",
    )
    for model, x, y, lo_x, hi_x, lo_y, hi_y in zip(
        models, xs, ys, xlos, xhis, ylos, yhis, strict=True
    ):
        color = COLOR_MODELS[model]
        ax.errorbar(
            x,
            y,
            xerr=[[max(x - lo_x, 0.0)], [max(hi_x - x, 0.0)]],
            yerr=[[max(y - lo_y, 0.0)], [max(hi_y - y, 0.0)]],
            fmt="o",
            markersize=7,
            markeredgecolor="#242424",
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
            markerfacecolor=COLOR_MODELS[model],
            markeredgecolor="#242424",
            markersize=7,
            label=DISPLAY_NAMES[model],
        )
        for model in models
    ]
    handles.append(
        Line2D([0], [0], color="#8091a5", linewidth=1.6, label="95% ticker-clustered CI")
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.28, top=0.88, left=0.12, right=0.98)
    _save(fig, path, dpi=260)


def make_data_temporal_audit(path: Path) -> None:
    rows = _read_prediction_rows(FEATURE_METADATA_PATH)
    label_dates = [_parse_date(row["label_start_date"]) for row in rows]
    months = _month_range(min(label_dates), max(label_dates))
    month_to_x = {month: idx for idx, month in enumerate(months)}
    controls = np.zeros(len(months), dtype=int)
    events = np.zeros(len(months), dtype=int)
    gap_counts: dict[str, dict[int, int]] = {
        "real_event": {},
        "same_ticker_no_event": {},
    }
    for row in rows:
        month = str(row["label_start_date"])[:7]
        if row["control_type"] == "real_event":
            events[month_to_x[month]] += 1
        else:
            controls[month_to_x[month]] += 1
        gap = (
            _parse_date(row["label_start_date"])
            - _parse_date(row["feature_max_date"])
        ).days
        bucket = gap_counts.setdefault(row["control_type"], {})
        bucket[gap] = bucket.get(gap, 0) + 1

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.2),
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.72},
    )

    ax = axes[0]
    x = np.arange(len(months))
    ax.set_facecolor("#fbfdff")
    ax.grid(True, axis="y", color="#e5edf4", linewidth=0.7)
    ax.stackplot(
        x,
        events,
        controls,
        colors=["#4fbf78", "#4f82d6"],
        alpha=0.92,
        labels=["Real 8-K rows", "Matched controls"],
    )
    for boundary, label in [("2023-12", "Train cutoff"), ("2024-12", "Validation cutoff")]:
        if boundary in month_to_x:
            bx = month_to_x[boundary] + 0.5
            ax.axvline(bx, color="#8091a5", linewidth=0.9, linestyle="--")
            ax.text(
                bx + 0.25,
                ax.get_ylim()[1] * 0.92,
                label,
                fontsize=8,
                color="#334155",
                ha="left",
                va="top",
            )
    tick_idx = list(range(0, len(months), 6))
    if len(months) - 1 - tick_idx[-1] >= 3:
        tick_idx.append(len(months) - 1)
    ax.set_xticks(
        tick_idx,
        [months[idx] for idx in tick_idx],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Rows per month")
    ax.set_title("Chronological sample and leakage gate audit", pad=10, fontweight="semibold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    ax = axes[1]
    ax.set_facecolor("#fbfdff")
    ax.grid(True, axis="x", color="#e5edf4", linewidth=0.7, zorder=1)
    gap_palette = {
        1: "#4fbf78",
        2: "#8a6fd1",
        3: "#4f82d6",
        4: "#d1b84e",
    }
    y_lookup = {"Real 8-K rows": "real_event", "Matched controls": "same_ticker_no_event"}
    gap_values = sorted({gap for bucket in gap_counts.values() for gap in bucket})
    for y, (_label, control_type) in enumerate(y_lookup.items()):
        total = sum(gap_counts[control_type].values())
        left = 0.0
        for gap in gap_values:
            count = gap_counts[control_type].get(gap, 0)
            percent = 100.0 * count / total if total else 0.0
            if percent == 0.0:
                continue
            ax.barh(
                y,
                percent,
                left=left,
                height=0.42,
                color=gap_palette.get(gap, "#e5e7eb"),
                edgecolor="#334155",
                linewidth=0.55,
                zorder=3,
            )
            if percent >= 4.0:
                ax.text(
                    left + percent / 2,
                    y,
                    f"{gap}d\n{percent:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="#242424",
                    linespacing=0.95,
                )
            left += percent
        ax.text(
            101.0,
            y,
            f"n={total:,}",
            ha="left",
            va="center",
            fontsize=8,
            color="#334155",
        )
    ax.set_yticks([0, 1], list(y_lookup))
    ax.set_xlim(0.0, 109.0)
    ax.set_ylim(-0.55, 1.45)
    ax.set_xlabel("Share of rows by guard margin")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.set_title(
        "Feature-label guard margin distribution",
        loc="left",
        fontsize=10,
        fontweight="semibold",
    )
    ax.legend(
        handles=[
            Patch(
                facecolor=gap_palette.get(gap, "#e5e7eb"),
                edgecolor="#334155",
                linewidth=0.55,
                label=f"{gap} calendar day{'s' if gap != 1 else ''}",
            )
            for gap in gap_values
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=len(gap_values),
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.18, top=0.92, left=0.16, right=0.95)
    _save(fig, path, dpi=260)


def make_bridge_vector_field(path: Path) -> None:
    full = _read_prediction_rows(RCDJB_FULL_PATH)
    base_by_id = {row["sample_id"]: row for row in _read_prediction_rows(RCDJB_NO_JUMP_PATH)}
    joined = [(row, base_by_id[row["sample_id"]]) for row in full if row["sample_id"] in base_by_id]
    real = [(row, base) for row, base in joined if row["control_type"] == "real_event"]
    controls = [(row, base) for row, base in joined if row["control_type"] != "real_event"]

    x_real = np.asarray([base["prediction_volatility_jump"] for _, base in real], dtype=float)
    y_real = np.asarray([base["prediction_volume_jump"] for _, base in real], dtype=float)
    dx_real = np.asarray(
        [
            row["prediction_volatility_jump"] - base["prediction_volatility_jump"]
            for row, base in real
        ],
        dtype=float,
    )
    dy_real = np.asarray(
        [row["prediction_volume_jump"] - base["prediction_volume_jump"] for row, base in real],
        dtype=float,
    )
    x_ctl = np.asarray([base["prediction_volatility_jump"] for _, base in controls], dtype=float)
    y_ctl = np.asarray([base["prediction_volume_jump"] for _, base in controls], dtype=float)

    xlim = _robust_limits(np.concatenate([x_real, x_ctl]), padding=0.08)
    ylim = _robust_limits(np.concatenate([y_real, y_ctl]), padding=0.08)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.set_facecolor("#fbfdff")
    hist, x_edges, y_edges = np.histogram2d(x_real, y_real, bins=42, range=[xlim, ylim])
    cmap = LinearSegmentedColormap.from_list(
        "transport_density",
        ["#f3fbf6", "#b7e9c7", "#73d997", "#4fbf78"],
    )
    ax.imshow(
        hist.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap,
        aspect="auto",
        alpha=0.90,
        zorder=0,
    )
    ax.scatter(
        x_ctl,
        y_ctl,
        s=8,
        color="#4f82d6",
        alpha=0.42,
        linewidths=0,
        label="Matched controls",
        zorder=1,
    )

    bins = 7
    qx, qy, qu, qv = _binned_vector_field(
        x_real,
        y_real,
        dx_real,
        dy_real,
        xlim,
        ylim,
        bins=bins,
        min_count=18,
    )
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    cell_w = x_span / bins
    cell_h = y_span / bins
    normalized_u = qu / x_span
    normalized_v = qv / y_span
    norms = np.sqrt(normalized_u**2 + normalized_v**2)
    max_norm = float(np.max(norms)) if norms.size else 1.0
    for x0, y0, ux, vy, norm in zip(qx, qy, normalized_u, normalized_v, norms, strict=True):
        if norm <= 0.0:
            continue
        strength = 0.45 + 0.55 * min(norm / max_norm, 1.0)
        x1 = x0 + (ux / norm) * cell_w * 0.34 * strength
        y1 = y0 + (vy / norm) * cell_h * 0.34 * strength
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "mutation_scale": 12,
                "linewidth": 1.55,
                "color": "#242424",
                "alpha": 0.92,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=4,
        )
        ax.scatter(
            [x0],
            [y0],
            s=12,
            color="#242424",
            edgecolor="white",
            linewidth=0.35,
            zorder=5,
        )

    ax.axhline(0.0, color="#9aa8b8", linewidth=0.8)
    ax.axvline(0.0, color="#9aa8b8", linewidth=0.8)
    ax.grid(True, color="#e5edf4", linewidth=0.65)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("No-bridge volatility-jump prediction")
    ax.set_ylabel("No-bridge volume-jump prediction")
    ax.set_title("Disclosure bridge transport field", pad=12, fontweight="semibold")
    ax.legend(
        handles=[
            Patch(facecolor="#4fbf78", edgecolor="none", label="Real 8-K density"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#4f82d6",
                markersize=6,
                label="Matched controls",
            ),
            Line2D([0], [0], color="#242424", linewidth=2.0, label="Mean bridge vector"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.21, top=0.88, left=0.12, right=0.98)
    _save(fig, path, dpi=260)


def make_rank_decile_ribbons(path: Path) -> None:
    sources = {
        "Concat": MONTHLY_PREDICTION_SOURCES["Concat"],
        "Event bottleneck": MONTHLY_PREDICTION_SOURCES["Event bottleneck"],
        "Event-jump SSM": MONTHLY_PREDICTION_SOURCES["Event-jump SSM"],
        "DJB": MONTHLY_PREDICTION_SOURCES["DJB"],
        "RC-DJB": MONTHLY_PREDICTION_SOURCES["RC-DJB"],
    }
    colors = {
        "Concat": "#5f6f80",
        "Event bottleneck": "#8a6fd1",
        "Event-jump SSM": "#4f82d6",
        "DJB": "#d1b84e",
        "RC-DJB": "#4fbf78",
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.set_facecolor("#fbfdff")
    ax.grid(True, color="#e5edf4", linewidth=0.7)
    x = np.arange(1, 11)
    for name, source in sources.items():
        rows = [
            row
            for row in _read_prediction_rows(source)
            if row.get("control_type") == "real_event"
        ]
        scores = np.asarray([row["prediction_abnormal_return"] for row in rows], dtype=float)
        targets = np.asarray([row["target_abnormal_return"] for row in rows], dtype=float)
        means, lows, highs = _decile_bootstrap_profile(scores, targets)
        is_main = name == "DJB"
        ax.plot(
            x,
            means,
            color=colors[name],
            linewidth=2.6 if is_main else 1.4,
            marker="o" if is_main else None,
            markersize=4.5,
            alpha=1.0 if is_main else 0.84,
            label=name,
            zorder=4 if is_main else 3,
        )
        ax.fill_between(
            x,
            lows,
            highs,
            color=colors[name],
            alpha=0.24 if is_main else 0.12,
            linewidth=0,
            zorder=2 if is_main else 1,
        )
    ax.axhline(0.0, color="#8091a5", linewidth=0.9)
    ax.set_xlim(0.6, 10.4)
    ax.set_xticks(x)
    ax.set_xlabel("Predicted abnormal-return decile")
    ax.set_ylabel("Realized five-day abnormal return")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:.1f}%"))
    ax.set_title("Rank profiles reveal where signal concentrates", pad=10, fontweight="semibold")
    ax.text(
        0.015,
        0.955,
        "Lines are held-out real 8-K decile means; ribbons are 95% row-bootstrap intervals.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d8e2ea"},
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=0.23, top=0.88, left=0.12, right=0.98)
    _save(fig, path, dpi=260)


def make_bridge_delta_ridges(path: Path) -> None:
    sources = {
        "Full": RCDJB_FULL_PATH,
        "No bridge": RCDJB_NO_JUMP_PATH,
        "Zero text": RCDJB_ZERO_TEXT_PATH,
        "Shuffled text": RCDJB_SHUFFLED_TEXT_PATH,
    }
    values_by_variant: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_values: list[float] = []
    for label, source in sources.items():
        rows = _read_prediction_rows(source)
        real = np.asarray(
            [
                row["event_delta_abs_mean"]
                for row in rows
                if row.get("control_type") == "real_event"
            ],
            dtype=float,
        )
        controls = np.asarray(
            [
                row["event_delta_abs_mean"]
                for row in rows
                if row.get("control_type") != "real_event"
            ],
            dtype=float,
        )
        values_by_variant[label] = (real, controls)
        all_values.extend(real.tolist())
        all_values.extend(controls.tolist())

    max_x = float(np.percentile(np.asarray(all_values), 99.3))
    max_x = max(max_x, 0.01)
    bins = np.linspace(0.0, max_x, 75)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.set_facecolor("#fbfdff")
    ax.grid(True, axis="x", color="#e5edf4", linewidth=0.7)
    labels = list(sources)
    offsets = np.arange(len(labels))[::-1].astype(float)
    for offset, label in zip(offsets, labels, strict=True):
        real, controls = values_by_variant[label]
        for values, color, alpha, line_color in [
            (controls, "#4f82d6", 0.66, "#3c6db9"),
            (real, "#4fbf78", 0.72, "#2f7f53"),
        ]:
            hist, _ = np.histogram(values, bins=bins)
            if hist.max() > 0:
                density = hist / hist.max() * 0.42
            else:
                density = hist.astype(float)
            ax.fill_between(
                centers,
                offset,
                offset + density,
                color=color,
                alpha=alpha,
                linewidth=0,
                zorder=2,
            )
            ax.plot(centers, offset + density, color=line_color, linewidth=0.9, zorder=3)
        median_real = float(np.median(real)) if real.size else 0.0
        median_ctl = float(np.median(controls)) if controls.size else 0.0
        ax.plot(
            [median_real, median_real],
            [offset, offset + 0.44],
            color="#242424",
            linewidth=1.05,
        )
        ax.plot([median_ctl, median_ctl], [offset, offset + 0.36], color="#3c6db9", linewidth=1.05)
    ax.set_yticks(offsets + 0.20, labels)
    ax.set_xlabel(r"Mean absolute bridge transport $|\Delta\mu|$")
    ax.set_ylabel("Intervention")
    ax.set_xlim(0.0, max_x)
    ax.set_ylim(-0.18, offsets[0] + 0.72)
    ax.set_title(
        "Response-transport fingerprints under interventions",
        pad=10,
        fontweight="semibold",
    )
    ax.legend(
        handles=[
            Patch(facecolor="#4fbf78", edgecolor="none", alpha=0.78, label="Real 8-K rows"),
            Patch(facecolor="#4f82d6", edgecolor="none", alpha=0.78, label="Matched controls"),
            Line2D([0], [0], color="#334155", linewidth=1.0, label="Within-row median"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.23, top=0.88, left=0.14, right=0.98)
    _save(fig, path, dpi=260)


def make_firewall_audit(path: Path) -> None:
    full_by_id = {row["sample_id"]: row for row in _read_prediction_rows(RCDJB_FULL_PATH)}
    variants = {
        "No bridge": RCDJB_NO_JUMP_PATH,
        "Zero text": RCDJB_ZERO_TEXT_PATH,
        "Shuffled text": RCDJB_SHUFFLED_TEXT_PATH,
    }
    targets = [
        ("prediction_abnormal_return", "Abnormal return"),
        ("prediction_volatility_jump", "Volatility jump"),
        ("prediction_volume_jump", "Volume jump"),
    ]
    colors = {
        "No bridge": "#7a8898",
        "Zero text": "#d65a6e",
        "Shuffled text": "#8a6fd1",
    }
    offsets = {"No bridge": -0.18, "Zero text": 0.0, "Shuffled text": 0.18}

    stats: list[tuple[str, str, float, float, float]] = []
    for variant, source in variants.items():
        rows = {row["sample_id"]: row for row in _read_prediction_rows(source)}
        for target, label in targets:
            diffs = np.asarray(
                [
                    abs(full_by_id[sample_id][target] - rows[sample_id][target])
                    for sample_id in full_by_id
                    if sample_id in rows
                ],
                dtype=float,
            )
            mean, lo, hi = _bootstrap_mean_ci(diffs)
            stats.append((variant, label, mean, lo, hi))

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.set_facecolor("#fbfdff")
    ax.grid(True, axis="x", color="#e5edf4", linewidth=0.7)
    ax.axvline(0.0, color="#8091a5", linewidth=0.9)
    y_base = {label: idx for idx, (_, label) in enumerate(targets)}
    for variant, label, mean, lo, hi in stats:
        y = y_base[label] + offsets[variant]
        ax.errorbar(
            mean,
            y,
            xerr=[[max(mean - lo, 0.0)], [max(hi - mean, 0.0)]],
            fmt="o",
            markersize=7,
            color=colors[variant],
            ecolor=colors[variant],
            markeredgecolor="#242424",
            markeredgewidth=0.8,
            elinewidth=2.0,
            capsize=3,
            zorder=3,
        )
    ax.set_yticks(np.arange(len(targets)), [label for _, label in targets])
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute prediction change from full RC-DJB")
    ax.set_title("Return-mean firewall audit across interventions", pad=10, fontweight="semibold")
    ax.set_xlim(-0.001, max(stat[4] for stat in stats) * 1.18)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=colors[variant],
                markeredgecolor="#242424",
                markersize=7,
                label=variant,
            )
            for variant in variants
        ]
        + [Line2D([0], [0], color="#9aa8b8", linewidth=2.0, label="95% row-bootstrap CI")],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.24, top=0.88, left=0.17, right=0.98)
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
        "full": "#4fbf78",
        "no_bridge": "#7a8898",
        "zero_text": "#d65a6e",
        "shuffled_text": "#8a6fd1",
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
    ax.axvline(0.0, color="#8091a5", linewidth=0.9)
    for y, value, lo, hi, color in zip(y_positions, values, lows, highs, colors, strict=True):
        ax.errorbar(
            value,
            y,
            xerr=[[max(value - lo, 0.0)], [max(hi - value, 0.0)]],
            fmt="o",
            markersize=7,
            markeredgecolor="#242424",
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
        handles=[Line2D([0], [0], color="#9aa8b8", linewidth=2.0, label="95% paired bootstrap CI")],
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
        color="#4fbf78",
        edgecolor="#242424",
        linewidth=0.6,
        label="Real 8-K rows",
    )
    ax.bar(
        x + width / 2,
        controls,
        width,
        color="#4f82d6",
        edgecolor="#242424",
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


def make_bridge_activation_ladder(path: Path) -> None:
    rows = [
        row
        for row in _read_prediction_rows(RCDJB_FULL_PATH)
        if row.get("control_type") == "real_event"
    ]
    if not rows:
        raise ValueError("No real-event rows available for bridge activation ladder.")

    score = np.asarray(
        [float(row["event_delta_abs_response_transport"]) for row in rows],
        dtype=float,
    )
    order = np.argsort(score)
    deciles = np.empty_like(order)
    deciles[order] = np.minimum((np.arange(score.size) * 10) // max(score.size, 1), 9)
    x = np.arange(1, 11)

    vol_transport = np.asarray(
        [float(row["event_delta_abs_volatility"]) for row in rows],
        dtype=float,
    )
    volume_transport = np.asarray(
        [float(row["event_delta_abs_volume"]) for row in rows],
        dtype=float,
    )
    realized_vol = np.abs(
        np.asarray([float(row["target_volatility_jump"]) for row in rows], dtype=float)
    )
    realized_volume = np.abs(
        np.asarray([float(row["target_volume_jump"]) for row in rows], dtype=float)
    )
    realized_vol = realized_vol / max(float(np.std(realized_vol)), 1e-8)
    realized_volume = realized_volume / max(float(np.std(realized_volume)), 1e-8)

    def decile_means(values: np.ndarray) -> np.ndarray:
        return np.asarray([float(np.mean(values[deciles == idx])) for idx in range(10)])

    def decile_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for idx in range(10):
            mean, lo, hi = _bootstrap_mean_ci(values[deciles == idx])
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
        return np.asarray(means), np.asarray(lows), np.asarray(highs)

    vol_means = decile_means(vol_transport)
    volume_means = decile_means(volume_transport)
    counts = np.asarray([int(np.sum(deciles == idx)) for idx in range(10)])
    vol_response, vol_low, vol_high = decile_ci(realized_vol)
    volume_response, volume_low, volume_high = decile_ci(realized_volume)

    fig = plt.figure(figsize=(9.2, 6.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.30)
    ax_transport = fig.add_subplot(grid[0])
    ax_response = fig.add_subplot(grid[1], sharex=ax_transport)

    for ax in [ax_transport, ax_response]:
        ax.set_facecolor("#fbfdff")
        ax.grid(True, axis="y", color="#e5edf4", linewidth=0.7)
        ax.set_xlim(0.35, 10.65)
        for spine in ax.spines.values():
            spine.set_color("#8091a5")

    vol_floor = float(np.min(vol_means))
    vol_ceiling = float(np.max(vol_means))
    ax_transport.fill_between(
        [0.55, 10.45],
        [vol_floor, vol_floor],
        [vol_ceiling, vol_ceiling],
        color="#77b0e4",
        alpha=0.18,
        linewidth=0.0,
        label=r"Volatility shift range $|\Delta\mu_\sigma|$",
    )
    ax_transport.plot(
        x,
        vol_means,
        color="#4f82d6",
        linewidth=1.7,
        marker="o",
        markersize=3.2,
        markeredgecolor="#242424",
        markeredgewidth=0.45,
        label=r"Volatility shift mean",
    )
    ax_transport.bar(
        x,
        volume_means,
        width=0.72,
        color="#6ed49a",
        edgecolor="#334155",
        linewidth=0.55,
        label=r"Volume shift $|\Delta\mu_V|$",
    )
    ax_transport.axhline(0.0, color="#8091a5", linewidth=0.9)
    ax_transport.set_ylabel("Mean bridge\ntransport")
    ax_transport.set_title(
        "Bridge activation ladder across held-out real 8-K rows",
        pad=10,
        fontweight="semibold",
    )
    ax_transport.annotate(
        "",
        xy=(10.0, 0.96),
        xytext=(1.0, 0.96),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops={"arrowstyle": "->", "linewidth": 1.1, "color": "#8091a5"},
    )
    ax_transport.text(
        5.5,
        0.985,
        "Rows sorted from weakest to strongest response transport",
        transform=ax_transport.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#334155",
    )
    ax_transport.text(
        1.0,
        float(np.mean(vol_means)) * 1.04,
        "volatility channel is nearly flat",
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#334155",
    )
    ymax = max(float(np.max(vol_means)), float(np.max(volume_means)))
    for idx, count in enumerate(counts, start=1):
        ax_transport.text(
            idx,
            ymax * 1.18,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#64748b",
        )
    ax_transport.set_ylim(0.0, ymax * 1.34)
    ax_transport.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=7.8,
    )

    response_specs = [
        (
            volume_response,
            volume_low,
            volume_high,
            "#4fbf78",
            "Realized volume response",
        ),
        (
            vol_response,
            vol_low,
            vol_high,
            "#4f82d6",
            "Realized volatility response",
        ),
    ]
    for means, lows, highs, color, label in response_specs:
        yerr = np.vstack([means - lows, highs - means])
        ax_response.errorbar(
            x,
            means,
            yerr=yerr,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.2,
            markeredgecolor="#242424",
            markeredgewidth=0.55,
            capsize=3,
            label=label,
        )
    ax_response.set_ylabel("Mean absolute realized\nresponse, SD units")
    ax_response.set_xlabel("Bridge response-transport decile")
    ax_response.set_xticks(x, [str(idx) for idx in x])
    ax_response.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.51,
        0.02,
        "Return-mean transport is fixed at zero by the RC-DJB firewall; "
        "the ladder isolates the risk and liquidity channels.",
        ha="center",
        va="center",
        fontsize=8,
        color="#334155",
    )
    fig.subplots_adjust(bottom=0.18, top=0.93, left=0.11, right=0.98)
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
        ["#d65a6e", "#ef9caa", "#fbfbfb", "#b7e9c7", "#4fbf78"],
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
        spine.set_color("#8091a5")
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
                    color="#8091a5",
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


def _parse_date(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def _month_range(start: date, end: date) -> list[str]:
    months: list[str] = []
    year = start.year
    month = start.month
    while (year, month) <= (end.year, end.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


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


def _robust_limits(values: np.ndarray, *, padding: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if np.isclose(lo, hi):
        lo, hi = float(lo) - 0.5, float(hi) + 0.5
    span = float(hi - lo)
    return float(lo - padding * span), float(hi + padding * span)


def _binned_vector_field(
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    bins: int = 9,
    min_count: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], bins + 1)
    qx: list[float] = []
    qy: list[float] = []
    qu: list[float] = []
    qv: list[float] = []
    for x_idx in range(bins):
        for y_idx in range(bins):
            mask = (
                (x >= x_edges[x_idx])
                & (x < x_edges[x_idx + 1])
                & (y >= y_edges[y_idx])
                & (y < y_edges[y_idx + 1])
            )
            if int(mask.sum()) < min_count:
                continue
            qx.append(float(np.mean(x[mask])))
            qy.append(float(np.mean(y[mask])))
            qu.append(float(np.mean(dx[mask])))
            qv.append(float(np.mean(dy[mask])))
    return np.asarray(qx), np.asarray(qy), np.asarray(qu), np.asarray(qv)


def _decile_bootstrap_profile(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    num_bootstrap: int = 700,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores)
    deciles = np.empty_like(order)
    deciles[order] = np.minimum((np.arange(scores.size) * 10) // max(scores.size, 1), 9)
    rng = np.random.default_rng(19)
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for decile in range(10):
        bucket = targets[deciles == decile]
        if bucket.size == 0:
            means.append(float("nan"))
            lows.append(float("nan"))
            highs.append(float("nan"))
            continue
        means.append(float(np.mean(bucket)))
        if bucket.size == 1:
            lows.append(float(bucket[0]))
            highs.append(float(bucket[0]))
            continue
        draws = rng.choice(bucket, size=(num_bootstrap, bucket.size), replace=True)
        boot_means = np.mean(draws, axis=1)
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        lows.append(float(lo))
        highs.append(float(hi))
    return np.asarray(means), np.asarray(lows), np.asarray(highs)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    num_bootstrap: int = 700,
) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(23)
    draws = rng.choice(values, size=(num_bootstrap, values.size), replace=True)
    boot_means = np.mean(draws, axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean, float(lo), float(hi)


if __name__ == "__main__":
    main()
