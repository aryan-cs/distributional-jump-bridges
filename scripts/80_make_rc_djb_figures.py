"""Generate RC-DJB paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main() -> None:
    output_dir = Path("paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    make_architecture_figure(output_dir / "rc_djb_architecture.png")


def make_architecture_figure(path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.8,
        }
    )
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
        "protected": "#2f6f4e",
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
        rad: float = 0.0,
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
            connectionstyle=f"arc3,rad={rad}",
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
    box((0.055, 0.295), (0.205, 0.130), "SEC 8-K disclosure", r"$e_t$", "#fefce8")
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

    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
