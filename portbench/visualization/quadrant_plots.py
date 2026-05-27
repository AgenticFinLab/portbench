"""
S2 vs S4 quadrant scatter plot for PortBench analysis.

Figure: plot_s2_s4_quadrant
  Each point = one model, X = S2 score, Y = S4 score.
  Median-based crosshairs partition the plane into four quadrants
  corresponding to four signal-execution alignment patterns.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .style import apply_paper_style, abbrev_model_name
from .risk_return_plots import _MODEL_COLOURS, _MODEL_MARKERS

_QUADRANT_COLORS = {
    "signal_rich": "#e8f0fe",
    "balanced": "#e6f3e6",
    "weak_both": "#f5f0f0",
    "execution_lean": "#fef3e4",
}


def plot_s2_s4_quadrant(
    stage_scores: dict[str, dict[str, float]],
    title: str = "S2 vs S4: Signal-Execution Dissociation",
    figsize: tuple = (5.5, 5),
) -> Figure:
    """Quadrant scatter: X=S2 score, Y=S4 score.

    Each point is a model. Crosshairs at median S2 and median S4.
    Four quadrants are shaded and labeled with the corresponding
    signal-execution alignment pattern.

    Args:
        stage_scores: {model_key: {"S2": float, "S4": float, ...}}
        title: Figure title.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure.
    """
    apply_paper_style()

    # ── Extract (S2, S4) pairs ──────────────────────────────────────────
    points: list[dict] = []
    for model_key, scores in stage_scores.items():
        s2 = scores.get("S2", 0.0)
        s4 = scores.get("S4", 0.0)
        if s2 > 0 or s4 > 0:
            points.append({"model": model_key, "s2": s2, "s4": s4})

    if not points:
        raise ValueError("No S2/S4 data to plot.")

    s2_vals = [p["s2"] for p in points]
    s4_vals = [p["s4"] for p in points]
    med_s2 = float(np.median(s2_vals))
    med_s4 = float(np.median(s4_vals))

    # ── Assign per-model (colour, marker) ───────────────────────────────
    model_keys = sorted({p["model"] for p in points})
    model_meta: dict[str, dict] = {}
    for i, mk in enumerate(model_keys):
        model_meta[mk] = {
            "color": _MODEL_COLOURS[i % len(_MODEL_COLOURS)],
            "marker": _MODEL_MARKERS[i % len(_MODEL_MARKERS)],
        }

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    # ── Axis limits ────────────────────────────────────────────────────
    x_min, x_max = min(s2_vals), max(s2_vals)
    y_min, y_max = min(s4_vals), max(s4_vals)
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # ── Median crosshairs ───────────────────────────────────────────────
    ax.axhline(
        med_s4, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7, zorder=3
    )
    ax.axvline(
        med_s2, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7, zorder=3
    )

    # ── Scatter points ──────────────────────────────────────────────────
    for p in points:
        short = abbrev_model_name(p["model"])
        meta = model_meta[p["model"]]
        ax.scatter(
            p["s2"],
            p["s4"],
            c=meta["color"],
            marker=meta["marker"],
            s=120,
            alpha=0.9,
            linewidths=1.0,
            edgecolors="#333333",
            zorder=5,
        )
        ax.annotate(
            short,
            (p["s2"], p["s4"]),
            textcoords="offset points",
            xytext=(10, 0),
            fontsize=8,
            color="#222222",
            ha="left",
            va="center",
            zorder=6,
        )

    # ── Quadrant corner labels ──────────────────────────────────────────
    label_style = dict(fontsize=8.5, color="#444444", style="italic", zorder=7)
    pad_x = (x_max - x_min) * 0.03
    pad_y = (y_max - y_min) * 0.03

    ax.text(
        x_min - x_pad + pad_x,
        y_max + y_pad - pad_y,
        "Predicts well,\nexecutes poorly",
        ha="left",
        va="top",
        **label_style,
    )
    ax.text(
        x_max + x_pad - pad_x,
        y_max + y_pad - pad_y,
        "Balanced",
        ha="right",
        va="top",
        **label_style,
    )
    ax.text(
        x_min - x_pad + pad_x,
        y_min - y_pad + pad_y,
        "Weak on both",
        ha="left",
        va="bottom",
        **label_style,
    )
    ax.text(
        (med_s2 + x_max + x_pad) / 2,
        y_min - y_pad + pad_y,
        "Executes well,\npredicts poorly",
        ha="center",
        va="bottom",
        **label_style,
    )

    # ── Axis labels ─────────────────────────────────────────────────────
    ax.set_xlabel("S2 — Signal Generation Score", fontsize=10)
    ax.set_ylabel("S4 — Execution Simulation Score", fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.4, color="#aaaaaa")

    fig.tight_layout()
    return fig
