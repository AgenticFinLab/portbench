"""
Risk-return scatter plot for PortBench analysis.

Figure: plot_risk_return_scatter
  Conservative-profile only. Each model = unique colour + unique marker.
  Clean background (no tier bands), failed-gate points marked with a red ✗.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.figure import Figure

from .style import apply_paper_style, abbrev_model_name

# ── Per-model colour palette (high-contrast, no gradient) ────────────────────
_MODEL_COLOURS = [
    "#c0392b",  # deep red
    "#2471a3",  # deep water blue
    "#1a7a3c",  # forest green
    "#d35400",  # amber orange
    "#7d3c98",  # deep purple
    "#00838f",  # dark cyan
    "#8b4513",  # saddle brown
    "#e91e9e",  # magenta
    "#2c3e50",  # deep ink blue
    "#c0830b",  # dark goldenrod
    "#355e3b",  # hunter green
    "#b4464b",  # cranberry
]

# ── Per-model marker shape (guarantees unique shape+colour per model) ────────
_MODEL_MARKERS = [
    "o",  # circle
    "s",  # square
    "^",  # triangle up
    "D",  # diamond
    "*",  # five-pointed star
    "H",  # hexagon
    "v",  # triangle down
    "<",  # triangle left
    ">",  # triangle right
    "P",  # plus (thick)
    "X",  # X (thin cross — reserved for fail overlay, never assigned)
    "p",  # pentagon
    "h",  # hexagon (flat-top)
]


def plot_risk_return_scatter(
    rows: list[dict],
    title: str = "Conservative Profile: Model Performance by Stress Drawdown vs Sharpe Ratio",
    figsize: tuple = (5, 4),
) -> Figure:
    """Scatter plot: X=stress dd_score, Y=normal Sharpe, Conservative only.

    Each model gets a unique (colour, marker) pair.
    Failed-gate points are overlaid with a red ✗; passed points have no extra mark.
    No tier bands, clean white background with light grey grid.
    """
    apply_paper_style()

    # ── Aggregate stress dd_score: worst (min) per (model, profile) ─────────
    stress_dd_score: dict[tuple, float] = {}
    stress_passed: dict[tuple, bool] = {}
    for r in rows:
        if r.get("phase") != "stress":
            continue
        key = (r["model"], r["profile"])
        tol = r.get("tolerance", 0.2)
        dd = r.get("max_drawdown", 0.0)
        stored = r.get("dd_score")
        score = (
            float(stored)
            if (stored is not None and float(stored) > 0.0)
            else max(0.0, 1.0 - abs(dd) / max(tol, 1e-6))
        )
        if key not in stress_dd_score or score < stress_dd_score[key]:
            stress_dd_score[key] = score
        # Use overall profile-level gate (stress_gate_passed), not per-scenario passed
        overall_passed = r.get("stress_gate_passed", True)
        if key not in stress_passed:
            stress_passed[key] = overall_passed
        else:
            stress_passed[key] = stress_passed[key] and overall_passed

    # ── Collect data — Conservative profile only ────────────────────────────
    points: list[dict] = []
    for r in rows:
        if r.get("phase") != "normal":
            continue
        if r.get("profile") != "conservative":
            continue
        key = (r["model"], r["profile"])
        if key not in stress_dd_score:
            continue
        points.append(
            {
                "model": r["model"],
                "sharpe": r.get("sharpe_ratio", 0.0),
                "dd_score": stress_dd_score[key],
                "passed": stress_passed[key],
            }
        )

    if not points:
        raise ValueError("No Conservative (stress, normal) paired data to plot.")

    # ── LLM models only; baselines excluded from this scatter ───────────────
    points = [p for p in points if not p["model"].startswith("baseline/")]
    if not points:
        raise ValueError("No LLM Conservative data points found.")

    # ── Assign (colour, marker) per model ───────────────────────────────────
    model_keys = sorted({p["model"] for p in points})
    model_meta: dict[str, dict] = {}
    for i, mk in enumerate(model_keys):
        short = abbrev_model_name(mk)
        model_meta[short] = {
            "full_key": mk,
            "color": _MODEL_COLOURS[i % len(_MODEL_COLOURS)],
            "marker": _MODEL_MARKERS[i % len(_MODEL_MARKERS)],
        }

    fig, ax = plt.subplots(figsize=figsize)

    # White background, light-grey grid
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#cccccc")

    # Sharpe = 0 baseline
    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)

    # ── Plot points ──────────────────────────────────────────────────────────
    for p in points:
        short = abbrev_model_name(p["model"])
        meta = model_meta[short]
        ax.scatter(
            p["dd_score"],
            p["sharpe"],
            c=meta["color"],
            marker=meta["marker"],
            s=140,
            alpha=0.88,
            linewidths=1.2,
            edgecolors="#333333",
            zorder=4,
        )
        if not p["passed"]:
            ax.annotate(
                "✗",
                (p["dd_score"], p["sharpe"]),
                textcoords="offset points",
                xytext=(8, 0),
                fontsize=11,
                color="#e74c3c",
                fontweight="bold",
                ha="left",
                va="center",
                zorder=5,
            )

    # ── Axis limits & labels ─────────────────────────────────────────────────
    dd_scores = [p["dd_score"] for p in points]
    x_pad = 0.03
    ax.set_xlim(min(dd_scores) - x_pad, max(dd_scores) + x_pad)
    sharpes = [p["sharpe"] for p in points]
    y_min, y_max = min(sharpes), max(sharpes)
    y_pad = (y_max - y_min) * 0.22 if y_max != y_min else 0.3
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel("Stress Drawdown Score", fontsize=10)
    ax.set_ylabel("Normal Period Sharpe Ratio", fontsize=10)

    # ── Legend: one entry per model (color + shape) + red-X explanation ─────
    model_handles = [
        mlines.Line2D(
            [],
            [],
            color=meta["color"],
            marker=meta["marker"],
            linestyle="None",
            markersize=8,
            markerfacecolor=meta["color"],
            markeredgecolor="#333",
            markeredgewidth=0.8,
            label=short,
        )
        for short, meta in model_meta.items()
    ]
    fail_handle = mlines.Line2D(
        [],
        [],
        color="#e74c3c",
        marker="$✗$",
        linestyle="None",
        markersize=9,
        markeredgewidth=0,
        label="Failed gate",
    )
    spacer = mlines.Line2D([], [], color="none", label="")
    legend_handles = model_handles + [spacer, fail_handle]

    ncol = 2 if len(model_handles) > 5 else 1
    ax.legend(
        handles=legend_handles,
        fontsize=8.5,
        loc="lower right",
        framealpha=0.92,
        edgecolor="0.7",
        ncols=ncol,
        columnspacing=0.8,
        handletextpad=0.5,
        borderpad=0.5,
        labelspacing=0.4,
    )

    fig.tight_layout()
    return fig
