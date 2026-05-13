"""
Risk-return scatter plot for PortBench analysis.

Figure: plot_risk_return_scatter
  X: worst stress max_drawdown across all scenarios (more negative = higher risk)
  Y: normal period Sharpe ratio
  Color: stress_gate_passed (green = passed, red = failed)
  Marker: investor profile (o=conservative, s=balanced, ^=aggressive)

Each point is one (model, profile) pair. Models with high Sharpe but large
drawdown occupy the top-left quadrant — high return, low risk management.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.figure import Figure

from .style import apply_paper_style, PAPER_COLORS


_PROFILE_MARKERS = {
    "conservative": "o",
    "balanced":     "s",
    "aggressive":   "^",
}
_PASSED_COLOR  = "#2ecc71"
_FAILED_COLOR  = "#e74c3c"


def plot_risk_return_scatter(
    rows: list[dict],
    title: str = "Risk vs. Return — Stress Gate Analysis",
    figsize: tuple = (8, 6),
) -> Figure:
    """
    Scatter plot of (worst stress drawdown, normal Sharpe) per (model, profile).

    Args:
        rows: flattened rows from _flatten_rows(), containing both phase=="stress"
              and phase=="normal" entries.
        title:   Figure title.
        figsize: Figure size.

    Returns:
        matplotlib Figure, or raises ValueError if no plottable data.
    """
    apply_paper_style()

    # ── Aggregate stress risk: worst drawdown per (model, profile) ──────────
    stress_dd: dict[tuple, float] = {}  # (model, profile) → worst drawdown
    for r in rows:
        if r.get("phase") != "stress":
            continue
        key = (r["model"], r["profile"])
        dd = r.get("max_drawdown", 0.0)
        if key not in stress_dd or dd < stress_dd[key]:
            stress_dd[key] = dd

    # ── Collect normal performance per (model, profile) ─────────────────────
    points: list[dict] = []
    for r in rows:
        if r.get("phase") != "normal":
            continue
        key = (r["model"], r["profile"])
        if key not in stress_dd:
            continue
        points.append({
            "model":   r["model"],
            "profile": r["profile"],
            "sharpe":  r.get("sharpe_ratio", 0.0),
            "ret":     r.get("total_return", 0.0),
            "dd":      stress_dd[key],
            "passed":  r.get("stress_gate_passed", False),
        })

    if not points:
        raise ValueError("No (stress, normal) paired data to plot.")

    fig, ax = plt.subplots(figsize=figsize)

    # ── Draw quadrant guide lines ────────────────────────────────────────────
    all_dd = [p["dd"] for p in points]
    mid_dd = sum(all_dd) / len(all_dd)
    ax.axvline(mid_dd, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

    # Quadrant labels
    xlim_left = min(all_dd) * 1.15
    ax.text(xlim_left, ax.get_ylim()[1] if ax.get_ylim()[1] != ax.get_ylim()[0] else 1,
            "High risk\nHigh return", fontsize=7, color="gray",
            ha="left", va="top", style="italic")

    # ── Plot each point ──────────────────────────────────────────────────────
    label_set: set[str] = set()
    for p in points:
        color  = _PASSED_COLOR if p["passed"] else _FAILED_COLOR
        marker = _PROFILE_MARKERS.get(p["profile"], "o")
        ax.scatter(
            p["dd"], p["sharpe"],
            c=color, marker=marker,
            s=90, alpha=0.85, linewidths=0.5, edgecolors="white", zorder=3,
        )
        # Short model label (last component after /)
        short = p["model"].split("/")[-1]
        label_key = f"{short}_{p['profile']}"
        if label_key not in label_set:
            ax.annotate(
                short,
                xy=(p["dd"], p["sharpe"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="black", alpha=0.8,
            )
            label_set.add(label_key)

    # ── Axes labels & title ──────────────────────────────────────────────────
    ax.set_xlabel("Worst Stress Max Drawdown  (more negative = higher risk)", fontsize=9)
    ax.set_ylabel("Normal Period Sharpe Ratio", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=_PASSED_COLOR, label="Stress gate PASSED"),
        mpatches.Patch(color=_FAILED_COLOR, label="Stress gate FAILED"),
    ] + [
        mlines.Line2D([], [], color="gray", marker=m, linestyle="None",
                      markersize=7, label=prof.capitalize())
        for prof, m in _PROFILE_MARKERS.items()
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
              framealpha=0.85, edgecolor="0.8")

    fig.tight_layout()
    return fig
