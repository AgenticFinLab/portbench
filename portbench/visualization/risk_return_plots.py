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
import matplotlib.lines as mlines
from matplotlib.figure import Figure

from .style import apply_paper_style, PAPER_COLORS, CATEGORICAL_PALETTE, abbrev_model_name


_PROFILE_MARKERS = {
    "conservative": "o",
    "balanced":     "s",
    "aggressive":   "^",
}


def plot_risk_return_scatter(
    rows: list[dict],
    title: str = "Risk vs. Return — Stress Gate Analysis",
    figsize: tuple = (8, 6),
) -> Figure:
    """
    Scatter plot of (worst stress drawdown, normal Sharpe) per (model, profile).
    Color = model, shape = investor profile.
    Pass/fail indicated by gold star (★) / red exclamation (!) overlay.
    """
    apply_paper_style()

    # ── Aggregate stress risk: worst drawdown per (model, profile) ──────────
    stress_dd: dict[tuple, float] = {}
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
            "dd":      stress_dd[key],
            "passed":  r.get("stress_gate_passed", False),
        })

    if not points:
        raise ValueError("No (stress, normal) paired data to plot.")

    # ── Assign one color per model ───────────────────────────────────────────
    model_order = sorted({abbrev_model_name(p["model"]) for p in points})
    model_color = {m: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
                   for i, m in enumerate(model_order)}

    fig, ax = plt.subplots(figsize=figsize)

    # ── Plot scatter points (color = model, shape = profile) ─────────────────
    for p in points:
        short  = abbrev_model_name(p["model"])
        color  = model_color[short]
        marker = _PROFILE_MARKERS.get(p["profile"], "o")
        ax.scatter(p["dd"], p["sharpe"],
                   c=color, marker=marker,
                   s=130, alpha=0.85, linewidths=0.5, edgecolors="white", zorder=3)

    # ── Pass/fail overlays ────────────────────────────────────────────────────
    for p in points:
        if p["passed"]:
            # Gold five-pointed star slightly above the point
            ax.scatter(p["dd"], p["sharpe"],
                       marker="*", color="#FFD700", s=55,
                       linewidths=0.4, edgecolors="#B8860B", zorder=5)
        else:
            # Red exclamation mark centred on the point
            ax.text(p["dd"], p["sharpe"], "!",
                    fontsize=9, color="red", fontweight="bold",
                    ha="center", va="center", zorder=5)

    # ── Axis limits ──────────────────────────────────────────────────────────
    all_dd     = [p["dd"]     for p in points]
    all_sharpe = [p["sharpe"] for p in points]
    x_min, x_max = min(all_dd),     max(all_dd)
    y_min, y_max = min(all_sharpe), max(all_sharpe)
    x_pad = (x_max - x_min) * 0.25 if x_max != x_min else 0.05
    y_pad = (y_max - y_min) * 0.25 if y_max != y_min else 0.5
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # ── Quadrant guide lines ──────────────────────────────────────────────────
    mid_dd = sum(all_dd) / len(all_dd)
    ax.axvline(mid_dd, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.axhline(0,      color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.text(x_min - x_pad * 0.9, y_max + y_pad * 0.85,
            "← High risk / High return",
            fontsize=7, color="gray", ha="left", va="top", style="italic")

    # ── Axes labels ───────────────────────────────────────────────────────────
    ax.set_xlabel("Worst Stress Max Drawdown  (more negative = higher risk)", fontsize=9)
    ax.set_ylabel("Normal Period Sharpe Ratio", fontsize=9)

    # ── Legend ────────────────────────────────────────────────────────────────
    model_handles = [
        mlines.Line2D([], [], color=model_color[m], marker="o", linestyle="None",
                      markersize=7, label=m)
        for m in model_order
    ]
    profile_handles = [
        mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                      markersize=7, label=prof.capitalize())
        for prof, mk in _PROFILE_MARKERS.items()
    ]
    indicator_handles = [
        mlines.Line2D([], [], color="#FFD700", marker="*", linestyle="None",
                      markersize=9, markeredgecolor="#B8860B", label="Passed"),
        mlines.Line2D([], [], color="red", marker="$!$", linestyle="None",
                      markersize=8, label="Failed"),
    ]
    legend_handles = model_handles + profile_handles + indicator_handles
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              framealpha=0.85, edgecolor="0.8",
              ncols=max(1, len(legend_handles) // 5))

    fig.tight_layout()
    return fig
