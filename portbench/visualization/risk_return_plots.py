"""
Risk-return scatter plot for PortBench analysis.

Figure: plot_risk_return_scatter
  X: worst stress dd_score across all scenarios (0=full loss, 1=no loss),
     normalized by per-profile tolerance — comparable across profiles
  Y: normal period Sharpe ratio
  Color: model (one color per model via NAV_LLM_PALETTE)
  Marker: investor profile (o=conservative, s=balanced, ^=aggressive)
  Edge color: green=passed stress gate, red=failed

Each point is one (model, profile) pair. Top-right quadrant = high Sharpe + strong
stress resilience; bottom-left = worst performers on both dimensions.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from .style import apply_paper_style, PAPER_COLORS, NAV_LLM_PALETTE, NAV_BASELINE_PALETTE, abbrev_model_name


_PROFILE_MARKERS = {
    "conservative": "o",
    "balanced":     "s",
    "aggressive":   "^",
}

_TIER_THRESHOLDS = [0.1, 0.4, 0.6, 0.8]
_TIER_LABELS     = ["D", "C", "B", "A"]
_TIER_COLORS     = ["#fde8e8", "#fde8c8", "#fdfbe8", "#e8fde8"]


def plot_risk_return_scatter(
    rows: list[dict],
    title: str = "Risk vs. Return — Stress Gate Analysis",
    figsize: tuple = (8, 6),
) -> Figure:
    """
    Scatter plot of (worst stress dd_score, normal Sharpe) per (model, profile).

    X-axis: worst dd_score across scenarios — profile-normalized, [0,1], higher=better.
    Y-axis: normal-period Sharpe ratio.
    Color: model identity.  Marker: investor profile.
    Edge color: green border = passed stress gate, red = failed.
    Vertical bands mark tier thresholds D/C/B/A (matches stress.png).

    Small reproducible jitter is applied on X to separate overlapping points.
    """
    apply_paper_style()

    # ── Aggregate stress dd_score: worst (min) per (model, profile) ─────────
    stress_dd_score: dict[tuple, float] = {}
    stress_passed:   dict[tuple, bool]  = {}
    for r in rows:
        if r.get("phase") != "stress":
            continue
        key = (r["model"], r["profile"])
        tol = r.get("tolerance", 0.2)
        dd  = r.get("max_drawdown", 0.0)
        stored = r.get("dd_score")
        score = (
            float(stored) if (stored is not None and float(stored) > 0.0)
            else max(0.0, 1.0 - abs(dd) / max(tol, 1e-6))
        )
        if key not in stress_dd_score or score < stress_dd_score[key]:
            stress_dd_score[key] = score
            stress_passed[key]   = r.get("passed", False)

    # ── Collect normal performance per (model, profile) ─────────────────────
    points: list[dict] = []
    for r in rows:
        if r.get("phase") != "normal":
            continue
        key = (r["model"], r["profile"])
        if key not in stress_dd_score:
            continue
        points.append({
            "model":    r["model"],
            "profile":  r["profile"],
            "sharpe":   r.get("sharpe_ratio", 0.0),
            "dd_score": stress_dd_score[key],
            "passed":   stress_passed[key],
        })

    if not points:
        raise ValueError("No (stress, normal) paired data to plot.")

    # ── Assign colors ────────────────────────────────────────────────────────
    all_model_keys = sorted({p["model"] for p in points})
    llm_keys  = [m for m in all_model_keys if not m.startswith("baseline/")]
    base_keys = [m for m in all_model_keys if     m.startswith("baseline/")]
    model_color: dict[str, str] = {}
    for i, m in enumerate(llm_keys):
        model_color[abbrev_model_name(m)] = NAV_LLM_PALETTE[i % len(NAV_LLM_PALETTE)]
    for i, m in enumerate(base_keys):
        model_color[abbrev_model_name(m)] = NAV_BASELINE_PALETTE[i % len(NAV_BASELINE_PALETTE)]
    model_order = [abbrev_model_name(m) for m in llm_keys + base_keys]

    # ── Small reproducible jitter on X to separate overlapping points ────────
    rng = np.random.default_rng(42)
    jitter_x = rng.uniform(-0.012, 0.012, len(points))
    jitter_y = rng.uniform(-0.02, 0.02, len(points))

    fig, ax = plt.subplots(figsize=figsize)

    # ── Tier background bands (vertical) ─────────────────────────────────────
    band_bounds = [0.0] + _TIER_THRESHOLDS + [1.0]
    for bi in range(len(band_bounds) - 1):
        ax.axvspan(band_bounds[bi], band_bounds[bi + 1],
                   color=_TIER_COLORS[min(bi, len(_TIER_COLORS) - 1)],
                   alpha=0.30, zorder=0)

    for thresh, label in zip(_TIER_THRESHOLDS, _TIER_LABELS):
        ax.axvline(thresh, color="#aab0b8", linestyle="--", linewidth=0.7, zorder=1)
        ax.text(thresh + 0.005, 1.0, f"Tier {label}",
                fontsize=7, color="#555", va="top", ha="left",
                transform=ax.get_xaxis_transform(), zorder=5)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # ── Plot scatter points ───────────────────────────────────────────────────
    for idx, p in enumerate(points):
        short      = abbrev_model_name(p["model"])
        face_color = model_color[short]
        edge_color = "#27ae60" if p["passed"] else "#e74c3c"
        marker     = _PROFILE_MARKERS.get(p["profile"], "o")
        ax.scatter(
            p["dd_score"] + jitter_x[idx],
            p["sharpe"]   + jitter_y[idx],
            c=face_color, marker=marker,
            s=160, alpha=0.85,
            linewidths=2.0, edgecolors=edge_color,
            zorder=3,
        )

    # ── Axis limits & labels ──────────────────────────────────────────────────
    ax.set_xlim(-0.02, 1.02)
    all_sharpe = [p["sharpe"] for p in points]
    y_min, y_max = min(all_sharpe), max(all_sharpe)
    y_pad = (y_max - y_min) * 0.20 if y_max != y_min else 0.5
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel("Stress Drawdown Score  (↑ better)", fontsize=9)
    ax.set_ylabel("Normal Period Sharpe Ratio", fontsize=9)

    # ── Two separate legends: models (top-right), profile+gate (bottom-right) ─
    model_handles = [
        mlines.Line2D([], [], color=model_color[m], marker="o", linestyle="None",
                      markersize=7, label=m)
        for m in model_order
    ]
    profile_gate_handles = [
        mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                      markersize=7, label=prof.capitalize())
        for prof, mk in _PROFILE_MARKERS.items()
    ] + [
        mlines.Line2D([], [], color="#27ae60", marker="s", linestyle="None",
                      markersize=8, markerfacecolor="none", markeredgewidth=2,
                      label="Passed gate"),
        mlines.Line2D([], [], color="#e74c3c", marker="s", linestyle="None",
                      markersize=8, markerfacecolor="none", markeredgewidth=2,
                      label="Failed gate"),
    ]

    leg1 = ax.legend(handles=model_handles, fontsize=8, loc="upper right",
                     framealpha=0.9, edgecolor="0.8", title="Model", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=profile_gate_handles, fontsize=8, loc="lower right",
              framealpha=0.9, edgecolor="0.8", title="Profile / Gate", title_fontsize=8)

    fig.tight_layout()
    return fig
