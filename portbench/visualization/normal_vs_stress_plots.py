"""
Normal-vs-Stress CEPS scatter plot for PortBench analysis.

Pass = filled marker; fail = open marker.
Model identity is a color-matched legend (no on-plot text overlapping points).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.figure import Figure

from .style import apply_paper_style, abbrev_model_name
from .risk_return_plots import _MODEL_COLOURS, _MODEL_MARKERS


def plot_normal_vs_stress_scatter(
    points: list[dict],
    title: str = "Normal vs Stress CEPS — Conservative Profile",
    figsize: tuple = (3.5, 3.5),
) -> Figure:
    """Scatter: X=CEPS_normal, Y=CEPS_stress (2022 Crypto), conservative profile."""
    apply_paper_style()

    if not points:
        raise ValueError("No data points to plot.")

    xs = [float(p["ceps_normal"]) for p in points]
    ys = [float(p["ceps_crypto"]) for p in points]
    pad = 0.03
    lo = max(0.0, min(min(xs), min(ys)) - pad)
    hi = min(1.0, max(max(xs), max(ys)) + pad)
    lo = min(lo, 0.15)
    hi = max(hi, 0.55)
    diag = np.linspace(lo, hi, 50)

    # Stable model order for legend (alphabetical by display name)
    model_keys = sorted({p["model"] for p in points}, key=abbrev_model_name)
    model_meta: dict[str, dict] = {}
    for i, mk in enumerate(model_keys):
        model_meta[mk] = {
            "color": _MODEL_COLOURS[i % len(_MODEL_COLOURS)],
            "marker": _MODEL_MARKERS[i % len(_MODEL_MARKERS)],
            "short": abbrev_model_name(mk),
        }

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.35, color="#b0c0d0")

    ax.fill_between(diag, diag - 0.02, diag + 0.02, color="#dfe6ee", alpha=0.45, zorder=1)
    ax.plot(diag, diag, color="#666666", linestyle="--", linewidth=1.0, alpha=0.85, zorder=2)

    point_by_model = {p["model"]: p for p in points}
    for mk in model_keys:
        p = point_by_model[mk]
        meta = model_meta[mk]
        x, y = float(p["ceps_normal"]), float(p["ceps_crypto"])
        passed = bool(p.get("stress_gate_passed", True))
        if passed:
            ax.scatter(
                x, y, c=meta["color"], marker=meta["marker"], s=95,
                alpha=0.95, linewidths=0.9, edgecolors="#222222", zorder=5,
            )
        else:
            ax.scatter(
                x, y, facecolors="none", marker=meta["marker"], s=95,
                linewidths=1.8, edgecolors=meta["color"], zorder=5,
            )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("CEPS — Normal (Conservative)", fontsize=9)
    ax.set_ylabel("CEPS — 2022 Crypto (Conservative)", fontsize=9)
    ax.set_aspect("equal")

    ax.text(
        lo + 0.68 * (hi - lo),
        lo + 0.62 * (hi - lo),
        "y = x",
        fontsize=7.5,
        color="#666666",
        rotation=40,
        ha="left",
        va="bottom",
    )

    # Single combined legend (lower-right): gate encoding + model markers
    pass_handle = mlines.Line2D(
        [], [], color="#555555", marker="o", linestyle="None",
        markersize=6.5, markerfacecolor="#888888", markeredgecolor="#222222",
        label="Passed gate",
    )
    fail_handle = mlines.Line2D(
        [], [], color="#555555", marker="o", linestyle="None",
        markersize=6.5, markerfacecolor="none", markeredgecolor="#555555",
        markeredgewidth=1.5, label="Failed gate",
    )
    model_handles = []
    for mk in model_keys:
        meta = model_meta[mk]
        p = point_by_model[mk]
        passed = bool(p.get("stress_gate_passed", True))
        model_handles.append(
            mlines.Line2D(
                [], [],
                color=meta["color"],
                marker=meta["marker"],
                linestyle="None",
                markersize=6,
                markerfacecolor=meta["color"] if passed else "none",
                markeredgecolor=meta["color"] if not passed else "#222222",
                markeredgewidth=1.1,
                label=meta["short"],
            )
        )

    ax.legend(
        handles=[pass_handle, fail_handle] + model_handles,
        fontsize=6,
        loc="lower right",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        framealpha=0.95,
        borderpad=0.35,
        labelspacing=0.28,
        columnspacing=0.8,
        handletextpad=0.35,
    )

    fig.tight_layout()
    return fig
