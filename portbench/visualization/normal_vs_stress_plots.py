"""
Normal-vs-Stress CEPS scatter plot for PortBench analysis.

Figure: plot_normal_vs_stress_scatter
  X = CEPS_normal, Y = CEPS_stress (2022 Crypto Collapse).
  y = x diagonal separates models that degrade vs improve under stress.
  Gate failures are encoded as a thick red marker edge (no floating ✗).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.figure import Figure

from .style import apply_paper_style, abbrev_model_name
from .risk_return_plots import _MODEL_COLOURS, _MODEL_MARKERS

_QWEN_LEFT = {"Qwen3.6-35b", "Qwen3.6-Plus", "Qwen3.7-Max"}


def plot_normal_vs_stress_scatter(
    points: list[dict],
    title: str = "Normal vs Stress CEPS — Conservative Profile",
    figsize: tuple = (5, 5),
) -> Figure:
    """Scatter plot: X=CEPS_normal, Y=CEPS_stress (2022 Crypto), conservative profile.

    y=x diagonal divides models that improve (above) vs degrade (below) under stress.
    Same axis ranges ensure 45° diagonal. Failed stress-gate models use a thick
    red edge on the marker instead of a separate ✗ annotation.
    """
    apply_paper_style()

    if not points:
        raise ValueError("No data points to plot.")

    xs = [float(p["ceps_normal"]) for p in points]
    ys = [float(p["ceps_crypto"]) for p in points]
    pad = 0.03
    lo = max(0.0, min(min(xs), min(ys)) - pad)
    hi = min(1.0, max(max(xs), max(ys)) + pad)
    # Keep a readable lower floor so the panel is not overly sparse.
    lo = min(lo, 0.15)
    hi = max(hi, 0.55)
    diag = np.linspace(lo, hi, 50)

    model_keys = sorted({p["model"] for p in points})
    model_meta: dict[str, dict] = {}
    for i, mk in enumerate(model_keys):
        model_meta[mk] = {
            "color": _MODEL_COLOURS[i % len(_MODEL_COLOURS)],
            "marker": _MODEL_MARKERS[i % len(_MODEL_MARKERS)],
        }

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.4, color="#aaaaaa")

    ax.plot(
        diag, diag, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2
    )

    for p in points:
        short = abbrev_model_name(p["model"])
        meta = model_meta[p["model"]]
        x, y = p["ceps_normal"], p["ceps_crypto"]
        failed = not p.get("stress_gate_passed", True)

        # Failed gate: outer red halo (no floating ✗; works for any fill color).
        if failed:
            ax.scatter(
                x,
                y,
                facecolors="none",
                marker="o",
                s=280,
                linewidths=2.0,
                edgecolors="#e74c3c",
                zorder=4,
            )

        ax.scatter(
            x,
            y,
            c=meta["color"],
            marker=meta["marker"],
            s=120,
            alpha=0.92,
            linewidths=1.0,
            edgecolors="#333333",
            zorder=5,
        )

        # Qwen models: label on the left; others: label on the right.
        if short in _QWEN_LEFT:
            ax.annotate(
                short,
                (x, y),
                textcoords="offset points",
                xytext=(-10, 0),
                fontsize=8,
                color="#222222",
                ha="right",
                va="center",
                zorder=6,
            )
        else:
            ax.annotate(
                short,
                (x, y),
                textcoords="offset points",
                xytext=(10, 0),
                fontsize=8,
                color="#222222",
                ha="left",
                va="center",
                zorder=6,
            )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("CEPS — Normal Period (Conservative)", fontsize=10)
    ax.set_ylabel("CEPS — 2022 Crypto Collapse (Conservative)", fontsize=10)
    ax.set_aspect("equal")

    ax.text(
        lo + 0.62 * (hi - lo),
        lo + 0.58 * (hi - lo),
        "y = x",
        fontsize=8,
        color="#777777",
        rotation=38,
        ha="left",
        va="bottom",
    )

    pass_handle = mlines.Line2D(
        [],
        [],
        color="#888888",
        marker="o",
        linestyle="None",
        markersize=8,
        markerfacecolor="#bbbbbb",
        markeredgecolor="#333333",
        markeredgewidth=1.0,
        label="Passed gate",
    )
    fail_handle = mlines.Line2D(
        [],
        [],
        color="#e74c3c",
        marker="o",
        linestyle="None",
        markersize=11,
        markerfacecolor="none",
        markeredgecolor="#e74c3c",
        markeredgewidth=2.0,
        label="Failed gate",
    )
    ax.legend(
        handles=[pass_handle, fail_handle],
        fontsize=8,
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        framealpha=0.95,
    )

    fig.tight_layout()
    return fig
