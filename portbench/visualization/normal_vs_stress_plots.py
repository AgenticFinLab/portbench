"""
Normal-vs-Stress CEPS scatter plot for PortBench analysis.

Figure: plot_normal_vs_stress_scatter
  X = CEPS_normal, Y = CEPS_stress (2022 Crypto Collapse).
  y = x diagonal separates models that degrade vs improve under stress.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
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
    Same axis ranges ensure 45° diagonal.
    """
    apply_paper_style()

    if not points:
        raise ValueError("No data points to plot.")

    lo, hi = 0.30, 0.52
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

        ax.scatter(
            x,
            y,
            c=meta["color"],
            marker=meta["marker"],
            s=120,
            alpha=0.9,
            linewidths=1.0,
            edgecolors="#333333",
            zorder=5,
        )

        # Qwen models: label on the left; others: label on the right
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

        # Red ✗ for gate fail
        if not p.get("stress_gate_passed", True):
            ax.annotate(
                "✗",
                (x, y),
                textcoords="offset points",
                xytext=(0, 9),
                fontsize=11,
                color="#e74c3c",
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=7,
            )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("CEPS — Normal Period (Conservative)", fontsize=10)
    ax.set_ylabel("CEPS — 2022 Crypto Collapse (Conservative)", fontsize=10)
    ax.set_aspect("equal")

    ax.text(
        0.505,
        0.498,
        "y = x",
        fontsize=8,
        color="#777777",
        rotation=38,
        ha="left",
        va="bottom",
    )

    fig.tight_layout()
    return fig
