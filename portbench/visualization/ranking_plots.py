"""
Risk-first ranking visualization.

Figure 5 — plot_risk_ranking: Horizontal bar chart sorted by mean CEPS.
    Models that failed the stress gate are grayed/hatched and labeled.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from .style import apply_paper_style, PAPER_COLORS, MODEL_PALETTE


def plot_risk_ranking(
    ranking_data: list[dict],
    title: str = "Risk-First Model Ranking",
    figsize: tuple = (7, 4),
) -> Figure:
    """
    Horizontal bar chart of mean CEPS, risk-gate status indicated by style.

    Args:
        ranking_data: list of dicts, each with keys:
            model_name, mean_ceps, std_ceps, risk_gate_passed
        title:   Figure title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    apply_paper_style()

    # Sort: passing models first (by CEPS desc), then failing (by CEPS desc)
    passed  = sorted([r for r in ranking_data if r.get("risk_gate_passed", True)],
                     key=lambda x: x["mean_ceps"], reverse=True)
    failed  = sorted([r for r in ranking_data if not r.get("risk_gate_passed", True)],
                     key=lambda x: x["mean_ceps"], reverse=True)
    ordered = passed + failed

    model_names = [r["model_name"] for r in ordered]
    scores      = [r["mean_ceps"]   for r in ordered]
    stds        = [r.get("std_ceps", 0.0) for r in ordered]
    is_passed   = [r.get("risk_gate_passed", True) for r in ordered]

    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, score, std, passed) in enumerate(zip(model_names, scores, stds, is_passed)):
        color  = MODEL_PALETTE[i % len(MODEL_PALETTE)] if passed else PAPER_COLORS["neutral"]
        hatch  = None if passed else "xxx"
        bar = ax.barh(i, score, xerr=std, height=0.6,
                      color=color, hatch=hatch, alpha=0.85,
                      edgecolor="white", capsize=4)
        # Annotate score
        ax.text(score + std + 0.01, i, f"{score:.3f}",
                va="center", ha="left", fontsize=8)
        # Label failed models
        if not passed:
            ax.text(0.01, i, "RISK GATE FAILED", va="center", ha="left",
                    fontsize=7, color="white", fontweight="bold")

    # Separator line between passed and failed
    if failed and passed:
        sep_y = len(passed) - 0.5
        ax.axhline(sep_y, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.text(0.5, sep_y + 0.1, "── Risk Gate Passed ──", ha="center",
                fontsize=7, color="gray", style="italic")

    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_xlabel("Mean CEPS Score")
    ax.set_xlim(0, 1.15)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()  # highest score at top

    legend_handles = [
        mpatches.Patch(color=MODEL_PALETTE[0], label="Passed risk gate"),
        mpatches.Patch(color=PAPER_COLORS["neutral"], hatch="xxx",
                       edgecolor="white", label="Failed risk gate"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    return fig
