"""
Investor profile evaluation visualization.

Figure 11 — plot_profile_alignment:  Grouped bars showing per-profile alignment
    scores, one bar group per investor profile, one bar per model.

Figure 12 — plot_profile_radar:  Triangle radar chart (3 axes = 3 profiles),
    one polygon per model, showing per-profile alignment score.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from .style import apply_paper_style, PAPER_COLORS, MODEL_PALETTE

_PROFILE_LABELS = ["Conservative", "Balanced", "Aggressive"]
_PROFILE_KEYS = ["conservative", "balanced", "aggressive"]


# ---------------------------------------------------------------------------
# Figure 11 — Profile Alignment Bar Chart
# ---------------------------------------------------------------------------


def plot_profile_alignment(
    profile_data: dict[str, dict[str, float]],
    title: str = "Investor Profile Alignment Score",
    figsize: tuple = (8, 4.5),
) -> Figure:
    """
    Grouped bar chart: alignment score per (investor profile, model).

    Args:
        profile_data: {model_name: {"conservative": float, "balanced": float,
                                    "aggressive": float}}
            Values are mean profile alignment scores in [0, 1].
            Optionally, each inner dict may also contain
            "{profile}_std" keys for error bars.
        title:   Figure title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    apply_paper_style()

    model_names = list(profile_data.keys())
    n_models = len(model_names)
    n_profiles = len(_PROFILE_KEYS)

    score_matrix = np.zeros((n_models, n_profiles))
    std_matrix = np.zeros((n_models, n_profiles))
    for i, model in enumerate(model_names):
        for j, key in enumerate(_PROFILE_KEYS):
            score_matrix[i, j] = profile_data[model].get(key, 0.0)
            std_matrix[i, j] = profile_data[model].get(f"{key}_std", 0.0)

    fig, ax = plt.subplots(figsize=figsize)

    bar_width = 0.7 / n_models
    profile_positions = np.arange(n_profiles)

    legend_handles = []
    for i, model in enumerate(model_names):
        offsets = profile_positions + (i - n_models / 2 + 0.5) * bar_width
        color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
        bars = ax.bar(
            offsets,
            score_matrix[i],
            width=bar_width * 0.9,
            color=color,
            alpha=0.85,
            yerr=std_matrix[i] if std_matrix[i].any() else None,
            capsize=3,
            error_kw={"elinewidth": 0.8, "ecolor": PAPER_COLORS["neutral"]},
        )
        legend_handles.append(mpatches.Patch(color=color, label=model))

    ax.set_xticks(profile_positions)
    ax.set_xticklabels(_PROFILE_LABELS, fontsize=10)
    ax.set_ylabel("Profile Alignment Score", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(
        0.5,
        color=PAPER_COLORS["failed"],
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Threshold (0.5)",
    )
    from matplotlib.lines import Line2D

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=PAPER_COLORS["failed"],
            linestyle="--",
            linewidth=1,
            label="Threshold (0.5)",
        )
    )
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    ax.set_title(title, fontsize=11, fontweight="bold")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 12 — Profile Radar Chart
# ---------------------------------------------------------------------------


def plot_profile_radar(
    profile_data: dict[str, dict[str, float]],
    title: str = "Profile Adaptation Radar",
    figsize: tuple = (5, 5),
) -> Figure:
    """
    Triangle radar chart with one axis per investor profile.

    Args:
        profile_data: {model_name: {"conservative": float, "balanced": float,
                                    "aggressive": float}}
        title:   Figure title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    apply_paper_style()

    n = len(_PROFILE_KEYS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(_PROFILE_LABELS, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7)

    legend_handles = []
    for i, (model, scores) in enumerate(profile_data.items()):
        values = [scores.get(k, 0.0) for k in _PROFILE_KEYS] + [
            scores.get(_PROFILE_KEYS[0], 0.0)
        ]
        color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.12)
        legend_handles.append(mpatches.Patch(color=color, label=model))

    ax.set_title(title, fontsize=11, fontweight="bold", pad=18)
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.25, 1.1),
        fontsize=8,
    )

    return fig
