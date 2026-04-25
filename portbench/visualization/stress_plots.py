"""
Stress test visualization: risk gate bar chart showing pass/fail per scenario.

Figure 4 — plot_stress_gate: Grouped bars per scenario, models as groups,
    red dashed threshold line, failed bars hatched.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from .style import apply_paper_style, PAPER_COLORS, MODEL_PALETTE


def plot_stress_gate(
    stress_data: dict[str, list[dict]],
    title: str = "Stress Test Risk Gate",
    figsize: tuple = (8, 4.5),
) -> Figure:
    """
    Grouped bar chart: CEPS score per (scenario, model).
    Bars below the pass threshold are shown in red with hatch.

    Args:
        stress_data: {model_name: [stress_result_dict, ...]}
            Each stress_result_dict has keys:
                scenario_name, mean_ceps, min_pass_score, passed
        title:   Figure title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    apply_paper_style()

    model_names = list(stress_data.keys())
    n_models = len(model_names)

    # Collect scenario names (preserve order from first model)
    scenario_names = [r["scenario_name"] for r in next(iter(stress_data.values()))]
    n_scenarios = len(scenario_names)

    # Build data matrix: shape (n_models, n_scenarios)
    ceps_matrix = np.zeros((n_models, n_scenarios))
    pass_matrix = np.ones((n_models, n_scenarios), dtype=bool)
    threshold = 0.5  # default; read from first result if available
    for i, model in enumerate(model_names):
        for j, sr in enumerate(stress_data[model]):
            ceps_matrix[i, j] = sr.get("mean_ceps", 0.0)
            pass_matrix[i, j] = sr.get("passed", True)
            threshold = sr.get("min_pass_score", threshold)

    fig, ax = plt.subplots(figsize=figsize)

    bar_width = 0.7 / n_models
    scenario_positions = np.arange(n_scenarios)

    for i, model in enumerate(model_names):
        offsets = scenario_positions + (i - n_models / 2 + 0.5) * bar_width
        for j in range(n_scenarios):
            score = ceps_matrix[i, j]
            passed = pass_matrix[i, j]
            color = (
                MODEL_PALETTE[i % len(MODEL_PALETTE)]
                if passed
                else PAPER_COLORS["failed"]
            )
            hatch = None if passed else "//"
            bar = ax.bar(
                offsets[j],
                score,
                width=bar_width * 0.9,
                color=color,
                hatch=hatch,
                alpha=0.85,
                edgecolor="white",
            )

    # Pass threshold line
    ax.axhline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Pass threshold ({threshold:.2f})",
        zorder=5,
    )

    ax.set_xticks(scenario_positions)
    ax.set_xticklabels(scenario_names, fontsize=10)
    ax.set_ylabel("Mean CEPS Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Legend: model colors + failed pattern
    handles = [
        mpatches.Patch(color=MODEL_PALETTE[i % len(MODEL_PALETTE)], label=m)
        for i, m in enumerate(model_names)
    ]
    handles.append(
        mpatches.Patch(
            facecolor=PAPER_COLORS["failed"],
            hatch="//",
            edgecolor="white",
            label="Failed",
        )
    )
    handles.append(
        plt.Line2D(
            [0], [0], color="red", linestyle="--", label=f"Threshold ({threshold:.2f})"
        )
    )
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    return fig
