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

from .style import apply_paper_style, MODEL_PALETTE, abbrev_model_name


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

    model_keys = list(stress_data.keys())
    model_names = [abbrev_model_name(k) for k in model_keys]
    n_models = len(model_names)

    # Collect scenario names (preserve order from first model)
    scenario_names = [r["scenario_name"] for r in next(iter(stress_data.values()))]
    n_scenarios = len(scenario_names)

    # Build data matrix: shape (n_models, n_scenarios)
    ceps_matrix = np.zeros((n_models, n_scenarios))
    pass_matrix = np.ones((n_models, n_scenarios), dtype=bool)
    threshold = 0.5  # default; read from first result if available
    for i, key in enumerate(model_keys):
        for j, sr in enumerate(stress_data[key]):
            ceps_matrix[i, j] = sr.get("mean_ceps", 0.0)
            pass_matrix[i, j] = sr.get("passed", True)
            threshold = sr.get("min_pass_score", threshold)

    fig, ax = plt.subplots(figsize=figsize)

    # Hatch patterns cycle to visually distinguish models even when colors are similar
    _HATCHES = ['', '//', '\\\\', 'xx', '..', '++', '--', '||', '//', 'oo']

    bar_width = 0.7 / n_models
    scenario_positions = np.arange(n_scenarios)

    for i, model in enumerate(model_names):
        color      = MODEL_PALETTE[i % len(MODEL_PALETTE)]
        hatch_base = _HATCHES[i % len(_HATCHES)] if i >= len(MODEL_PALETTE) else ""
        offsets    = scenario_positions + (i - n_models / 2 + 0.5) * bar_width
        for j in range(n_scenarios):
            score  = ceps_matrix[i, j]
            passed = pass_matrix[i, j]
            ax.bar(
                offsets[j], score,
                width=bar_width * 0.9,
                color=color, hatch=hatch_base,
                alpha=0.85, edgecolor="gray",
            )
            if not passed:
                # Red "!" centred at the top of the bar
                ax.text(offsets[j], score, "!",
                        ha="center", va="bottom",
                        fontsize=10, color="red", fontweight="bold", zorder=6)

    # Tolerance threshold line
    ax.axhline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Drawdown tolerance ({threshold:.0%})",
        zorder=5,
    )

    ax.set_xticks(scenario_positions)
    ax.set_xticklabels(scenario_names, fontsize=10)
    ax.set_ylabel("Max Drawdown (worst profile)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, min(1.0, max(ceps_matrix.max() * 1.3, threshold * 1.5)))

    # Legend: model color + hatch, failed indicator, threshold line
    handles = [
        mpatches.Patch(
            facecolor=MODEL_PALETTE[i % len(MODEL_PALETTE)],
            hatch=_HATCHES[i % len(_HATCHES)] if i >= len(MODEL_PALETTE) else "",
            edgecolor="gray", label=m,
        )
        for i, m in enumerate(model_names)
    ]
    handles.append(
        plt.Line2D([0], [0], color="red", linestyle="--",
                   label=f"Tolerance ({threshold:.0%})")
    )
    handles.append(
        plt.Line2D([], [], marker="$!$", color="red", linestyle="None",
                   markersize=9, label="! Failed")
    )
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              ncols=max(1, len(handles) // 4))

    fig.tight_layout()
    return fig
