"""
CEPS visualization: radar chart, error propagation heatmap, violin plot.

Figure 1 — plot_ceps_radar:   Per-stage capability profile per model (polar axes)
Figure 2 — plot_ceps_heatmap: Stage×Model score heatmap showing error propagation
Figure 3 — plot_ceps_violin:  Per-episode CEPS distribution per model
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from .style import (
    apply_paper_style,
    STAGE_LABELS,
    STAGE_IDS,
    MODEL_PALETTE,
    LINE_STYLES,
    LINE_MARKERS,
    abbrev_model_name,
)


# ---------------------------------------------------------------------------
# Figure 1 — CEPS Radar Chart
# ---------------------------------------------------------------------------


def plot_ceps_radar(
    results: dict[str, dict[str, float]],
    title: str = "Per-Stage Capability Profile",
    figsize: tuple = (5, 5),
) -> Figure:
    """
    Pentagon radar chart, one polygon per model.

    Args:
        results: {model_name: {"S1": float, ..., "S5": float}}
        title:   Figure title.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure.
    """
    apply_paper_style()

    n_stages = len(STAGE_IDS)
    angles = np.linspace(0, 2 * np.pi, n_stages, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})

    # Draw stage grid lines
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(STAGE_LABELS, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7)
    ax.yaxis.set_tick_params(labelsize=7)

    legend_handles = []
    for i, (model_name, scores) in enumerate(results.items()):
        values = [scores.get(sid, 0.0) for sid in STAGE_IDS]
        values += values[:1]
        color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
        ls = LINE_STYLES[i % len(LINE_STYLES)]
        mk = LINE_MARKERS[i % len(LINE_MARKERS)]
        ax.plot(angles, values, color=color, linewidth=2, linestyle=ls,
                marker=mk, markersize=5)
        ax.fill(angles, values, color=color, alpha=0.12)
        legend_handles.append(mpatches.Patch(color=color, label=abbrev_model_name(model_name)))

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=8,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — CEPS Error Propagation Heatmap
# ---------------------------------------------------------------------------


def plot_ceps_heatmap(
    results: dict[str, dict[str, float]],
    ceps_totals: dict[str, float] | None = None,
    title: str = "Cross-Stage Error Propagation (CEPS)",
    figsize: tuple = (7, 3.5),
) -> Figure:
    """
    Heatmap: rows = models, columns = S1–S5 + CEPS total.

    Args:
        results:     {model_name: {"S1": float, ..., "S5": float}}
        ceps_totals: {model_name: mean_ceps} — if provided, appended as last column.
        title:       Figure title.
        figsize:     Figure size in inches.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.colors as mcolors

    apply_paper_style()

    model_keys = list(results.keys())
    model_names = [abbrev_model_name(k) for k in model_keys]
    col_ids = STAGE_IDS.copy()
    col_labels = [
        "S1\nInterpret.",
        "S2\nSignal",
        "S3\nOptim.",
        "S4\nExecute",
        "S5\nRisk",
    ]

    if ceps_totals:
        col_ids.append("CEPS")
        col_labels.append("CEPS\nTotal")

    data = []
    for k in model_keys:
        row = [results[k].get(sid, 0.0) for sid in STAGE_IDS]
        if ceps_totals:
            row.append(ceps_totals.get(k, 0.0))
        data.append(row)
    data = np.array(data)

    fig, ax = plt.subplots(figsize=figsize)

    # RdYlGn colormap: red=low, green=high
    cmap = plt.get_cmap("RdBu")
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Draw separator before CEPS column
    if ceps_totals:
        ax.axvline(x=len(STAGE_IDS) - 0.5, color="black", linewidth=2)

    # Annotations
    for r in range(len(model_names)):
        for c in range(len(col_ids)):
            val = data[r, c]
            text_color = "black" if 0.35 < val < 0.85 else "white"
            ax.text(
                c,
                r,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    ax.set_xticks(range(len(col_ids)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Score", fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — CEPS Distribution Violin Plot
# ---------------------------------------------------------------------------


def plot_ceps_violin(
    episode_scores: dict[str, list[float]],
    title: str = "CEPS Score Distribution per Model",
    figsize: tuple = (6, 4),
) -> Figure:
    """
    Violin + box plot of per-episode CEPS scores.

    Args:
        episode_scores: {model_name: [ceps_score_per_episode, ...]}
        title:          Figure title.
        figsize:        Figure size.

    Returns:
        matplotlib Figure.
    """
    apply_paper_style()

    model_keys = list(episode_scores.keys())
    model_names = [abbrev_model_name(k) for k in model_keys]
    data = [episode_scores[k] for k in model_keys]

    fig, ax = plt.subplots(figsize=figsize)

    parts = ax.violinplot(
        data,
        positions=range(len(model_names)),
        showmedians=True,
        showextrema=True,
        widths=0.7,
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(MODEL_PALETTE[i % len(MODEL_PALETTE)])
        pc.set_alpha(0.6)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.2)

    # Overlay individual points (jittered)
    rng = np.random.default_rng(0)
    for i, scores in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=len(scores))
        ax.scatter(
            np.full(len(scores), i) + jitter,
            scores,
            s=12,
            color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
            alpha=0.5,
            zorder=3,
        )

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("CEPS Score")
    ax.set_ylim(0, 1.05)
    ax.axhline(
        0.5, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Pass threshold"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
