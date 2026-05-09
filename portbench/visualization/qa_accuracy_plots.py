"""
QA accuracy visualizations — heatmaps, regime breakdowns, score distributions.

All functions follow the existing visualization module pattern:
  - Accept data dicts, return matplotlib Figure
  - Use apply_paper_style() for consistent look
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .style import apply_paper_style, PAPER_COLORS


_TEMPLATE_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
_TEMPLATE_NAMES = {
    "T1": "Return Pred.",
    "T2": "VaR Assess.",
    "T3": "Position Size",
    "T4": "Pairwise Alloc.",
    "T5": "Multi-Asset Opt.",
    "T6": "Rebalance Dec.",
    "T7": "Regime Detect.",
}


def plot_qa_accuracy_heatmap(
    results: dict[str, dict[str, float]],
    title: str = "QA Accuracy by Model and Template",
    figsize: tuple = (9, 4),
) -> Figure:
    """
    Model x Template heatmap of accuracy scores.

    Args:
        results: {model_name: {template_id: accuracy}}
    """
    apply_paper_style()

    models = sorted(results.keys())
    templates = [t for t in _TEMPLATE_ORDER if any(t in results[m] for m in models)]

    data = np.array([
        [results.get(m, {}).get(t, 0.0) for t in templates]
        for m in models
    ])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(templates)))
    ax.set_xticklabels(
        [_TEMPLATE_NAMES.get(t, t) for t in templates],
        rotation=45, ha="right",
    )
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(templates)):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_qa_accuracy_by_regime(
    results: dict[str, dict[str, dict[str, float]]],
    title: str = "QA Accuracy by Market Regime",
    figsize: tuple = (10, 5),
) -> Figure:
    """
    Grouped bar chart: templates on x-axis, bars colored by regime.

    Args:
        results: {model_name: {template_id: {regime: accuracy}}}
    """
    apply_paper_style()

    # Merge all models' regime data (average if multi-model)
    merged: dict[str, dict[str, list[float]]] = {}
    for model, tdata in results.items():
        for tid, regimes in tdata.items():
            if tid not in merged:
                merged[tid] = {}
            for regime, acc in regimes.items():
                merged[tid].setdefault(regime, []).append(acc)

    templates = [t for t in _TEMPLATE_ORDER if t in merged]
    all_regimes = sorted({r for t in merged.values() for r in t.keys()})

    regime_colors = {
        "BULL": "#2ecc71", "bull": "#2ecc71",
        "BEAR": "#e74c3c", "bear": "#e74c3c",
        "SIDEWAYS": "#f39c12", "sideways": "#f39c12",
        "CRISIS": "#8e44ad", "crisis": "#8e44ad",
    }

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(templates))
    width = 0.8 / max(len(all_regimes), 1)

    for i, regime in enumerate(all_regimes):
        vals = [
            np.mean(merged[t].get(regime, [0.0])) for t in templates
        ]
        offset = (i - len(all_regimes) / 2 + 0.5) * width
        color = regime_colors.get(regime, PAPER_COLORS.get("accent", "#3498db"))
        ax.bar(x + offset, vals, width, label=regime, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([_TEMPLATE_NAMES.get(t, t) for t in templates], rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Regime", loc="upper right")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_qa_score_distribution(
    results: dict[str, dict[str, list[float]]],
    title: str = "QA Score Distribution",
    figsize: tuple = (10, 5),
) -> Figure:
    """
    Box plot of per-question scores, grouped by template.

    Args:
        results: {model_name: {template_id: [score, ...]}}
    """
    apply_paper_style()

    # Merge all models
    merged: dict[str, list[float]] = {}
    for model, tdata in results.items():
        for tid, scores in tdata.items():
            merged.setdefault(tid, []).extend(scores)

    templates = [t for t in _TEMPLATE_ORDER if t in merged and merged[t]]
    data = [merged[t] for t in templates]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(
        data,
        labels=[_TEMPLATE_NAMES.get(t, t) for t in templates],
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
    )

    colors = plt.cm.Set2(np.linspace(0, 1, len(templates)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_qa_model_comparison(
    results: dict[str, float],
    title: str = "QA Model Comparison — Mean Accuracy",
    figsize: tuple = (7, 4),
) -> Figure:
    """
    Horizontal bar chart of mean accuracy per model.

    Args:
        results: {model_name: mean_accuracy}
    """
    apply_paper_style()

    models = sorted(results.keys(), key=lambda m: results[m], reverse=True)
    scores = [results[m] for m in models]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax.barh(range(len(models)), scores, color=colors)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("Mean Accuracy")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)

    ax.set_title(title)
    fig.tight_layout()
    return fig
