"""
QA accuracy visualizations.

All functions follow the codebase convention:
  - apply_paper_style() at entry, no ax.set_title()
  - abbrev_model_name() for all model labels
  - MODEL_PALETTE + LINE_STYLES/LINE_MARKERS for series distinction
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from .style import (
    apply_paper_style,
    MODEL_PALETTE,
    REGIME_COLORS,
    LINE_STYLES,
    LINE_MARKERS,
    abbrev_model_name,
)


_TEMPLATE_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
_TEMPLATE_NAMES = {
    "T1": "Return\nPred.",
    "T2": "VaR\nAssess.",
    "T3": "Position\nSize",
    "T4": "Pairwise\nAlloc.",
    "T5": "Multi-Asset\nOpt.",
    "T6": "Rebalance\nDec.",
    "T7": "Regime\nDetect.",
}
_TEMPLATE_NAMES_SHORT = {
    "T1": "Return Pred.",
    "T2": "VaR Assess.",
    "T3": "Position Size",
    "T4": "Pairwise Alloc.",
    "T5": "Multi-Asset Opt.",
    "T6": "Rebalance Dec.",
    "T7": "Regime Detect.",
}


# ---------------------------------------------------------------------------
# Fig 1 — Accuracy Heatmap (model × template)
# ---------------------------------------------------------------------------

def plot_qa_accuracy_heatmap(
    results: dict[str, dict[str, float]],
    title: str = "",
    figsize: tuple = (9, 4),
) -> Figure:
    """
    Model × Template heatmap of accuracy scores.

    Args:
        results: {model_name: {template_id: accuracy}}
    """
    apply_paper_style()

    models = sorted(results.keys())
    templates = [t for t in _TEMPLATE_ORDER if any(t in results[m] for m in models)]
    model_labels = [abbrev_model_name(m) for m in models]

    data = np.array([
        [results.get(m, {}).get(t, 0.0) for t in templates]
        for m in models
    ])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="RdBu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(templates)))
    ax.set_xticklabels(
        [_TEMPLATE_NAMES_SHORT.get(t, t) for t in templates],
        rotation=30, ha="right",
    )
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=9)

    for i in range(len(models)):
        for j in range(len(templates)):
            val = data[i, j]
            text_color = "black" if 0.35 < val < 0.85 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 2 — Per-template grouped bar chart (one bar group per template, bars = models)
# ---------------------------------------------------------------------------

def plot_qa_per_template_comparison(
    results: dict[str, dict[str, float]],
    title: str = "",
    figsize: tuple = (10, 4),
) -> Figure:
    """
    Grouped bar chart: x = templates, one bar per model per template.

    Args:
        results: {model_name: {template_id: accuracy}}
    """
    apply_paper_style()

    model_keys = sorted(results.keys())
    model_labels = [abbrev_model_name(m) for m in model_keys]
    templates = [t for t in _TEMPLATE_ORDER if any(t in results[m] for m in model_keys)]
    n_models = len(model_keys)
    n_templates = len(templates)

    x = np.arange(n_templates)
    width = 0.75 / n_models

    _HATCHES = ["//", "xx", "..", "++", "\\\\", "oo", "--", "**"]
    palette_size = len(MODEL_PALETTE)

    fig, ax = plt.subplots(figsize=figsize)

    for i, (key, label) in enumerate(zip(model_keys, model_labels)):
        vals = [results.get(key, {}).get(t, 0.0) for t in templates]
        offset = (i - n_models / 2 + 0.5) * width
        # Only add hatch when palette color repeats (i >= palette_size)
        hatch = _HATCHES[(i - palette_size) % len(_HATCHES)] if i >= palette_size else ""
        ax.bar(
            x + offset, vals, width * 0.92,
            color=MODEL_PALETTE[i % palette_size],
            hatch=hatch,
            alpha=0.85, edgecolor="white",
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_TEMPLATE_NAMES_SHORT.get(t, t) for t in templates],
        rotation=20, ha="right", fontsize=9,
    )
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=8, loc="upper right", ncol=max(1, n_models // 3))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 3 — Radar chart: T1-T7 capability profile per model
# ---------------------------------------------------------------------------

def plot_qa_template_radar(
    results: dict[str, dict[str, float]],
    title: str = "",
    figsize: tuple = (6, 6),
) -> Figure:
    """
    Radar (spider) chart with one polygon per model across T1-T7.

    Args:
        results: {model_name: {template_id: accuracy}}
    """
    apply_paper_style()

    model_keys = sorted(results.keys())
    templates = [t for t in _TEMPLATE_ORDER if any(t in results[m] for m in model_keys)]
    n = len(templates)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    labels = [_TEMPLATE_NAMES_SHORT.get(t, t) for t in templates]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7)

    legend_handles = []
    palette_size = len(MODEL_PALETTE)
    for i, key in enumerate(model_keys):
        vals = [results.get(key, {}).get(t, 0.0) for t in templates]
        vals += vals[:1]
        color = MODEL_PALETTE[i % palette_size]
        # Only add line style / marker decoration when palette color repeats
        if i >= palette_size:
            ls = LINE_STYLES[1 + (i - palette_size) % (len(LINE_STYLES) - 1)]
            mk = LINE_MARKERS[(i - palette_size) % len(LINE_MARKERS)]
            ax.plot(angles, vals, color=color, linewidth=2, linestyle=ls,
                    marker=mk, markersize=5)
        else:
            ax.plot(angles, vals, color=color, linewidth=2, linestyle="-")
        ax.fill(angles, vals, color=color, alpha=0.10)
        legend_handles.append(
            mpatches.Patch(color=color, label=abbrev_model_name(key))
        )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=8,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 4 — Accuracy by market regime (averaged across models)
# ---------------------------------------------------------------------------

def plot_qa_accuracy_by_regime(
    results: dict[str, dict[str, dict[str, float]]],
    title: str = "",
    figsize: tuple = (10, 4),
) -> Figure:
    """
    Grouped bar chart: templates on x-axis, bars = regime.

    Args:
        results: {model_name: {template_id: {regime: accuracy}}}
    """
    apply_paper_style()

    merged: dict[str, dict[str, list[float]]] = {}
    for model, tdata in results.items():
        for tid, regimes in tdata.items():
            for regime, acc in regimes.items():
                merged.setdefault(tid, {}).setdefault(regime, []).append(acc)

    templates = [t for t in _TEMPLATE_ORDER if t in merged]
    all_regimes = sorted({r for t in merged.values() for r in t.keys()})

    # Use theme regime colors; fall back to MODEL_PALETTE
    _regime_color_map = {
        "bull": REGIME_COLORS.get("bull", MODEL_PALETTE[0]),
        "bear": REGIME_COLORS.get("bear", MODEL_PALETTE[1]),
        "sideways": REGIME_COLORS.get("sideways", MODEL_PALETTE[2]),
        "crisis": REGIME_COLORS.get("crisis", MODEL_PALETTE[3]),
    }

    x = np.arange(len(templates))
    width = 0.75 / max(len(all_regimes), 1)

    fig, ax = plt.subplots(figsize=figsize)

    for i, regime in enumerate(all_regimes):
        vals = [np.mean(merged[t].get(regime, [0.0])) for t in templates]
        offset = (i - len(all_regimes) / 2 + 0.5) * width
        color = _regime_color_map.get(regime.lower(), MODEL_PALETTE[i % len(MODEL_PALETTE)])
        ax.bar(x + offset, vals, width * 0.92, label=regime.capitalize(),
               color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_TEMPLATE_NAMES_SHORT.get(t, t) for t in templates],
        rotation=20, ha="right", fontsize=9,
    )
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    if all_regimes:
        ax.legend(title="Regime", loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 5 — Score distribution (box + jitter per template)
# ---------------------------------------------------------------------------

def plot_qa_score_distribution(
    results: dict[str, dict[str, list[float]]],
    title: str = "",
    figsize: tuple = (10, 4),
) -> Figure:
    """
    Box + jitter plot of per-question scores grouped by template.

    Args:
        results: {model_name: {template_id: [score, ...]}}
    """
    apply_paper_style()

    merged: dict[str, list[float]] = {}
    for model, tdata in results.items():
        for tid, scores in tdata.items():
            merged.setdefault(tid, []).extend(scores)

    templates = [t for t in _TEMPLATE_ORDER if t in merged and merged[t]]
    data = [merged[t] for t in templates]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(
        data,
        labels=[_TEMPLATE_NAMES_SHORT.get(t, t) for t in templates],
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor=MODEL_PALETTE[0], markersize=6),
        widths=0.5,
    )

    for i, (patch, t) in enumerate(zip(bp["boxes"], templates)):
        patch.set_facecolor(MODEL_PALETTE[i % len(MODEL_PALETTE)])
        patch.set_alpha(0.65)
    for part in ("whiskers", "caps", "medians"):
        for line in bp[part]:
            line.set_color("black")
            line.set_linewidth(1.0)

    # Jitter overlay
    rng = np.random.default_rng(0)
    for i, scores in enumerate(data):
        if not scores:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(scores))
        ax.scatter(
            np.full(len(scores), i + 1) + jitter, scores,
            s=8, color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
            alpha=0.35, zorder=3,
        )

    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 6 — Model comparison bar chart (mean accuracy per model)
# ---------------------------------------------------------------------------

def plot_qa_model_comparison(
    results: dict[str, float],
    title: str = "",
    figsize: tuple = (7, 4),
) -> Figure:
    """
    Horizontal bar chart of mean accuracy per model, sorted descending.

    Args:
        results: {model_name: mean_accuracy}
    """
    apply_paper_style()

    model_keys = sorted(results.keys(), key=lambda m: results[m], reverse=True)
    model_labels = [abbrev_model_name(m) for m in model_keys]
    scores = [results[k] for k in model_keys]

    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, score) in enumerate(zip(model_labels, scores)):
        color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
        bar = ax.barh(i, score, color=color, alpha=0.85, edgecolor="white")
        ax.text(score + 0.01, i, f"{score:.3f}", va="center", fontsize=9)

    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=9)
    ax.set_xlabel("Mean Accuracy")
    ax.set_xlim(0, 1.15)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Info-level comparison — Full vs Restricted (T4/T5)
# ---------------------------------------------------------------------------

def plot_info_level_comparison(
    full_data: dict[str, dict[str, float]],
    restricted_data: dict[str, dict[str, float]],
    figsize: tuple = (10, 5),
) -> Figure:
    """Grouped bar chart: for each model, show full vs restricted accuracy for T4 and T5.

    Args:
        full_data: {model_label: {template_id: accuracy}} for full-info run.
        restricted_data: {model_label: {template_id: accuracy}} for restricted run.
    """
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    compare_templates = ["T4", "T5"]
    template_titles = {"T4": "T4 – Pairwise Allocation", "T5": "T5 – Multi-Asset Opt."}

    common_models = sorted(
        set(full_data) & set(restricted_data),
        key=lambda m: -full_data.get(m, {}).get("T4", 0.0),
    )
    labels = [abbrev_model_name(m) for m in common_models]
    x = np.arange(len(common_models))
    width = 0.38

    for ax, tid in zip(axes, compare_templates):
        full_vals = [full_data.get(m, {}).get(tid, 0.0) for m in common_models]
        rest_vals = [restricted_data.get(m, {}).get(tid, 0.0) for m in common_models]

        bars_full = ax.bar(x - width / 2, full_vals, width, label="Full info",
                           color="#4C72B0", alpha=0.85, edgecolor="white")
        bars_rest = ax.bar(x + width / 2, rest_vals, width, label="Restricted",
                           color="#DD8452", alpha=0.85, edgecolor="white")

        for bar in bars_full:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)
        for bar in bars_rest:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.15)
        ax.set_title(template_titles[tid], fontsize=10)
        ax.legend(fontsize=8)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

    fig.tight_layout()
    return fig


def plot_info_level_drop_heatmap(
    drop_data: dict[str, dict[str, float]],
    figsize: tuple = (6, 4),
) -> Figure:
    """Heatmap of accuracy drop (full − restricted) for T4 and T5 across models.

    Args:
        drop_data: {model_label: {"T4": drop, "T5": drop}}  (positive = drop, negative = gain)
    """
    apply_paper_style()

    compare_templates = ["T4", "T5"]
    models = sorted(drop_data, key=lambda m: -(drop_data[m].get("T4", 0) + drop_data[m].get("T5", 0)))
    labels = [abbrev_model_name(m) for m in models]

    matrix = np.array([[drop_data[m].get(t, 0.0) for t in compare_templates] for m in models])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy drop (full − restricted)")

    ax.set_xticks(range(len(compare_templates)))
    ax.set_xticklabels(
        [_TEMPLATE_NAMES_SHORT[t] for t in compare_templates], fontsize=9
    )
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=9)

    for i, m in enumerate(models):
        for j, t in enumerate(compare_templates):
            val = matrix[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(val) > 0.25 else "black")

    fig.tight_layout()
    return fig
