"""
QA dataset statistics visualization (extended).

Figures:
  plot_dataset_overview  — 3×2 panel covering all key dataset dimensions
  plot_text_richness     — bar chart of avg context length + % with news per template
  plot_regime_split_heatmap — template × regime count heatmap
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from .style import apply_paper_style, REGIME_COLORS, STAGE_COLORS

# Arctic Frost source values used directly for split encoding
_AF1 = "#4a6fa5"  # steel blue
_AF2 = "#d4e4f7"  # ice blue
_AF3 = "#c0c0c0"  # silver

_TEMPLATE_FULL = {
    "T1": "T1: Return\nPrediction",
    "T2": "T2: Risk\nAssessment",
    "T3": "T3: Position\nSizing",
    "T4": "T4: Pairwise\nAlloc.",
    "T5": "T5: Multi-asset\nOptim.",
    "T6": "T6: Rebalancing\nDecision",
    "T7": "T7: Regime\nDetection",
}
_TEMPLATE_IDS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
_SPLITS = ["train", "val", "test"]
_REGIMES = ["sideways", "bull", "bear"]
_SPLIT_COLORS = ["#1e3d6e", _AF1, _AF2]  # dark→mid→light for train/val/test
_COMPLEXITY = {"T1": 1, "T2": 1, "T3": 1, "T4": 2, "T5": 3, "T6": 3, "T7": 4}
_COMPLEXITY_LABEL = {
    1: "L1 Single-asset",
    2: "L2 Pairwise",
    3: "L3 Multi-asset",
    4: "L4 Full portfolio",
}


def plot_dataset_overview(
    stats: dict,
    title: str = "PortBench QA Dataset Overview",
    figsize: tuple = (14, 9),
) -> Figure:
    """
    3×2 comprehensive overview panel.

    Panel layout:
      (a) Stacked bar: sample count per template, colored by split
      (b) Grouped bar: sample count per template by regime
      (c) Horizontal bar: complexity level distribution (template count per level)
      (d) Pie: overall regime distribution (pooled across templates)
      (e) Bar: % of samples with news/filing text per template
      (f) Split timeline: data range annotation per split

    Args:
        stats: dict from outputs/qa_dataset/stats.json
        title, figsize: passed to plt.subplots
    """
    apply_paper_style()
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    templates = [t for t in _TEMPLATE_IDS if t in stats]

    # ----------------------------------------------------------------
    # (a) Stacked bar: count by split
    # ----------------------------------------------------------------
    ax = axes[0, 0]
    bottoms = np.zeros(len(templates))
    for i, split in enumerate(_SPLITS):
        counts = [stats[t]["by_split"].get(split, 0) for t in templates]
        bars = ax.bar(
            templates,
            counts,
            bottom=bottoms,
            color=_SPLIT_COLORS[i],
            label=split,
            alpha=0.88,
        )
        bottoms += np.array(counts)
    totals = [stats[t]["n_total"] for t in templates]
    for j, (x, tot) in enumerate(zip(templates, totals)):
        ax.text(
            j,
            tot + 10,
            str(tot),
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
    ax.set_ylabel("Number of QA Pairs")
    ax.set_title("(a) Sample Count by Template & Split", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks(range(len(templates)))
    ax.set_xticklabels(templates)

    # ----------------------------------------------------------------
    # (b) Grouped bar: count by regime
    # ----------------------------------------------------------------
    ax = axes[0, 1]
    regime_colors_list = [REGIME_COLORS.get(r, "#cccccc") for r in _REGIMES]
    x = np.arange(len(templates))
    bw = 0.25
    for i, (regime, rcolor) in enumerate(zip(_REGIMES, regime_colors_list)):
        counts = [stats[t]["by_regime"].get(regime, 0) for t in templates]
        ax.bar(
            x + (i - 1) * bw,
            counts,
            width=bw,
            color=rcolor,
            label=regime.capitalize(),
            alpha=0.88,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(templates)
    ax.set_ylabel("Number of QA Pairs")
    ax.set_title("(b) Sample Count by Template & Market Regime", fontsize=10)
    ax.legend(fontsize=8)

    # ----------------------------------------------------------------
    # (c) Complexity level distribution (n templates per level)
    # ----------------------------------------------------------------
    ax = axes[1, 0]
    level_template_map: dict[int, list[str]] = {}
    for t in templates:
        lvl = _COMPLEXITY.get(t, 1)
        level_template_map.setdefault(lvl, []).append(t)
    levels = sorted(level_template_map.keys())
    level_labels = [_COMPLEXITY_LABEL.get(l, f"Level {l}") for l in levels]
    level_n_pairs = [
        sum(stats[t]["n_total"] for t in level_template_map[l]) for l in levels
    ]
    bars = ax.bar(
        range(len(levels)), level_n_pairs, color=STAGE_COLORS[: len(levels)], alpha=0.88
    )
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(level_labels, fontsize=8.5)
    ax.set_ylabel("Total QA Pairs")
    ax.set_title("(c) Pairs by Complexity Level", fontsize=10)
    for bar, n in zip(bars, level_n_pairs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            str(n),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # ----------------------------------------------------------------
    # (d) Pie: overall regime distribution
    # ----------------------------------------------------------------
    ax = axes[1, 1]
    regime_totals: dict[str, int] = {}
    for t in templates:
        for r, cnt in stats[t]["by_regime"].items():
            regime_totals[r] = regime_totals.get(r, 0) + cnt
    labels = list(regime_totals.keys())
    counts = list(regime_totals.values())
    colors = [REGIME_COLORS.get(r, "#cccccc") for r in labels]
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=[l.capitalize() for l in labels],
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.80,
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title("(d) Overall Regime Distribution", fontsize=10)

    # ----------------------------------------------------------------
    # (e) Bar: % samples with news text + avg context length
    # ----------------------------------------------------------------
    ax = axes[2, 0]
    pct_text = [stats[t]["text"]["pct_with_text"] for t in templates]
    color_pct = _AF1
    bars = ax.bar(
        templates, pct_text, color=color_pct, alpha=0.80, label="% with news/filing"
    )
    ax.set_ylabel("% Samples with News / Filing Text", color=color_pct)
    ax.set_ylim(0, 115)
    ax.tick_params(axis="y", labelcolor=color_pct)
    for bar, pct in zip(bars, pct_text):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Overlay: avg context chars on twin axis
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    avg_chars = [stats[t]["text"]["avg_chars"] for t in templates]
    ax2.plot(
        templates,
        avg_chars,
        "o--",
        color="#1e3d6e",
        linewidth=1.8,
        markersize=5,
        label="Avg context length (chars)",
    )
    ax2.set_ylabel("Avg Context Length (chars)", color="#1e3d6e")
    ax2.tick_params(axis="y", labelcolor="#1e3d6e")

    lines1 = [mpatches.Patch(color=color_pct, label="% with news")]
    lines2 = [
        plt.Line2D(
            [0], [0], color="#1e3d6e", marker="o", linestyle="--", label="Avg chars"
        )
    ]
    ax.legend(handles=lines1 + lines2, fontsize=8, loc="upper left")
    ax.set_title("(e) Text Richness per Template", fontsize=10)

    # ----------------------------------------------------------------
    # (f) Split time range annotation
    # ----------------------------------------------------------------
    ax = axes[2, 1]
    meta = stats.get("_meta", {})
    split_bounds = meta.get(
        "split_boundaries",
        {
            "train": ["2015-01-02", "2022-12-31"],
            "val": ["2023-01-01", "2024-12-31"],
            "test": ["2025-01-01", "2025-12-31"],
        },
    )

    split_list = ["train", "val", "test"]
    ys = [2, 1, 0]
    for split, y, color in zip(split_list, ys, _SPLIT_COLORS):
        bounds = split_bounds.get(split, ["?", "?"])
        start_yr = int(bounds[0][:4])
        end_yr = int(bounds[1][:4]) + 1
        ax.barh(
            y,
            end_yr - start_yr,
            left=start_yr,
            height=0.5,
            color=color,
            alpha=0.80,
            label=split.capitalize(),
        )
        ax.text(
            (start_yr + end_yr) / 2,
            y,
            f"{bounds[0]} → {bounds[1]}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([s.capitalize() for s in split_list])
    ax.set_xlabel("Year")
    ax.set_xlim(2014, 2027)
    ax.set_title("(f) Temporal Data Split", fontsize=10)
    overall = meta.get("text_overall", {})
    if overall:
        info = (
            f"Total QA pairs: {overall.get('n_total', '?'):,}\n"
            f"{overall.get('pct_with_text', '?'):.0f}% with news/filing context\n"
            f"Avg context: {overall.get('avg_chars', '?'):.0f} chars"
        )
        ax.text(
            0.98,
            0.05,
            info,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85),
        )

    fig.tight_layout()
    return fig


def plot_regime_heatmap(
    stats: dict,
    title: str = "Template × Regime Sample Distribution",
    figsize: tuple = (7, 4),
) -> Figure:
    """
    Heatmap: rows = templates, cols = regimes, cell = sample count.
    """
    apply_paper_style()

    templates = [t for t in _TEMPLATE_IDS if t in stats]
    regimes = _REGIMES

    data = np.array(
        [[stats[t]["by_regime"].get(r, 0) for r in regimes] for t in templates],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="Blues", aspect="auto")

    for r in range(len(templates)):
        for c in range(len(regimes)):
            val = int(data[r, c])
            text_color = "white" if data[r, c] > data.max() * 0.6 else "black"
            ax.text(
                c,
                r,
                str(val),
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([r.capitalize() for r in regimes])
    ax.set_yticks(range(len(templates)))
    ax.set_yticklabels(templates)
    ax.set_title(title, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Sample Count", fontsize=9)
    fig.tight_layout()
    return fig
