"""
QA sample card visualization — renders one example per template as a figure.

Each card shows:
  - Template ID + task name  (colored header)
  - Metadata chips (date, regime, complexity, assets)
  - Context (news snippet if available, else context_summary)
  - Question (first meaningful line, truncated)
  - Answer (highlighted box)
  - Explanation label + italic text

Figure:
  plot_qa_sample_cards  — 4×2 grid (7 cards + legend cell)
  plot_single_card       — standalone card for one template
"""

from __future__ import annotations

import textwrap
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .style import apply_paper_style, REGIME_COLORS


_TEMPLATE_NAMES = {
    "T1": "Return Prediction",
    "T2": "Risk Assessment (VaR)",
    "T3": "Position Sizing",
    "T4": "Pairwise Allocation",
    "T5": "Multi-Asset Optimization",
    "T6": "Rebalancing Decision",
    "T7": "Regime Detection & Allocation",
}

_COMPLEXITY_COLOR = {
    1: "#1e3d6e",   # darkest steel blue
    2: "#4a6fa5",   # steel blue
    3: "#7a9fc5",   # steel→ice midpoint
    4: "#c0c0c0",   # silver
}

_COMPLEXITY_LABEL = {
    1: "Level 1  Single-asset",
    2: "Level 2  Pairwise",
    3: "Level 3  Multi-asset",
    4: "Level 4  Full portfolio",
}


def _wrap(text: str, width: int, max_lines: int) -> str:
    lines = textwrap.wrap(text, width=width)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines]) + " ..."


def _extract_context_for_display(sample: dict) -> str:
    """
    Prefer news/filing text over context_summary.
    Returns a short snippet (≤200 chars) followed by '...' if truncated.
    """
    # Check for news text embedded in question (after 'Recent filing/news:')
    question = sample.get("question", "")
    news_marker = "Recent filing/news:"
    if news_marker in question:
        news_part = question.split(news_marker, 1)[1].strip()
        # Take first 200 chars
        snippet = news_part[:200].replace("\n", " ").strip()
        return snippet + " ..." if len(news_part) > 250 else snippet

    # Fallback: context_summary
    ctx = sample.get("context_summary", "")
    if ctx:
        return ctx[:250] + " ..." if len(ctx) > 250 else ctx
    return "(no context available)"


def _first_meaningful_question_line(question: str) -> str:
    """
    Extract the actual question sentence from multi-line question strings.
    Skips lines that are purely data (prices, matrices, macro values).
    """
    lines = [l.strip() for l in question.split("\n") if l.strip()]
    # Find the line that ends with '?' or contains instruction verbs
    keywords = ("predict", "compute", "determine", "identify", "allocate",
                 "calculate", "what", "should", "estimate", "?")
    for line in lines:
        if any(line.lower().endswith(k) or line.lower().startswith(k) for k in keywords):
            return line
    # Fall back to last non-empty line
    return lines[-1] if lines else question


def _draw_card(ax: Axes, sample: dict) -> None:
    """Render a single QA pair as a styled card inside ax."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    tmpl        = sample.get("template", "?")
    complexity  = sample.get("complexity", 1)
    regime      = sample.get("market_regime", "sideways")
    assets      = sample.get("assets", [])
    date_str    = sample.get("decision_date", "")
    question    = sample.get("question", "")
    answer      = str(sample.get("answer", ""))
    explanation = sample.get("explanation", "")

    header_color = _COMPLEXITY_COLOR.get(complexity, "#555555")
    regime_color = REGIME_COLORS.get(regime, "#95a7b5")

    # ── Header band ──────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0.88), 1, 0.12,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor=header_color, transform=ax.transAxes, clip_on=False
    ))
    ax.text(0.5, 0.941,
            f"{tmpl}  |  {_TEMPLATE_NAMES.get(tmpl, tmpl)}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white")
    ax.text(0.97, 0.941,
            _COMPLEXITY_LABEL.get(complexity, f"L{complexity}"),
            transform=ax.transAxes, ha="right", va="center",
            fontsize=6.2, color="white", alpha=0.85)

    # ── Card background ───────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 0.88,
        boxstyle="round,pad=0.01", linewidth=0.8,
        facecolor="#f8f9fa", edgecolor="#cccccc",
        transform=ax.transAxes, clip_on=False
    ))

    # ── Metadata chips ────────────────────────────────────────────
    y_meta = 0.845
    chip_x = 0.03
    asset_str = ", ".join(assets[:3]) + (f" +{len(assets)-3}" if len(assets) > 3 else "")
    meta_items = [
        (f"Date: {date_str}",         "#e8eaf6"),
        (f"Regime: {regime.upper()}", regime_color + "44"),
        (f"Assets: {asset_str}",      "#e8f5e9"),
    ]
    for chip_text, bg_color in meta_items:
        tw = len(chip_text) * 0.0115 + 0.04
        ax.add_patch(mpatches.FancyBboxPatch(
            (chip_x, y_meta - 0.023), tw, 0.038,
            boxstyle="round,pad=0.005", linewidth=0.5,
            facecolor=bg_color, edgecolor="#bbbbbb",
            transform=ax.transAxes, clip_on=False
        ))
        ax.text(chip_x + tw / 2, y_meta - 0.004, chip_text,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=6.2, color="#333333")
        chip_x += tw + 0.018

    # ── CONTEXT section ───────────────────────────────────────────
    ax.text(0.03, 0.760, "CONTEXT",
            transform=ax.transAxes, fontsize=5.8, color="#888888", fontweight="bold")
    ctx_display = _extract_context_for_display(sample)
    ctx_text    = _wrap(ctx_display, width=70, max_lines=2)
    ax.text(0.03, 0.737, ctx_text,
            transform=ax.transAxes, fontsize=6.6, color="#444444",
            va="top", linespacing=1.35)

    # ── Divider ───────────────────────────────────────────────────
    ax.axhline(0.625, xmin=0.03, xmax=0.97, color="#cccccc", linewidth=0.6)

    # ── QUESTION section ──────────────────────────────────────────
    ax.text(0.03, 0.588, "QUESTION",
            transform=ax.transAxes, fontsize=5.8, color="#888888", fontweight="bold")
    q_display = _first_meaningful_question_line(question)
    q_text    = _wrap(q_display, width=70, max_lines=3)
    ax.text(0.03, 0.565, q_text,
            transform=ax.transAxes, fontsize=6.6, color="#222222",
            va="top", linespacing=1.35)

    # ── Divider ───────────────────────────────────────────────────
    ax.axhline(0.390, xmin=0.03, xmax=0.97, color="#cccccc", linewidth=0.6)

    # ── ANSWER section ────────────────────────────────────────────
    ax.text(0.03, 0.353, "ANSWER",
            transform=ax.transAxes, fontsize=5.8, color="#888888", fontweight="bold")
    ans_short = _wrap(answer, width=55, max_lines=2)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.03, 0.213), 0.94, 0.115,
        boxstyle="round,pad=0.01", linewidth=0.6,
        facecolor=header_color + "18", edgecolor=header_color + "66",
        transform=ax.transAxes, clip_on=False
    ))
    ax.text(0.5, 0.270, ans_short,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=header_color)

    # ── EXPLANATION section ───────────────────────────────────────
    ax.text(0.03, 0.169, "EXPLANATION",
            transform=ax.transAxes, fontsize=5.8, color="#888888", fontweight="bold")
    exp_text = _wrap(explanation, width=74, max_lines=2)
    ax.text(0.03, 0.145, exp_text,
            transform=ax.transAxes, fontsize=6.2, color="#555555",
            va="top", linespacing=1.3, style="italic")


def plot_qa_sample_cards(
    samples: dict[str, dict],
    title: str = "PortBench QA Dataset — One Example per Template",
    figsize: tuple = (18, 14),
) -> Figure:
    """
    4×2 grid of QA sample cards (7 cards + 1 legend cell).

    Args:
        samples: {template_id: qa_pair_dict}
        title, figsize: figure-level settings.
    """
    apply_paper_style(font_family="sans-serif", base_size=9)

    fig, axes = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.005)

    template_ids = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    positions    = [(r, c) for r in range(4) for c in range(2)]

    for i, tmpl in enumerate(template_ids):
        row, col = positions[i]
        ax = axes[row, col]
        sample = samples.get(tmpl, {})
        if sample:
            _draw_card(ax, sample)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{tmpl}: no sample", ha="center",
                    va="center", fontsize=10, color="gray")

    # Legend cell (bottom-right)
    ax_legend = axes[3, 1]
    ax_legend.axis("off")
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)

    ax_legend.text(0.5, 0.96, "Complexity Level Color Key",
                   ha="center", va="top", fontsize=9, fontweight="bold")
    for j, (lvl, color) in enumerate(_COMPLEXITY_COLOR.items()):
        y = 0.82 - j * 0.14
        ax_legend.add_patch(mpatches.FancyBboxPatch(
            (0.06, y - 0.04), 0.09, 0.08,
            boxstyle="round,pad=0.01", facecolor=color,
            linewidth=0, transform=ax_legend.transAxes
        ))
        ax_legend.text(0.20, y, _COMPLEXITY_LABEL[lvl],
                       transform=ax_legend.transAxes,
                       ha="left", va="center", fontsize=8)

    ax_legend.text(0.5, 0.24, "Market Regime Colors",
                   ha="center", va="top", fontsize=9, fontweight="bold")
    regime_items = list(REGIME_COLORS.items())
    for j, (regime, color) in enumerate(regime_items):
        col_x = 0.06 if j < 2 else 0.52
        row_y = 0.12 - (j % 2) * 0.10
        ax_legend.add_patch(mpatches.FancyBboxPatch(
            (col_x, row_y - 0.025), 0.07, 0.05,
            boxstyle="round,pad=0.005", facecolor=color,
            linewidth=0, transform=ax_legend.transAxes
        ))
        ax_legend.text(col_x + 0.09, row_y, regime.capitalize(),
                       transform=ax_legend.transAxes,
                       ha="left", va="center", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 1])
    return fig


def plot_single_card(
    sample: dict,
    figsize: tuple = (6, 4),
) -> Figure:
    """Standalone card figure for one template (paper appendix)."""
    apply_paper_style(font_family="sans-serif", base_size=9)
    fig, ax = plt.subplots(figsize=figsize)
    _draw_card(ax, sample)
    fig.tight_layout()
    return fig
