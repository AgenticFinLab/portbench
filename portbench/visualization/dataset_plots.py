"""
QA dataset statistics visualization (extended).

Figures:
  plot_dataset_overview  — 3x2 panel covering all key dataset dimensions
  plot_regime_heatmap    — template x regime count heatmap
  plot_rawdata_overview  — two-panel raw time-series + news coverage overview
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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

    # ----------------------------------------------------------------
    # (e) Bar: % samples with news text + avg context length
    # ----------------------------------------------------------------
    ax = axes[2, 0]
    pct_text = [stats[t]["text"]["pct_with_text"] for t in templates]
    color_pct = _AF1
    bars = ax.bar(
        templates, pct_text, color=color_pct, alpha=0.80, label="% with news/filing"
    )
    ax.set_ylabel("% Samples with News / Filing Text", color="black")
    ax.set_ylim(0, 115)
    ax.tick_params(axis="y", labelcolor="black")
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

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Sample Count", fontsize=9)
    fig.tight_layout()
    return fig


# ── Raw data overview ──────────────────────────────────────────────────────

_RAW_ASSET_COLORS = {
    "equities": "#1e3d6e",  # dark navy
    "bonds": "#5b9bd5",  # cornflower blue
    "commodities": "#7a9fc5",  # mid steel-blue
    "crypto": "#8a8a8a",  # dark silver-gray
    "real_estate": "#a8b8c0",  # slate-silver
    "cash": "#2e6db4",  # medium royal blue
}

_RAW_ASSET_LABELS = {
    "equities": "Equities (IVV)",
    "bonds": "Bonds — 10Y Yield",
    "commodities": "Commodities (GLD)",
    "crypto": "Crypto (BTC)",
    "real_estate": "Real Estate (VNQ)",
    "cash": "Cash — Fed Funds",
}

_RATE_COLORS = {
    "bonds": "#c44e52",
    "cash": "#d9744a",
}

_NEWS_CHARS_COLOR = "#8b5cf6"

_CRISES = [
    ("2015-08-01", "2015-09-30", "2015 China Shock"),
    ("2020-02-15", "2020-04-20", "2020 COVID Crash   "),
    ("2022-01-01", "2022-12-31", "   2022 Crypto Collapse"),
]


def _normalize_series(s: pd.Series) -> pd.Series:
    first_valid = s.first_valid_index()
    if first_valid is None:
        return s
    return s / s.loc[first_valid] * 100.0


def _monthly_headline_count(text_col: pd.Series) -> pd.Series:
    """Count total headlines per row, return monthly-resampled sum."""
    counts = pd.Series(0, index=text_col.index, dtype=float)
    for idx, val in text_col.items():
        if pd.isna(val):
            continue
        try:
            items = json.loads(val)
            n = 0
            for item in items:
                if isinstance(item, dict):
                    raw = item.get("text") or item.get("texts") or []
                    if isinstance(raw, str):
                        try:
                            raw = json.loads(raw)
                        except Exception:
                            raw = [raw]
                    n += len(raw) if isinstance(raw, list) else 1
                elif isinstance(item, str):
                    n += 1
            counts[idx] = n
        except Exception:
            pass
    return counts.resample("ME").sum()


def _monthly_char_count(text_col: pd.Series) -> pd.Series:
    """Count total text characters per row, return monthly-resampled sum."""
    counts = pd.Series(0, index=text_col.index, dtype=float)
    for idx, val in text_col.items():
        if pd.isna(val):
            continue
        try:
            items = json.loads(val)
            n = 0
            for item in items:
                if isinstance(item, dict):
                    raw = item.get("text") or item.get("texts") or []
                    if isinstance(raw, str):
                        try:
                            raw = json.loads(raw)
                        except Exception:
                            raw = [raw]
                    if isinstance(raw, list):
                        n += sum(len(str(t)) for t in raw)
                    else:
                        n += len(str(raw))
                elif isinstance(item, str):
                    n += len(item)
            counts[idx] = n
        except Exception:
            pass
    return counts.resample("ME").sum()


def plot_rawdata_overview(
    csv_path: str = "datasets/processed/portbench.csv",
    figsize: tuple = (8, 4.0),
) -> Figure:
    """
    Single-panel figure showing portbench.csv raw data at a glance.

    Main panel: normalized price index (equities, commodities, crypto,
                real estate) with right y-axis for interest rates (10Y
                Treasury yield, Fed Funds rate). Market-stress windows
                highlighted.
    Inset strip: monthly news headline count (equities + crypto stacked)
                 embedded at the bottom of the main panel.

    Args:
        csv_path: Path to datasets/processed/portbench.csv.
        figsize:  Figure size.

    Returns:
        matplotlib Figure.
    """
    apply_paper_style()

    df = pd.read_csv(csv_path, parse_dates=["date"], low_memory=False)
    df = df.sort_values("date").reset_index(drop=True).set_index("date")

    # ── Representative series ────────────────────────────────────────────
    price_cfg = [
        ("equities", "equities_IVV_close"),
        ("commodities", "commodities_GLD_close"),
        ("crypto", "cryptocurrency_BTC_USD_close"),
        ("real_estate", "real_estate_VNQ_close"),
    ]
    rate_cfg = [
        ("bonds", "bonds_fred_DGS10"),
        ("cash", "cash_fred_DFF"),
    ]

    price_series = {
        k: _normalize_series(df[c]) for k, c in price_cfg if c in df.columns
    }
    rate_series = {k: df[c].dropna() for k, c in rate_cfg if c in df.columns}

    # ── Monthly character count ───────────────────────────────────────────
    chars_eq = (
        _monthly_char_count(df["equities_text_json"])
        if "equities_text_json" in df.columns
        else pd.Series(dtype=float)
    )
    chars_cry = (
        _monthly_char_count(df["cryptocurrency_text_json"])
        if "cryptocurrency_text_json" in df.columns
        else pd.Series(dtype=float)
    )

    # ── Layout: single panel ─────────────────────────────────────────────
    fig, ax_top = plt.subplots(1, 1, figsize=figsize)
    ax_rate = ax_top.twinx()  # right axis — interest rates
    ax_chars = ax_top.twinx()  # right axis — news char count (offset)

    # ── Price lines (left axis) ──────────────────────────────────────────
    _MARKERS = ["o", "s", "^", "D"]
    for idx, (key, series) in enumerate(price_series.items()):
        every = max(1, len(series) // 12)
        ax_top.plot(
            series.index,
            series.values,
            label=_RAW_ASSET_LABELS[key],
            color=_RAW_ASSET_COLORS[key],
            linewidth=1.6,
            marker=_MARKERS[idx % len(_MARKERS)],
            markevery=every,
            markersize=3.5,
            markerfacecolor=_RAW_ASSET_COLORS[key],
            markeredgewidth=0.4,
            markeredgecolor="white",
            zorder=3,
        )

    # ── Rate lines (right axis, dashed) ─────────────────────────────────
    _RATE_LS = ["--", "--"]
    for idx, (key, series) in enumerate(rate_series.items()):
        ax_rate.plot(
            series.index,
            series.values,
            label=_RAW_ASSET_LABELS[key],
            color=_RATE_COLORS[key],
            linestyle=_RATE_LS[idx % len(_RATE_LS)],
            linewidth=1.2,
            alpha=0.85,
            zorder=2,
        )

    ax_top.axhline(100, color="#c0c0c0", linestyle=":", linewidth=0.7, alpha=0.5)
    ax_top.set_ylabel("Normalized Index  (base = 100)", fontsize=10)
    ax_rate.set_ylabel("Interest Rate  (%)", fontsize=10, color="#c44e52")
    ax_rate.tick_params(axis="y", colors="#c44e52", labelsize=9)
    ax_top.set_xlim(df.index.min(), df.index.max())

    # ── ylim: positive region only ───────────────────────────────────────
    all_price_vals = [s.dropna().max() for s in price_series.values()]
    price_max = max(all_price_vals) if all_price_vals else 400
    price_top = price_max * 1.06
    ax_top.set_ylim(0, price_top)
    ax_rate.set_ylim(0, 9.0)

    # ── News text coverage background (height = % of max chars) ─────────
    ax_chars.set_visible(False)
    chars_idx = chars_eq.index.union(chars_cry.index)
    chars_total = chars_eq.reindex(chars_idx, fill_value=0) + chars_cry.reindex(
        chars_idx, fill_value=0
    )

    if len(chars_idx) > 0 and chars_total.max() > 0:
        chars_frac = chars_total / chars_total.max()
        ax_top.fill_between(
            chars_idx,
            0,
            chars_frac * price_top,
            color="#d4e4f7",
            alpha=0.35,
            zorder=0,
        )

    # ── Crisis shading (deep blue) ──────────────────────────────────────
    for start_s, end_s, label in _CRISES:
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        ax_top.axvspan(start, end, color="#1e3d6e", alpha=0.15, zorder=0)
        ax_top.text(
            start + (end - start) / 2,
            1.01,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1e3d6e",
            fontweight="bold",
            transform=ax_top.get_xaxis_transform(),
            zorder=5,
        )

    # ── Legend (upper left) ──────────────────────────────────────────────
    price_handles = [
        mlines.Line2D(
            [],
            [],
            color=_RAW_ASSET_COLORS[k],
            marker=_MARKERS[i % len(_MARKERS)],
            linestyle="-",
            markersize=5,
            label=_RAW_ASSET_LABELS[k],
        )
        for i, k in enumerate(price_series)
    ]
    rate_handles = [
        mlines.Line2D(
            [],
            [],
            color=_RATE_COLORS[k],
            linestyle=_RATE_LS[i % len(_RATE_LS)],
            linewidth=1.3,
            label=_RAW_ASSET_LABELS[k],
        )
        for i, k in enumerate(rate_series)
    ]
    crisis_patch = mpatches.Patch(
        color="#1e3d6e", alpha=0.15, label="Market Stress Window"
    )
    news_bg_patch = mpatches.Patch(
        color="#d4e4f7", alpha=0.25, label="News Text Coverage"
    )
    ax_top.legend(
        handles=price_handles + rate_handles + [crisis_patch, news_bg_patch],
        fontsize=9,
        loc="upper left",
        framealpha=0.9,
        ncols=2,
    )

    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_top.xaxis.set_major_locator(mdates.YearLocator())
    ax_top.set_xlabel("Year", fontsize=10)
    ax_top.tick_params(axis="both", labelsize=9)

    fig.tight_layout()
    return fig


# ── QA Dataset Capability Overview ────────────────────────────────────────

_QA_TEMPLATES = [
    dict(
        id="T1",
        level=1,
        task="Return Prediction",
        desc="Forecast 21-day price\ndirection (pos / neg / flat)",
        ex_q="Forecast IVV's 21-day return direction.",
        ex_ctx="Momentum: −3.2% (20-day);\nNews: Fed signals rate cut",
        ex_a="→  Negative ↓  (−3.2%)",
    ),
    dict(
        id="T2",
        level=1,
        task="Risk Assessment",
        desc="Compute VaR at a given\nconfidence level",
        ex_q="Compute 95% VaR for $10K IVV, 10 days.",
        ex_ctx="σ = 18.4% ann.; t-distribution;\nnormal-period volatility used",
        ex_a="→  VaR = −$412  (−4.1%)",
    ),
    dict(
        id="T3",
        level=1,
        task="Position Sizing",
        desc="Size positions under a\nmax-drawdown constraint",
        ex_q="Max drawdown ≤ 10%; size IVV position.",
        ex_ctx="Sharpe = 0.82; σ = 18.4% ann.;\nKelly fraction = 0.52",
        ex_a="→  Position = 47% of capital",
    ),
    dict(
        id="T4",
        level=2,
        task="Pairwise Allocation",
        desc="Min-variance 2-asset\nalloc. with return floor",
        ex_q="Min-var: IVV + AGG, return ≥ 0.3%.",
        ex_ctx="ρ(IVV,AGG) = 0.12;  σ_eq = 18%;\nnews: macro risk-off signal",
        ex_a="→  w_eq = 38%,  w_bond = 62%",
    ),
    dict(
        id="T5",
        level=3,
        task="Max-Sharpe Optimization",
        desc="Maximize portfolio Sharpe\nfor 3+ assets",
        ex_q="Maximize Sharpe: IVV, GLD, BTC.",
        ex_ctx="Risk-off regime in headlines;\ncovariance matrix provided",
        ex_a="→  60% / 25% / 15%",
    ),
    dict(
        id="T6",
        level=3,
        task="Rebalancing Decision",
        desc="Detect drift; compute\nrebalancing trade sizes",
        ex_q="IVV drifted to 58% (target: 50%).",
        ex_ctx="Drift threshold: 5%;\ncurrent Eq = 58%,  Bond = 42%",
        ex_a="→  Sell $800 equities → bonds",
    ),
    dict(
        id="T7",
        level=4,
        task="Regime Detection",
        desc="Classify market regime;\nset adaptive weights",
        ex_q="Classify regime; set adaptive weights.",
        ex_ctx="MA crossover: bullish signal;\nnews: strong earnings beats",
        ex_a="→  Bull: Eq 55%, Bd 30%, Cm 15%",
    ),
]

_QA_LEVEL_META = {
    1: dict(
        label="L1  Single-Asset",
        subtitle="Basic forecasting and risk",
        bg="#e4f0fc",
        border="#8ab8e0",
        fg="#1e3d6e",
        card="#f5f9fe",
    ),
    2: dict(
        label="L2  Pairwise",
        subtitle="Two-asset optimization",
        bg="#b8d4ee",
        border="#4a6fa5",
        fg="#1e3d6e",
        card="#ddeaf8",
    ),
    3: dict(
        label="L3  Multi-Asset",
        subtitle="Portfolio construction",
        bg="#4a6fa5",
        border="#2e5f8a",
        fg="#fafafa",
        card="#5b7fb5",
    ),
    4: dict(
        label="L4  Full Portfolio",
        subtitle="Regime-adaptive management",
        bg="#1e3d6e",
        border="#0d2040",
        fg="#fafafa",
        card="#2b4f8a",
    ),
}

# complexity → bar color for bottom stats
_LEVEL_COLORS = {1: "#8ab8e0", 2: "#4a6fa5", 3: "#2e5f8a", 4: "#1e3d6e"}
_SPLIT_COLORS_QA = ["#1e3d6e", "#4a6fa5", "#8ab8e0"]  # train / val / test
_REGIME_COLORS_QA = {"sideways": "#c0c0c0", "bull": "#4a6fa5", "bear": "#1e3d6e"}


def plot_qa_capability_overview(
    stats: dict,
    figsize: tuple = (15, 4.2),
) -> Figure:
    """
    Single-figure QA dataset capability overview.

    Upper section (infographic): 7 template cards arranged left→right by
      complexity level (L1–L4).  Each card shows the template ID, task
      name, 2-line description, and financial-skill keyword tags.

    Lower section (3 subplots):
      (a) Sample count per template colored by complexity level
      (b) Text-context coverage (%) per template, with avg-chars overlay
      (c) Market-regime distribution per template (stacked %)

    Args:
        stats:   stats.json dict from datasets/qa_dataset/stats.json.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    apply_paper_style()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1,
        1,
        left=0.02,
        right=0.98,
        top=0.97,
        bottom=0.03,
    )
    ax_diagram = fig.add_subplot(gs[0, 0])

    # ── 1. Capability architecture diagram ─────────────────────────────
    ax_diagram.set_xlim(0, 7)
    ax_diagram.set_ylim(0, 1)
    ax_diagram.axis("off")

    # ── Level background bands ──────────────────────────────────────────
    level_x = {1: (0, 3), 2: (3, 4), 3: (4, 6), 4: (6, 7)}
    for lvl, (x0, x1) in level_x.items():
        meta = _QA_LEVEL_META[lvl]
        # Background
        ax_diagram.add_patch(
            mpatches.FancyBboxPatch(
                (x0 + 0.04, 0.01),
                (x1 - x0 - 0.08),
                0.98,
                boxstyle="round,pad=0.02",
                facecolor=meta["bg"],
                edgecolor=meta["border"],
                linewidth=1.2,
                zorder=1,
            )
        )
        # Level label (top of band)
        ax_diagram.text(
            (x0 + x1) / 2,
            0.935,
            meta["label"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=meta["fg"],
            zorder=3,
        )
        ax_diagram.text(
            (x0 + x1) / 2,
            0.875,
            meta["subtitle"],
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color=meta["fg"],
            alpha=0.90,
            zorder=3,
        )
        # Separator line below level header
        ax_diagram.plot(
            [x0 + 0.07, x1 - 0.07],
            [0.845, 0.845],
            color=meta["border"],
            linewidth=0.8,
            zorder=2,
        )

    # ── Template cards ─────────────────────────────────────────────────
    for i, tmpl in enumerate(_QA_TEMPLATES):
        meta = _QA_LEVEL_META[tmpl["level"]]
        x_ctr = i + 0.5
        n = stats.get(tmpl["id"], {}).get("n_total", 0)

        # Thin inner card background
        ax_diagram.add_patch(
            mpatches.FancyBboxPatch(
                (i + 0.09, 0.07),
                0.82,
                0.75,
                boxstyle="round,pad=0.015",
                facecolor=meta["card"],
                edgecolor=meta["border"],
                linewidth=0.7,
                alpha=0.55,
                zorder=2,
            )
        )

        # Template ID
        ax_diagram.text(
            x_ctr,
            0.790,
            tmpl["id"],
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=meta["fg"],
            zorder=4,
        )

        # Task name
        ax_diagram.text(
            x_ctr,
            0.710,
            tmpl["task"],
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=meta["fg"],
            zorder=4,
        )

        # Description — bold for L1/L2, normal for L3/L4
        desc_weight = "bold" if tmpl["level"] <= 2 else "normal"
        ax_diagram.text(
            x_ctr,
            0.615,
            tmpl["desc"],
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight=desc_weight,
            color=meta["fg"],
            alpha=0.90,
            linespacing=1.4,
            zorder=4,
        )

        # Thin separator inside card (below description)
        ax_diagram.plot(
            [i + 0.13, i + 0.87],
            [0.545, 0.545],
            color=meta["border"],
            linewidth=0.5,
            alpha=0.50,
            zorder=3,
        )

        # Three sections: Question / Context / Answer
        _sections = [
            ("Question", tmpl["ex_q"], 0.505, 0.455, False, "italic"),
            ("Context", tmpl["ex_ctx"], 0.385, 0.322, False, "normal"),
            ("Answer", tmpl["ex_a"], 0.240, 0.193, True, "normal"),
        ]
        for sec_label, sec_text, lbl_y, txt_y, is_ans, style in _sections:
            ax_diagram.text(
                x_ctr,
                lbl_y,
                sec_label,
                ha="center",
                va="center",
                fontsize=4.8,
                fontweight="bold",
                color=meta["fg"],
                alpha=0.52,
                zorder=4,
            )
            ax_diagram.text(
                x_ctr,
                txt_y,
                sec_text,
                ha="center",
                va="center",
                fontsize=5.5,
                fontweight="bold" if is_ans else "normal",
                fontstyle=style,
                color=meta["fg"],
                alpha=0.90,
                linespacing=1.28,
                zorder=4,
            )

        # Sample count (n=...) only
        ax_diagram.text(
            x_ctr,
            0.113,
            f"n = {n:,}",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color=meta["fg"],
            zorder=4,
        )

        # Vertical dividers between templates in same level
        if i in (1, 2, 4):  # dividers within L1 and L3
            ax_diagram.plot(
                [i + 1, i + 1],
                [0.08, 0.83],
                color=meta["border"],
                linewidth=0.5,
                linestyle=":",
                alpha=0.60,
                zorder=2,
            )

    # Complexity progression arrows
    for x_arr in (3.0, 4.0, 6.0):
        ax_diagram.annotate(
            "",
            xy=(x_arr + 0.05, 0.5),
            xytext=(x_arr - 0.05, 0.5),
            arrowprops=dict(
                arrowstyle="->",
                color="#4a6fa5",
                lw=1.4,
                mutation_scale=10,
            ),
            zorder=5,
        )

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Single-panel statistics figures for raw dataset (portbench.csv)
# ═══════════════════════════════════════════════════════════════════════════════

# Source colors
_SOURCE_COLORS = {
    "Yahoo Finance": "#1e3d6e",
    "FRED": "#4a6fa5",
    "Kaggle": "#7a9fc5",
    "SEC": "#d4e4f7",
}

# Asset class colors (consistent with existing _RAW_ASSET_COLORS)
_ASSET_CLASS_COLORS = {
    "equities": "#1e3d6e",
    "bonds": "#4a6fa5",
    "commodities": "#7a9fc5",
    "cryptocurrency": "#c0c0c0",
    "real_estate": "#d4e4f7",
    "cash": "#8a8a8a",
}

_ASSET_CLASS_LABELS = {
    "equities": "Equities",
    "bonds": "Bonds",
    "commodities": "Commodities",
    "cryptocurrency": "Crypto",
    "real_estate": "Real Estate",
    "cash": "Cash",
}

_ASSET_CLASS_ORDER = ["equities", "bonds", "commodities", "cryptocurrency", "real_estate", "cash"]


def _infer_source(col_name: str) -> str:
    """Infer data source from column name."""
    if col_name == "date":
        return "meta"
    if col_name.endswith("_text_json"):
        return "SEC"
    if "_fred_" in col_name:
        return "FRED"
    if "_kaggle_" in col_name:
        return "Kaggle"
    return "Yahoo Finance"


def _infer_asset_class(col_name: str) -> str | None:
    """Infer asset class from column name. Returns None for 'date'."""
    if col_name == "date":
        return None
    # Handle multi-word prefixes: real_estate_*, cryptocurrency_*
    for prefix in ("real_estate", "cryptocurrency"):
        if col_name.startswith(prefix + "_"):
            return prefix
    return col_name.split("_")[0]


def _classify_columns(columns: list[str]) -> dict:
    """
    Classify each column by source and asset class.

    Returns dict with keys: by_source, by_class, source_class_matrix,
    ticker_set, n_numeric, n_text, n_total.
    """
    by_source: dict[str, list[str]] = {}
    by_class: dict[str, list[str]] = {}
    ticker_set: dict[str, set[str]] = {}
    n_numeric = 0
    n_text = 0

    for col in columns:
        if col == "date":
            continue
        src = _infer_source(col)
        cls = _infer_asset_class(col)
        by_source.setdefault(src, []).append(col)
        if cls:
            by_class.setdefault(cls, []).append(col)
            ticker_raw = col[len(cls) + 1:]
            if src == "SEC":
                ticker_set.setdefault(cls, set()).add(f"{cls}_text")
            else:
                if src == "Yahoo Finance":
                    # ticker_raw e.g. "AAPL_close" or "XRP_USD_close" — chop known suffixes
                    for s in ("_close", "_high", "_low", "_open", "_return", "_volume"):
                        if ticker_raw.endswith(s):
                            ticker = ticker_raw[: -len(s)]
                            break
                    else:
                        ticker = ticker_raw
                elif src == "FRED":
                    ticker = ticker_raw.replace("fred_", "", 1)
                else:
                    ticker = ticker_raw.replace("kaggle_", "", 1)
                ticker_set.setdefault(cls, set()).add(ticker)
        if col.endswith("_text_json"):
            n_text += 1
        else:
            n_numeric += 1

    source_class_matrix: dict[str, dict[str, int]] = {}
    for src, cols in by_source.items():
        class_counts: dict[str, int] = {}
        for col in cols:
            cls = _infer_asset_class(col)
            if cls:
                class_counts[cls] = class_counts.get(cls, 0) + 1
        source_class_matrix[src] = class_counts

    return {
        "by_source": by_source,
        "by_class": by_class,
        "source_class_matrix": source_class_matrix,
        "ticker_set": ticker_set,
        "n_numeric": n_numeric,
        "n_text": n_text,
        "n_total": len(columns) - 1,
    }


def plot_data_sources_pie(
    classification: dict,
    title: str = "Data Sources by Column Count",
    figsize: tuple = (7, 5),
) -> Figure:
    """Pie chart: proportion of columns from each data source."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    by_source = classification["by_source"]
    sources = sorted(by_source.keys(), key=lambda s: len(by_source[s]), reverse=True)
    counts = [len(by_source[s]) for s in sources]
    colors = [_SOURCE_COLORS.get(s, "#cccccc") for s in sources]

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=sources,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    legend_labels = [f"{s}  ({c:,} cols)" for s, c in zip(sources, counts)]
    ax.legend(wedges, legend_labels, fontsize=9, loc="lower center", ncols=2)
    fig.tight_layout()
    return fig


def plot_data_sources_tickers_pie(
    classification: dict,
    title: str = "Data Sources by Unique Ticker Count",
    figsize: tuple = (7, 5),
) -> Figure:
    """Pie chart: proportion of unique tickers from each data source."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    ticker_set = classification["ticker_set"]
    source_ticker_count: dict[str, int] = {}
    for cls, tickers in ticker_set.items():
        for t in tickers:
            if "_fred_" in t or t.startswith("fred_"):
                src = "FRED"
            elif "_kaggle_" in t or t.startswith("kaggle_"):
                src = "Kaggle"
            elif t.endswith("_text"):
                src = "SEC"
            else:
                src = "Yahoo Finance"
            source_ticker_count[src] = source_ticker_count.get(src, 0) + 1

    sources = sorted(source_ticker_count.keys(), key=lambda s: source_ticker_count[s], reverse=True)
    counts = [source_ticker_count[s] for s in sources]
    colors = [_SOURCE_COLORS.get(s, "#cccccc") for s in sources]

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=sources,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    legend_labels = [f"{s}  ({c:,} tickers)" for s, c in zip(sources, counts)]
    ax.legend(wedges, legend_labels, fontsize=9, loc="lower center", ncols=2)
    fig.tight_layout()
    return fig


def plot_asset_class_tickers_bar(
    classification: dict,
    title: str = "Number of Tickers per Asset Class",
    figsize: tuple = (8, 5),
) -> Figure:
    """Horizontal bar chart: unique ticker count per asset class."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    ticker_set = classification["ticker_set"]
    classes = [c for c in _ASSET_CLASS_ORDER if c in ticker_set]
    counts = [len(ticker_set[c]) for c in classes]
    labels = [_ASSET_CLASS_LABELS.get(c, c) for c in classes]
    colors = [_ASSET_CLASS_COLORS.get(c, "#cccccc") for c in classes]

    bars = ax.barh(range(len(classes)), counts, color=colors, alpha=0.88, height=0.6)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Unique Tickers")

    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(n),
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1e3d6e",
        )

    fig.tight_layout()
    return fig


def plot_asset_class_columns_bar(
    classification: dict,
    title: str = "Number of Columns per Asset Class",
    figsize: tuple = (8, 5),
) -> Figure:
    """Horizontal bar chart: total column count per asset class."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    by_class = classification["by_class"]
    classes = [c for c in _ASSET_CLASS_ORDER if c in by_class]
    counts = [len(by_class[c]) for c in classes]
    labels = [_ASSET_CLASS_LABELS.get(c, c) for c in classes]
    colors = [_ASSET_CLASS_COLORS.get(c, "#cccccc") for c in classes]

    bars = ax.barh(range(len(classes)), counts, color=colors, alpha=0.88, height=0.6)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Columns")

    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(n),
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1e3d6e",
        )

    fig.tight_layout()
    return fig


def plot_numeric_vs_text_pie(
    classification: dict,
    title: str = "Numeric vs Text Columns",
    figsize: tuple = (6, 5),
) -> Figure:
    """Pie chart: proportion of numeric vs text columns."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    labels = ["Numeric", "Text (News/Filings)"]
    counts = [classification["n_numeric"], classification["n_text"]]
    colors = ["#1e3d6e", "#d4e4f7"]

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.70,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    legend_labels = [f"{l}  ({c:,} cols)" for l, c in zip(labels, counts)]
    ax.legend(wedges, legend_labels, fontsize=9)
    fig.tight_layout()
    return fig


def plot_source_by_asset_class_stacked(
    classification: dict,
    title: str = "Data Sources per Asset Class",
    figsize: tuple = (10, 5),
) -> Figure:
    """Stacked bar chart: number of columns from each source per asset class."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    matrix = classification["source_class_matrix"]
    class_order = [c for c in _ASSET_CLASS_ORDER if c in classification["by_class"]]
    sources = sorted(
        {s for m in matrix.values() for s in m},
        key=lambda s: sum(m.get(s, 0) for m in matrix.values()),
        reverse=True,
    )

    x = np.arange(len(class_order))
    width = 0.55
    bottoms = np.zeros(len(class_order))

    for src in sources:
        counts = [matrix.get(src, {}).get(cls, 0) for cls in class_order]
        color = _SOURCE_COLORS.get(src, "#cccccc")
        bars = ax.bar(
            x, counts, width, bottom=bottoms, label=src, color=color, alpha=0.85
        )
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(c),
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="white" if src in ("Yahoo Finance", "FRED") else "#1e3d6e",
                )
        bottoms += np.array(counts)

    ax.set_xticks(x)
    ax.set_xticklabels([_ASSET_CLASS_LABELS.get(c, c) for c in class_order])
    ax.set_ylabel("Number of Columns")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def plot_raw_time_coverage(
    df: pd.DataFrame,
    classification: dict,
    title: str = "Time Range Coverage per Asset Class",
    figsize: tuple = (10, 5),
) -> Figure:
    """Horizontal bar chart: date range coverage for each asset class."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    class_order = [c for c in _ASSET_CLASS_ORDER if c in classification["by_class"]]

    ranges = []
    for cls in class_order:
        cols = classification["by_class"][cls]
        numeric_cols = [c for c in cols if not c.endswith("_text_json")]
        if numeric_cols:
            cls_df = df[numeric_cols].dropna(how="all")
            if not cls_df.empty:
                start = cls_df.index.min()
                end = cls_df.index.max()
                ranges.append((cls, start, end, len(numeric_cols)))
            else:
                ranges.append((cls, pd.NaT, pd.NaT, 0))
        else:
            ranges.append((cls, pd.NaT, pd.NaT, 0))

    ys = range(len(class_order))
    for i, (cls, start, end, n_cols) in enumerate(ranges):
        color = _ASSET_CLASS_COLORS.get(cls, "#cccccc")
        if pd.notna(start) and pd.notna(end):
            ax.barh(
                i, (end - start).days, left=start, height=0.55, color=color, alpha=0.85
            )
            ax.text(
                start + (end - start) / 2,
                i,
                f"{start.strftime('%Y-%m')}  →  {end.strftime('%Y-%m')}  ({n_cols} cols)",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )

    ax.set_yticks(range(len(class_order)))
    ax.set_yticklabels([_ASSET_CLASS_LABELS.get(c, c) for c in class_order])
    ax.set_xlabel("Date")
    fig.tight_layout()
    return fig


def plot_raw_missing_pct(
    df: pd.DataFrame,
    classification: dict,
    title: str = "Data Completeness per Asset Class",
    figsize: tuple = (8, 5),
) -> Figure:
    """Horizontal bar chart: percentage of non-null values per asset class."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    class_order = [c for c in _ASSET_CLASS_ORDER if c in classification["by_class"]]

    completeness = []
    for cls in class_order:
        cols = classification["by_class"][cls]
        numeric_cols = [c for c in cols if not c.endswith("_text_json")]
        if numeric_cols:
            pct = df[numeric_cols].notna().mean().mean() * 100
            completeness.append((cls, pct, len(numeric_cols)))

    classes = [c for c, _, _ in completeness]
    pcts = [p for _, p, _ in completeness]
    n_cols = [n for _, _, n in completeness]
    colors = [_ASSET_CLASS_COLORS.get(c, "#cccccc") for c in classes]

    bars = ax.barh(range(len(classes)), pcts, color=colors, alpha=0.88, height=0.6)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([_ASSET_CLASS_LABELS.get(c, c) for c in classes])
    ax.invert_yaxis()
    ax.set_xlabel("Data Completeness (%)")
    ax.set_xlim(0, 105)

    for bar, p, n in zip(bars, pcts, n_cols):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.1f}%  ({n} cols)",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#1e3d6e",
        )

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Single-panel statistics figures for QA dataset
# ═══════════════════════════════════════════════════════════════════════════════

_QA_REGIME_COLORS_BAR = {
    "sideways": "#c0c0c0",
    "bull": "#4a6fa5",
    "bear": "#1e3d6e",
}
_QA_SPLIT_COLORS_BAR = {
    "train": "#1e3d6e",
    "val": "#4a6fa5",
    "test": "#7a9fc5",
}


def plot_qa_template_counts_bar(
    stats: dict,
    title: str = "QA Sample Count per Template",
    figsize: tuple = (8, 4.5),
) -> Figure:
    """Bar chart: number of QA pairs per template."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    templates = [t for t in _TEMPLATE_IDS if t in stats]
    counts = [stats[t]["n_total"] for t in templates]
    complexity = [_COMPLEXITY.get(t, 1) for t in templates]
    bar_colors = [_LEVEL_COLORS.get(c, "#cccccc") for c in complexity]

    bars = ax.bar(templates, counts, color=bar_colors, alpha=0.88)
    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.015,
            str(n),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#1e3d6e",
        )

    ax.set_ylabel("Number of QA Pairs")
    fig.tight_layout()
    return fig


def plot_qa_template_by_split_bar(
    stats: dict,
    title: str = "QA Sample Count by Template and Split",
    figsize: tuple = (9, 4.5),
) -> Figure:
    """Grouped bar chart: QA pairs per template, colored by train/val/test."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    templates = [t for t in _TEMPLATE_IDS if t in stats]
    x = np.arange(len(templates))
    width = 0.25

    for i, split in enumerate(_SPLITS):
        counts = [stats[t]["by_split"].get(split, 0) for t in templates]
        bars = ax.bar(
            x + (i - 1) * width,
            counts,
            width,
            color=_QA_SPLIT_COLORS_BAR[split],
            label=split.capitalize(),
            alpha=0.88,
        )
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    str(c),
                    ha="center",
                    fontsize=7,
                    fontweight="bold",
                    color="#1e3d6e",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(templates)
    ax.set_ylabel("Number of QA Pairs")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_qa_template_by_regime_bar(
    stats: dict,
    title: str = "QA Sample Count by Template and Market Regime",
    figsize: tuple = (9, 4.5),
) -> Figure:
    """Grouped bar chart: QA pairs per template, colored by market regime."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    templates = [t for t in _TEMPLATE_IDS if t in stats]
    x = np.arange(len(templates))
    width = 0.25

    for i, regime in enumerate(_REGIMES):
        counts = [stats[t]["by_regime"].get(regime, 0) for t in templates]
        bars = ax.bar(
            x + (i - 1) * width,
            counts,
            width,
            color=_QA_REGIME_COLORS_BAR[regime],
            label=regime.capitalize(),
            alpha=0.88,
        )
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    str(c),
                    ha="center",
                    fontsize=7,
                    fontweight="bold",
                    color="#1e3d6e",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(templates)
    ax.set_ylabel("Number of QA Pairs")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_qa_regime_pie(
    stats: dict,
    title: str = "Overall Market Regime Distribution",
    figsize: tuple = (6, 5),
) -> Figure:
    """Pie chart: overall market regime distribution across all QA pairs."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    regime_totals: dict[str, int] = {}
    for t in _TEMPLATE_IDS:
        if t not in stats:
            continue
        for r, cnt in stats[t]["by_regime"].items():
            regime_totals[r] = regime_totals.get(r, 0) + cnt

    labels = list(regime_totals.keys())
    counts = list(regime_totals.values())
    colors = [_QA_REGIME_COLORS_BAR.get(l, "#cccccc") for l in labels]
    sorted_idx = np.argsort(counts)[::-1]
    labels = [labels[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=[l.capitalize() for l in labels],
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    legend_labels = [f"{l.capitalize()}  ({c:,} pairs)" for l, c in zip(labels, counts)]
    ax.legend(wedges, legend_labels, fontsize=9, loc="lower center", ncols=2)
    fig.tight_layout()
    return fig


def plot_qa_complexity_bar(
    stats: dict,
    title: str = "QA Pairs by Complexity Level",
    figsize: tuple = (7, 4.5),
) -> Figure:
    """Bar chart: total QA pairs per complexity level (L1–L4)."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    level_totals: dict[int, int] = {}
    for t in _TEMPLATE_IDS:
        if t not in stats:
            continue
        lvl = _COMPLEXITY.get(t, 1)
        level_totals[lvl] = level_totals.get(lvl, 0) + stats[t]["n_total"]

    levels = sorted(level_totals.keys())
    counts = [level_totals[l] for l in levels]
    labels = [_COMPLEXITY_LABEL.get(l, f"L{l}") for l in levels]
    colors = [_LEVEL_COLORS.get(l, "#cccccc") for l in levels]

    bars = ax.bar(range(len(levels)), counts, color=colors, alpha=0.88)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Total QA Pairs")

    for bar, n, lvl in zip(bars, counts, levels):
        n_tmpl = len([t for t in _TEMPLATE_IDS if t in stats and _COMPLEXITY.get(t, 1) == lvl])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.015,
            f"{n:,}  ({n_tmpl} templates)",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#1e3d6e",
        )

    fig.tight_layout()
    return fig


def plot_qa_text_coverage_bar(
    stats: dict,
    title: str = "Text Context Coverage per Template",
    figsize: tuple = (8, 4.5),
) -> Figure:
    """Bar chart: percentage of QA pairs with news/filing text per template."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    templates = [t for t in _TEMPLATE_IDS if t in stats]
    pcts = [stats[t]["text"]["pct_with_text"] for t in templates]
    bar_colors = ["#4a6fa5"] * len(templates)

    bars = ax.bar(templates, pcts, color=bar_colors, alpha=0.85)
    ax.set_ylabel("% with News / Filing Text")
    ax.set_ylim(0, 110)

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{pct:.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#1e3d6e",
        )

    overall = stats.get("_meta", {}).get("text_overall", {})
    if overall:
        overall_pct = overall.get("pct_with_text", 0)
        ax.axhline(overall_pct, color="#7a9fc5", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(
            len(templates) - 0.3,
            overall_pct + 1.5,
            f"Overall: {overall_pct:.1f}%",
            fontsize=8,
            color="#7a9fc5",
            ha="right",
        )

    fig.tight_layout()
    return fig


def plot_qa_context_length_bar(
    stats: dict,
    title: str = "Average Context Length per Template",
    figsize: tuple = (8, 4.5),
) -> Figure:
    """Bar chart: average context length (chars) per template."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    templates = [t for t in _TEMPLATE_IDS if t in stats]
    avg_chars = [stats[t]["text"]["avg_chars"] for t in templates]
    bar_colors = ["#1e3d6e"] * len(templates)

    bars = ax.bar(templates, avg_chars, color=bar_colors, alpha=0.85)
    ax.set_ylabel("Average Context Length (chars)")

    for bar, n in zip(bars, avg_chars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(avg_chars) * 0.015,
            f"{n:,.0f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1e3d6e",
        )

    overall = stats.get("_meta", {}).get("text_overall", {})
    if overall:
        overall_avg = overall.get("avg_chars", 0)
        ax.axhline(overall_avg, color="#7a9fc5", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(
            len(templates) - 0.3,
            overall_avg + max(avg_chars) * 0.015,
            f"Overall: {overall_avg:,.0f}",
            fontsize=8,
            color="#7a9fc5",
            ha="right",
        )

    fig.tight_layout()
    return fig


def plot_qa_split_pie(
    stats: dict,
    title: str = "Train / Val / Test Split Distribution",
    figsize: tuple = (6, 5),
) -> Figure:
    """Pie chart: overall train/val/test split distribution."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    split_totals: dict[str, int] = {}
    for t in _TEMPLATE_IDS:
        if t not in stats:
            continue
        for s, cnt in stats[t]["by_split"].items():
            split_totals[s] = split_totals.get(s, 0) + cnt

    labels = list(split_totals.keys())
    counts = list(split_totals.values())
    colors = [_QA_SPLIT_COLORS_BAR.get(l, "#cccccc") for l in labels]

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=[l.capitalize() for l in labels],
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    legend_labels = [f"{l.capitalize()}  ({c:,} pairs)" for l, c in zip(labels, counts)]
    ax.legend(wedges, legend_labels, fontsize=9)
    fig.tight_layout()
    return fig


def plot_qa_template_by_split_stacked(
    stats: dict,
    figsize: tuple = (8, 5),
) -> Figure:
    """Stacked bar chart: QA pairs per template, colored by train/val/test."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    templates = [t for t in _TEMPLATE_IDS if t in stats]
    bottoms = np.zeros(len(templates))
    for i, split in enumerate(_SPLITS):
        counts = [stats[t]["by_split"].get(split, 0) for t in templates]
        ax.bar(
            templates,
            counts,
            bottom=bottoms,
            color=_SPLIT_COLORS[i],
            label=split.capitalize(),
            alpha=0.88,
        )
        bottoms += np.array(counts)

    totals = [stats[t]["n_total"] for t in templates]
    for j, (_, tot) in enumerate(zip(templates, totals)):
        ax.text(
            j, tot + 10, str(tot),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="#1e3d6e",
        )

    ax.set_ylabel("Number of QA Pairs")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def plot_qa_text_richness(
    stats: dict,
    figsize: tuple = (8, 5),
) -> Figure:
    """Dual-axis chart: % with news/filing text (bars) + avg context length (line)."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    templates = [t for t in _TEMPLATE_IDS if t in stats]
    pct_text = [stats[t]["text"]["pct_with_text"] for t in templates]
    avg_chars = [stats[t]["text"]["avg_chars"] for t in templates]

    bars = ax.bar(
        templates, pct_text, color=_AF1, alpha=0.80, label="% with news/filing"
    )
    ax.set_ylabel("% Samples with News / Filing Text")
    ax.set_ylim(0, 115)
    for bar, pct in zip(bars, pct_text):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{pct:.0f}%",
            ha="center", va="bottom", fontsize=8, color="#1e3d6e",
        )

    ax2 = ax.twinx()
    ax2.plot(
        templates, avg_chars, "o--",
        color="#1e3d6e", linewidth=1.8, markersize=6,
        label="Avg context length (chars)",
    )
    ax2.set_ylabel("Avg Context Length (chars)", color="#1e3d6e")
    ax2.tick_params(axis="y", labelcolor="#1e3d6e")

    lines1 = [mpatches.Patch(color=_AF1, label="% with news")]
    lines2 = [mlines.Line2D([0], [0], color="#1e3d6e", marker="o", linestyle="--", label="Avg chars")]
    ax.legend(handles=lines1 + lines2, fontsize=9, loc="upper left")

    overall = stats.get("_meta", {}).get("text_overall", {})
    if overall:
        info = (
            f"Overall: {overall.get('pct_with_text', 0):.0f}% text, "
            f"{overall.get('avg_chars', 0):,.0f} avg chars"
        )
        ax.text(
            0.98, 0.05, info,
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85),
        )

    fig.tight_layout()
    return fig
