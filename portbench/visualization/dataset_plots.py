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

_CRISES = [
    ("2015-08-01", "2015-09-30", "2015 China Shock"),
    ("2020-02-15", "2020-04-20", "2020 COVID Crash"),
    ("2022-01-01", "2022-12-31", "2022 Bear Market"),
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
    figsize: tuple = (11, 5),
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
    _RATE_LS = ["--", ":"]
    for idx, (key, series) in enumerate(rate_series.items()):
        ax_rate.plot(
            series.index,
            series.values,
            label=_RAW_ASSET_LABELS[key],
            color=_RAW_ASSET_COLORS[key],
            linestyle=_RATE_LS[idx % len(_RATE_LS)],
            linewidth=1.2,
            alpha=0.85,
            zorder=2,
        )

    ax_top.axhline(100, color="#c0c0c0", linestyle=":", linewidth=0.7, alpha=0.5)
    ax_top.set_ylabel("Normalized Index  (base = 100)", fontsize=10)
    ax_rate.set_ylabel("Interest Rate  (%)", fontsize=10, color="#4a6fa5")
    ax_rate.tick_params(axis="y", colors="#4a6fa5", labelsize=9)
    ax_top.set_xlim(df.index.min(), df.index.max())

    # ── ylim: positive region only ───────────────────────────────────────
    all_price_vals = [s.dropna().max() for s in price_series.values()]
    price_max = max(all_price_vals) if all_price_vals else 400
    price_top = price_max * 1.06
    ax_top.set_ylim(0, price_top)
    ax_rate.set_ylim(0, 9.0)

    # ── News character count curve (second right axis) ────────────────────
    chars_idx = chars_eq.index.union(chars_cry.index)
    chars_total = chars_eq.reindex(chars_idx, fill_value=0) + chars_cry.reindex(
        chars_idx, fill_value=0
    )

    if len(chars_idx) > 0 and chars_total.max() > 0:
        chars_max = float(chars_total.max())
        ax_chars.set_ylim(0, chars_max * 1.08)

        ax_chars.spines["right"].set_position(("outward", 58))
        ax_chars.spines["right"].set_visible(True)
        for sp in ["left", "top", "bottom"]:
            ax_chars.spines[sp].set_visible(False)
        ax_chars.yaxis.set_label_position("right")
        ax_chars.yaxis.tick_right()
        ax_chars.tick_params(axis="y", colors="#7a9fc5", labelsize=9)
        ax_chars.set_ylabel("News chars / month", fontsize=10, color="#7a9fc5")
        ax_chars.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")
        )

        ax_chars.plot(
            chars_idx,
            chars_total.values,
            color="#7a9fc5",
            linewidth=1.4,
            linestyle="-.",
            alpha=0.85,
            zorder=4,
        )
    else:
        ax_chars.set_visible(False)

    # ── Crisis shading ───────────────────────────────────────────────────
    for start_s, end_s, label in _CRISES:
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        ax_top.axvspan(start, end, color="#d4e4f7", alpha=0.45, zorder=0)
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
            color=_RAW_ASSET_COLORS[k],
            linestyle=_RATE_LS[i % len(_RATE_LS)],
            linewidth=1.3,
            label=_RAW_ASSET_LABELS[k],
        )
        for i, k in enumerate(rate_series)
    ]
    crisis_patch = mpatches.Patch(
        color="#d4e4f7", alpha=0.7, label="Market Stress Window"
    )
    news_chars_line = mlines.Line2D(
        [],
        [],
        color="#7a9fc5",
        linestyle="-.",
        linewidth=1.4,
        label="News chars / month",
    )
    ax_top.legend(
        handles=price_handles + rate_handles + [crisis_patch, news_chars_line],
        fontsize=9,
        loc="upper left",
        framealpha=0.9,
        ncols=2,
    )

    # ── Dataset stats — below legend (left side) ─────────────────────────
    n_rows = len(df)
    n_cols = len(df.columns)
    date_min = df.index.min().strftime("%Y-%m")
    date_max = df.index.max().strftime("%Y-%m")
    has_news = (
        df.get("equities_text_json", pd.Series(dtype=str)).notna()
        | df.get("cryptocurrency_text_json", pd.Series(dtype=str)).notna()
    )
    pct_news = has_news.mean() * 100
    stats_text = (
        f"  {n_rows:,} trading days\n"
        f"  {n_cols:,} features\n"
        f"  {date_min} – {date_max}\n"
        f"  6 asset classes\n"
        f"  {pct_news:.0f}% days with news"
    )
    ax_top.text(
        0.013,
        0.73,
        stats_text,
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#1e3d6e",
        linespacing=1.6,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#8ab8e0",
            linewidth=0.8,
            alpha=0.90,
        ),
        zorder=7,
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
