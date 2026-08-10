"""
Cross-period Sharpe vs equal-weight visualization.

Figure: plot_cross_period_vs_ew
  1x4 panels (2015-16 / 2020 / 2022 / 2024) for full-width (figure*) layout.
  Fixed model row order; each row: range bar over three profiles + mean tick.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.figure import Figure

from .style import apply_paper_style, abbrev_model_name

# Fixed display order (matches paper naming conventions)
DEFAULT_MODEL_ORDER = [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "dashscope/qwen3.6-plus",
    "dashscope/qwen3.7-max",
    "dashscope/qwen3.6-35b-a3b",
    "dashscope/glm-5.1",
    "ark/doubao-seed-2-0-lite",
    "ark/doubao-seed-2-0-pro",
    "tencent/hy3-preview",
    "dashscope/kimi-k2.6",
]

PERIOD_LABELS = {
    "2015-16": "2015–16 Volatile",
    "2020": "2020 COVID",
    "2022": "2022 Bear",
    "2024": "2024 Bull",
}
PERIOD_PANEL = {
    "2015-16": "a",
    "2020": "b",
    "2022": "c",
    "2024": "d",
}

PERIOD_ORDER = ["2015-16", "2020", "2022", "2024"]
PROFILE_ORDER = ["conservative", "balanced", "aggressive"]
PROFILE_MARKERS = {"conservative": "o", "balanced": "s", "aggressive": "^"}
# Okabe–Ito inspired (print-friendly, colorblind-safe)
PROFILE_COLORS = {
    "conservative": "#0072B2",
    "balanced": "#E69F00",
    "aggressive": "#009E73",
}
# Small vertical offsets so near-identical profile points remain separable
_PROFILE_JITTER = {
    "conservative": -0.14,
    "balanced": 0.0,
    "aggressive": 0.14,
}

_EQW = "#4A4A4A"
_RANGE = "#9AA0A6"
_MEAN = "#1A1A1A"
_POS_BAND = "#EEF1F4"  # cool grey tint for Δ > 0
_INK = "#1A1A1A"
_MUTED = "#6B6B6B"


def _normalize_model_key(key: str) -> str:
    """Strip trailing date suffixes used in experiment folder names."""
    import re

    return re.sub(r"-\d{6,8}$", "", key)


def plot_cross_period_vs_ew(
    data: dict[str, dict[str, dict[str, float]]],
    eqw_sharpe: dict[str, float],
    title: str = "",
    figsize: tuple = (11.0, 3.95),
    model_order: list[str] | None = None,
) -> Figure:
    """
    Plot Sharpe − EqW across four periods in a 1×4 row (full-width figure*).

    Args:
        data: {model_key: {period: {profile: sharpe}}}
        eqw_sharpe: {period: eqw_sharpe}
        model_order: fixed row order (normalized keys); defaults to DEFAULT_MODEL_ORDER
    """
    apply_paper_style()

    # Normalize keys
    norm_data: dict[str, dict[str, dict[str, float]]] = {}
    for mk, periods in data.items():
        nk = _normalize_model_key(mk)
        if nk not in norm_data:
            norm_data[nk] = periods
        else:
            for per, profs in periods.items():
                norm_data[nk].setdefault(per, {}).update(profs)

    order = list(model_order or DEFAULT_MODEL_ORDER)
    present = [m for m in order if m in norm_data]
    extras = sorted(m for m in norm_data if m not in present)
    models = present + extras
    n_models = len(models)
    if n_models == 0:
        raise ValueError("No models to plot")

    # Beat rates per period (model × profile)
    beat_rates: dict[str, float] = {}
    for per in PERIOD_ORDER:
        n_beat, n_tot = 0, 0
        eqw = eqw_sharpe.get(per, 0.0)
        for mk in models:
            for prof in PROFILE_ORDER:
                sh = norm_data.get(mk, {}).get(per, {}).get(prof)
                if sh is None:
                    continue
                n_tot += 1
                if sh > eqw:
                    n_beat += 1
        beat_rates[per] = (100.0 * n_beat / n_tot) if n_tot else 0.0

    # Serif to match ACL body text; quiet default frost grid
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "axes.titleweight": "normal",
        "axes.grid": False,
        "axes.linewidth": 1.05,
        "axes.edgecolor": _INK,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }):
        fig, axes = plt.subplots(
            1, 4, figsize=figsize, sharey=True,
            gridspec_kw={"wspace": 0.08},
            constrained_layout=False,
        )

        y = np.arange(n_models)
        labels = [abbrev_model_name(m) for m in models]

        for idx, (ax, per) in enumerate(zip(axes, PERIOD_ORDER)):
            ax.set_facecolor("white")
            eqw = eqw_sharpe.get(per, 0.0)
            deltas_by_model: list[list[float]] = []
            for mk in models:
                dels = []
                for prof in PROFILE_ORDER:
                    sh = norm_data.get(mk, {}).get(per, {}).get(prof)
                    if sh is not None:
                        dels.append(sh - eqw)
                deltas_by_model.append(dels)

            all_dels = [d for dels in deltas_by_model for d in dels]
            if all_dels:
                span = max(all_dels) - min(all_dels) + 1e-6
                pad = 0.10 * span
                xmin, xmax = min(all_dels) - pad, max(all_dels) + pad
                xmin = min(xmin, -0.02)
                xmax = max(xmax, 0.02)
            else:
                xmin, xmax = -1.0, 1.0

            ax.axvspan(0.0, xmax, color=_POS_BAND, alpha=0.85, zorder=0, linewidth=0)
            ax.axvline(0.0, color=_EQW, linestyle=(0, (3.5, 2.2)), linewidth=1.15, zorder=1)

            for i, dels in enumerate(deltas_by_model):
                if not dels:
                    continue
                lo, hi = min(dels), max(dels)
                mean_d = float(np.mean(dels))
                ax.plot(
                    [lo, hi], [i, i],
                    color=_RANGE, linewidth=1.85, solid_capstyle="round",
                    zorder=2, alpha=0.95,
                )
                ax.plot(
                    [mean_d, mean_d], [i - 0.28, i + 0.28],
                    color=_MEAN, linewidth=1.55, solid_capstyle="butt", zorder=4,
                )

                for prof in PROFILE_ORDER:
                    sh = norm_data.get(models[i], {}).get(per, {}).get(prof)
                    if sh is None:
                        continue
                    ax.scatter(
                        sh - eqw,
                        i + _PROFILE_JITTER[prof],
                        marker=PROFILE_MARKERS[prof],
                        c=PROFILE_COLORS[prof],
                        s=72,
                        edgecolors="white",
                        linewidths=1.05,
                        zorder=5,
                    )

            # Centered panel title with beat-rate in parentheses
            letter = PERIOD_PANEL[per]
            rate = beat_rates.get(per, 0.0)
            ax.set_title(
                f"({letter})  {PERIOD_LABELS[per]}  ({rate:.0f}%)",
                loc="center",
                fontsize=11.5,
                fontweight="bold",
                color="black",
                pad=8,
            )

            # Narrow-scale callout for the bull panel (axis range ≪ other panels)
            if per == "2024":
                ax.text(
                    0.97, 0.06,
                    "narrow scale",
                    transform=ax.transAxes,
                    ha="right", va="bottom",
                    fontsize=9.0, color=_MUTED, style="italic",
                )

            ax.set_yticks(y)
            if idx == 0:
                ax.set_yticklabels(labels, fontsize=11.5)
                ax.tick_params(axis="y", length=0, pad=3)
            else:
                ax.tick_params(axis="y", length=0, labelleft=False)
            ax.set_xlabel("")  # single shared label below
            ax.tick_params(axis="x", labelsize=10.0, length=3.5, pad=2)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(n_models - 0.55, -0.55)

            for spine in ("left", "bottom"):
                ax.spines[spine].set_linewidth(1.05)
                ax.spines[spine].set_color(_INK)
            # Soft vertical guides only (no full grid)
            ax.xaxis.grid(True, linestyle="-", linewidth=0.35, alpha=0.22, color="#B8B8B8", zorder=0.5)
            ax.yaxis.grid(False)
            ax.set_axisbelow(True)

        # Shared x-axis label (once); shading / % explained in the LaTeX caption
        fig.text(
            0.55, 0.10,
            "Sharpe − EqW",
            ha="center", va="center",
            fontsize=11.5, color="black",
        )

        handles = [
            mlines.Line2D(
                [], [], color=PROFILE_COLORS[p], marker=PROFILE_MARKERS[p], linestyle="None",
                markersize=9.0, markeredgecolor="white", markeredgewidth=1.0,
                label=p.capitalize(),
            )
            for p in PROFILE_ORDER
        ]
        handles.append(
            mlines.Line2D([], [], color=_MEAN, linewidth=1.55, label="Mean")
        )
        handles.append(
            mlines.Line2D(
                [], [], color=_EQW, linestyle=(0, (3.5, 2.2)),
                linewidth=1.15, label="EqW (Δ=0)",
            )
        )
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=5,
            fontsize=10.5,
            frameon=False,
            bbox_to_anchor=(0.55, -0.015),
            handletextpad=0.45,
            columnspacing=1.35,
        )
        if title:
            fig.suptitle(title, fontsize=13, fontweight="normal", y=1.06)
        fig.subplots_adjust(left=0.10, right=0.995, top=0.88, bottom=0.20, wspace=0.10)
        return fig
