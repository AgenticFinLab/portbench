"""
Sandbox backtest visualization.

Figure 8  — plot_sandbox_nav:        NAV curve comparison across models/profiles.
Figure 9  — plot_sandbox_metrics:    Bar chart of key performance metrics.
Figure 10 — plot_ceps_vs_pnl:        CEPS score vs. realized return scatter.
Figure 13 — plot_stress_drawdown:    Heatmap of per-scenario max-drawdown by profile.
Figure 14 — plot_profile_nav:        NAV curves for all three profiles of one model.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from .style import (
    apply_paper_style,
    MODEL_PALETTE,
    PAPER_COLORS,
    LINE_STYLES,
    LINE_MARKERS,
    abbrev_model_name,
    NAV_LLM_PALETTE,
    NAV_BASELINE_PALETTE,
)
from .risk_return_plots import _MODEL_COLOURS
import matplotlib.colors as mcolors


def _desaturate(hex_color: str, sat_delta: float = -0.1, val_delta: float = 0.1) -> str:
    rgb = mcolors.hex2color(hex_color)
    hsv = list(mcolors.rgb_to_hsv(rgb))

    hsv[1] = max(0.0, min(1.0, hsv[1] + sat_delta))

    hsv[2] = max(0.0, min(1.0, hsv[2] + val_delta))

    return mcolors.to_hex(mcolors.hsv_to_rgb(hsv))


_MODEL_COLOURS_SOFT = [_desaturate(c) for c in _MODEL_COLOURS]


def plot_sandbox_nav(
    nav_results: dict[str, pd.Series],
    title: str = "Portfolio NAV Curves",
    figsize: tuple = (7.2, 4.2),
    ceps_data: dict[str, pd.Series] | None = None,
    legend_loc: str = "upper left",
    legend_bbox: tuple[float, float] | None = None,
    legend_col_align: str = "top",
    show_title: bool = False,
    event_markers: list[tuple[str, str]] | None = None,
) -> Figure:
    """
    NAV curve comparison (optional bottom CEPS panel).

    LLM models: solid lines + distinct markers.
    Baselines:  dashed grey lines.
    """
    apply_paper_style()
    import matplotlib.lines as mlines

    _FS = 6.5  # unify axis / tick / legend / annotation text size
    ink = "#1A1A1A"
    # Distinct LLM colours (print-friendly); baselines stay grey dashed
    _llm_palette = [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
        "#56B4E9", "#882255", "#44AA99", "#332288", "#117733",
    ]
    _MARKERS = ["o", "s", "^", "D", "v", "P", "h", "*"]

    has_ceps = bool(ceps_data)
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "axes.grid": False,
        "axes.linewidth": 1.05,
        "axes.edgecolor": ink,
    }):
        if has_ceps:
            fig, (ax_nav, ax_ceps) = plt.subplots(
                2, 1, figsize=figsize, sharex=True,
                gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
            )
        else:
            fig, ax_nav = plt.subplots(figsize=figsize)
            ax_ceps = None

        llm_items = [
            (n, s) for n, s in nav_results.items() if not n.startswith("baseline/")
        ]
        base_items = [
            (n, s) for n, s in nav_results.items() if n.startswith("baseline/")
        ]

        # Stable color order by display name
        _llm_keys = sorted((n for n, _ in llm_items), key=abbrev_model_name)
        _llm_color_map = {
            k: _llm_palette[i % len(_llm_palette)] for i, k in enumerate(_llm_keys)
        }
        _llm_marker_map = {
            k: _MARKERS[i % len(_MARKERS)] for i, k in enumerate(_llm_keys)
        }

        llm_handles, llm_colors = [], {}
        for name, nav in llm_items:
            color = _llm_color_map[name]
            marker = _llm_marker_map[name]
            nav_norm = nav / nav.iloc[0] * 100
            label = abbrev_model_name(name)
            ax_nav.plot(
                nav_norm.index, nav_norm.values,
                color=color, linewidth=1.15, linestyle="-",
                solid_capstyle="round",
                label=label, zorder=3, alpha=0.88,
            )
            # Thin proxy for legend (avoids oversized legend strokes)
            llm_handles.append(
                mlines.Line2D(
                    [], [], color=color, linewidth=1.15, linestyle="-",
                    label=label,
                )
            )
            llm_colors[name] = color

        llm_handles = sorted(llm_handles, key=lambda h: h.get_label())

        base_handles = []
        for i, (name, nav) in enumerate(base_items):
            color = NAV_BASELINE_PALETTE[i % len(NAV_BASELINE_PALETTE)]
            nav_norm = nav / nav.iloc[0] * 100
            label = abbrev_model_name(name)
            ax_nav.plot(
                nav_norm.index, nav_norm.values,
                color=color, linewidth=1.0, linestyle=(0, (3.0, 1.6)),
                label=label, zorder=2, alpha=0.80,
            )
            base_handles.append(
                mlines.Line2D(
                    [], [], color=color, linewidth=1.0,
                    linestyle=(0, (3.0, 1.6)), label=label,
                )
            )
        base_handles = sorted(base_handles, key=lambda h: h.get_label())

        ax_nav.axhline(100, color="#9A9A9A", linestyle=":", linewidth=0.9, alpha=0.75, zorder=1)

        # Optional dated event markers: light band + top horizontal label
        if event_markers:
            for date_str, label in event_markers:
                t = pd.Timestamp(date_str)
                ax_nav.axvline(
                    t, color="#B03A2E", linestyle="-",
                    linewidth=1.0, alpha=0.55, zorder=1.5,
                )
                ax_nav.annotate(
                    label,
                    xy=(t, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=_FS, color="#7B241C",
                    annotation_clip=False,
                    zorder=5,
                )

        ax_nav.set_ylabel("Normalized NAV (base=100)", fontsize=_FS, color="black")
        ax_nav.tick_params(axis="both", labelsize=_FS, length=3.5, pad=2)
        ax_nav.yaxis.grid(True, linestyle="-", linewidth=0.35, alpha=0.25, color="#B8B8B8", zorder=0.5)
        ax_nav.set_axisbelow(True)
        for spine in ("top", "right"):
            ax_nav.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax_nav.spines[spine].set_linewidth(1.05)
            ax_nav.spines[spine].set_color(ink)

        # matplotlib ncol fills column-major: col0 top→bottom, then col1
        spacer = mlines.Line2D([], [], color="none", label=" ")
        llm_title = mlines.Line2D([], [], color="none", label="LLMs")
        base_title = mlines.Line2D([], [], color="none", label="Baselines")
        n_llm, n_base = len(llm_handles), len(base_handles)
        col1 = [llm_title] + llm_handles
        col2 = [base_title] + base_handles
        pad = abs(n_llm - n_base)
        if legend_col_align == "bottom":
            # Pad the shorter column at the top so both columns end together
            if n_llm > n_base:
                col2 = [spacer] * pad + col2
            elif n_base > n_llm:
                col1 = [spacer] * pad + col1
        else:
            if n_llm > n_base:
                col2 = col2 + [spacer] * pad
            elif n_base > n_llm:
                col1 = col1 + [spacer] * pad
        handles_2col = col1 + col2
        leg_kw = dict(
            handles=handles_2col,
            fontsize=_FS,
            loc=legend_loc,
            frameon=True,
            fancybox=False,
            framealpha=0.92,
            edgecolor="#D0D0D0",
            borderpad=0.35,
            labelspacing=0.22,
            handlelength=1.35,
            handletextpad=0.4,
            columnspacing=1.15,
            ncol=2,
        )
        if legend_bbox is not None:
            leg_kw["bbox_to_anchor"] = legend_bbox
            leg_kw["bbox_transform"] = ax_nav.transAxes
        leg = ax_nav.legend(**leg_kw)
        for t in leg.get_texts():
            if t.get_text() in ("LLMs", "Baselines"):
                t.set_fontweight("bold")
                t.set_fontsize(_FS)

        if has_ceps and ax_ceps is not None:
            all_dates = sorted(
                set(d for s in ceps_data.values() if s is not None for d in s.index)
            )
            n_llm_ceps = len([n for n, _ in llm_items if n in ceps_data])
            if all_dates and n_llm_ceps:
                span_days = max(1, (all_dates[-1] - all_dates[0]).days)
                bar_w_days = span_days / max(1, len(all_dates)) * 0.7 / n_llm_ceps
                for i, (name, _) in enumerate(llm_items):
                    if name not in ceps_data or ceps_data[name] is None:
                        continue
                    color = llm_colors[name]
                    offset = (i - (n_llm_ceps - 1) / 2) * bar_w_days
                    for date, val in ceps_data[name].items():
                        ax_ceps.bar(
                            date + pd.Timedelta(days=offset), val,
                            width=pd.Timedelta(days=bar_w_days),
                            color=color, alpha=0.80, edgecolor="none",
                        )
            y_max = max(
                (
                    max(ceps_data[n].dropna().values)
                    for n, _ in llm_items
                    if n in ceps_data and ceps_data[n] is not None
                    and len(ceps_data[n].dropna()) > 0
                ),
                default=0.5,
            )
            ax_ceps.set_ylim(0, y_max * 1.15 if y_max > 0 else 1.0)
            ax_ceps.set_ylabel("CEPS", fontsize=_FS)
            ax_ceps.set_xlabel("Date", fontsize=_FS)
            ax_ceps.tick_params(axis="both", labelsize=_FS)
            for spine in ("top", "right"):
                ax_ceps.spines[spine].set_visible(False)
        else:
            ax_nav.set_xlabel("Date", fontsize=_FS, color="black")

        if show_title and title:
            ax_nav.set_title(title, fontsize=_FS, fontweight="bold", color="black", pad=8)

        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout()
        return fig


def plot_sandbox_metrics(
    metrics_data: dict[str, dict],
    metric_keys: list[str] = None,
    title: str = "Sandbox Performance Metrics",
    figsize: tuple = (9, 4),
) -> Figure:
    """
    Grouped bar chart comparing key backtest metrics across models/profiles.

    Args:
        metrics_data: {label: {metric_key: float, ...}}
        metric_keys:  Metrics to plot (defaults to four core metrics).
        title:        Figure title.
        figsize:      Figure dimensions.
    """
    apply_paper_style()

    if metric_keys is None:
        metric_keys = ["total_return", "sharpe_ratio", "max_drawdown", "volatility"]

    metric_labels = {
        "total_return": "Total Return",
        "cagr": "CAGR",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "max_drawdown": "Max Drawdown",
        "calmar_ratio": "Calmar Ratio",
        "volatility": "Annualized Vol",
        "mean_ceps": "Mean CEPS",
        "mean_profile_score": "Profile Score",
    }

    models = list(metrics_data.keys())
    sorted_models = sorted(models)
    _metrics_color_map = {
        m: _MODEL_COLOURS_SOFT[i % len(_MODEL_COLOURS_SOFT)]
        for i, m in enumerate(sorted_models)
    }
    n_models = len(models)
    n_metrics = len(metric_keys)

    # Wider figure + bar spacing for readability
    figsize = (max(figsize[0], n_models * 1.1), figsize[1])
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    bar_width = 0.65

    for ax, key in zip(axes, metric_keys):
        values = [metrics_data[m].get(key, 0.0) for m in models]
        colors = [
            (
                PAPER_COLORS["failed"]
                if (key == "max_drawdown" and v < -0.20)
                else _metrics_color_map.get(
                    models[i], _MODEL_COLOURS_SOFT[i % len(_MODEL_COLOURS_SOFT)]
                )
            )
            for i, v in enumerate(values)
        ]
        x_positions = [i * 1.15 for i in range(n_models)]
        ax.bar(
            x_positions,
            values,
            width=bar_width,
            color=colors,
            alpha=0.85,
            edgecolor="white",
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [abbrev_model_name(m) for m in models],
            rotation=30,
            ha="right",
            fontsize=6.5,
        )
        ax.set_title(metric_labels.get(key, key), fontsize=9)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    return fig


def plot_ceps_vs_pnl(
    model_data: list[dict],
    title: str = "CEPS Score vs. Realized Return",
    figsize: tuple = (6, 5),
) -> Figure:
    """
    Scatter plot validating CEPS scores against realized total returns.

    Args:
        model_data: list of dicts with keys:
            model_name, mean_ceps, total_return, stress_gate_passed (optional)
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, entry in enumerate(model_data):
        ceps = entry.get("mean_ceps", 0.0)
        ret = entry.get("total_return", 0.0)
        name = entry.get("model_name", f"model_{i}")
        passed = entry.get("stress_gate_passed", entry.get("risk_gate_passed", True))

        color = (
            MODEL_PALETTE[i % len(MODEL_PALETTE)] if passed else PAPER_COLORS["neutral"]
        )
        marker = "o" if passed else "x"
        ax.scatter(ceps, ret, color=color, marker=marker, s=80, zorder=3)
        ax.annotate(
            abbrev_model_name(name),
            (ceps, ret),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color=color,
        )

    if len(model_data) >= 3:
        xs = np.array([e.get("mean_ceps", 0.0) for e in model_data])
        ys = np.array([e.get("total_return", 0.0) for e in model_data])
        if xs.std() > 0:
            m, b = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(
                x_line,
                m * x_line + b,
                "--",
                color="gray",
                linewidth=1.0,
                alpha=0.6,
                label=f"trend (slope={m:.2f})",
            )
            ax.legend(fontsize=8)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Mean CEPS Score")
    ax.set_ylabel("Total Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))

    legend_handles = [
        mpatches.Patch(color=MODEL_PALETTE[0], label="Stress gate passed"),
        mpatches.Patch(color=PAPER_COLORS["neutral"], label="Stress gate failed"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    return fig


def plot_stress_drawdown(
    stress_data: dict[str, dict[str, dict]],
    title: str = "Stress Test Max Drawdown by Profile",
    figsize: tuple = (8, 4),
) -> Figure:
    """
    Heatmap: rows = scenarios, columns = profiles, cells = max_drawdown.
    Tolerance thresholds shown as overlaid text.

    Args:
        stress_data: {model_name: {scenario_name: {max_drawdown, tolerance, stress_passed}}}
                     Use load_sandbox_results_full() to build this.
        title:       Figure title.
        figsize:     Figure dimensions.
    """
    from matplotlib.colors import LinearSegmentedColormap

    apply_paper_style()

    # Expect a single model (call once per model, or pass one model's data)
    # stress_data: {profile: {scenario: {max_drawdown, tolerance, stress_passed}}}
    profiles = list(stress_data.keys())
    if not profiles:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    scenarios = list(next(iter(stress_data.values())).keys())
    n_profiles = len(profiles)
    n_scenarios = len(scenarios)

    matrix = np.zeros((n_scenarios, n_profiles))
    passed_matrix = np.zeros((n_scenarios, n_profiles), dtype=bool)
    tolerance_matrix = np.zeros((n_scenarios, n_profiles))

    for j, profile in enumerate(profiles):
        for i, scenario in enumerate(scenarios):
            entry = stress_data[profile].get(scenario, {})
            dd = abs(entry.get("max_drawdown", 0.0))
            matrix[i, j] = dd
            passed_matrix[i, j] = entry.get("stress_passed", False)
            tolerance_matrix[i, j] = entry.get("tolerance", 0.2)

    cmap = LinearSegmentedColormap.from_list("dd", ["#d4e4f7", "#c0392b"])
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=0.5, aspect="auto")

    ax.set_xticks(range(n_profiles))
    ax.set_xticklabels([p.capitalize() for p in profiles], fontsize=10)
    ax.set_yticks(range(n_scenarios))
    scenario_labels = [
        s.replace("_", " ")
        .replace("2015 china shock", "2015 China Shock")
        .replace("2020 covid flash crash", "2020 COVID Crash")
        .replace("2022 crypto collapse", "2022 Crypto Collapse")
        for s in scenarios
    ]
    ax.set_yticklabels(scenario_labels, fontsize=9)

    for i in range(n_scenarios):
        for j in range(n_profiles):
            dd_val = matrix[i, j]
            tol = tolerance_matrix[i, j]
            status = "✓" if passed_matrix[i, j] else "✗"
            cell_text = f"{dd_val:.1%}\n(tol {tol:.0%}) {status}"
            color = "white" if dd_val > 0.25 else "black"
            ax.text(j, i, cell_text, ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Max Drawdown", fraction=0.03, pad=0.04)
    fig.tight_layout()
    return fig


def plot_profile_nav(
    profile_nav: dict[str, pd.Series],
    model_name: str = "",
    title: str = None,
    figsize: tuple = (8, 4),
) -> Figure:
    """
    NAV curves for the three investor profiles of a single model.
    Shows how the same model performs differently under different constraints.

    Args:
        profile_nav: {"conservative": pd.Series, "balanced": ..., "aggressive": ...}
        model_name:  Model name for subtitle.
        title:       Figure title (auto-generated if None).
        figsize:     Figure dimensions.
    """
    apply_paper_style()

    profile_colors = {
        "conservative": PAPER_COLORS.get("passed", MODEL_PALETTE[0]),
        "balanced": MODEL_PALETTE[1] if len(MODEL_PALETTE) > 1 else "#e67e22",
        "aggressive": PAPER_COLORS.get("failed", "#c0392b"),
    }

    profile_styles = {
        "conservative": ("-", "o"),
        "balanced": ("--", "s"),
        "aggressive": ("-.", "^"),
    }

    fig, ax = plt.subplots(figsize=figsize)

    for profile, nav in profile_nav.items():
        if nav is None or len(nav) == 0:
            continue
        color = profile_colors.get(profile, MODEL_PALETTE[0])
        ls, mk = profile_styles.get(profile, ("-", "o"))
        nav_norm = nav / nav.iloc[0] * 100
        every = max(1, len(nav_norm) // 8)
        ax.plot(
            nav_norm.index,
            nav_norm.values,
            label=profile.capitalize(),
            color=color,
            linewidth=1.8,
            linestyle=ls,
            marker=mk,
            markevery=every,
            markersize=5,
        )

    ax.axhline(100, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized NAV (base=100)")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Data loaders (new structure: {model}/{ts}/{profile}/[stress_*|normal]/)
# ---------------------------------------------------------------------------


def load_sandbox_results(sandbox_dir: str) -> dict[str, dict]:
    """
    Load normal-market backtest results from sandbox output directory.

    Supports the new three-level structure:
        {model}/{timestamp}/{profile}/normal/backtest_result.json
    Falls back to legacy two-level:
        {model}/{timestamp}/backtest_result.json
    And legacy single-level:
        {run_id}/backtest_result.json

    Returns: {"{model}/{profile}": backtest_result_dict} for new structure,
             or {model_name: backtest_result_dict} for legacy.
    """
    import json
    from pathlib import Path

    results = {}
    base = Path(sandbox_dir)
    if not base.exists():
        return results

    first_level = [d for d in sorted(base.iterdir()) if d.is_dir()]

    for model_dir in first_level:
        # Check if model_dir contains timestamp subdirs
        ts_dirs = [d for d in sorted(model_dir.iterdir()) if d.is_dir()]
        if not ts_dirs:
            # Legacy single-level: model_dir is run_dir
            result_file = model_dir / "backtest_result.json"
            nav_file = model_dir / "nav_curve.csv"
            if result_file.exists():
                data = json.loads(result_file.read_text())
                if nav_file.exists():
                    nav_df = pd.read_csv(
                        nav_file, parse_dates=["date"], index_col="date"
                    )
                    data["_nav_series"] = nav_df["nav"]
                results[model_dir.name] = data
            continue

        # Pick latest timestamp
        latest_ts = max(ts_dirs, key=lambda d: d.name)

        # Detect new 3-level: {profile}/normal/backtest_result.json
        profile_dirs = [
            d
            for d in sorted(latest_ts.iterdir())
            if d.is_dir() and d.name in ("conservative", "balanced", "aggressive")
        ]

        if profile_dirs:
            # New structure
            for profile_dir in profile_dirs:
                normal_dir = profile_dir / "normal"
                result_file = normal_dir / "backtest_result.json"
                nav_file = normal_dir / "nav_curve.csv"
                if not result_file.exists():
                    continue
                data = json.loads(result_file.read_text())
                if nav_file.exists():
                    nav_df = pd.read_csv(
                        nav_file, parse_dates=["date"], index_col="date"
                    )
                    data["_nav_series"] = nav_df["nav"]
                key = f"{model_dir.name}/{profile_dir.name}"
                results[key] = data
        else:
            # Legacy two-level: {model}/{timestamp}/backtest_result.json
            result_file = latest_ts / "backtest_result.json"
            nav_file = latest_ts / "nav_curve.csv"
            if result_file.exists():
                data = json.loads(result_file.read_text())
                if nav_file.exists():
                    nav_df = pd.read_csv(
                        nav_file, parse_dates=["date"], index_col="date"
                    )
                    data["_nav_series"] = nav_df["nav"]
                results[model_dir.name] = data

    return results


def load_sandbox_results_full(sandbox_dir: str) -> dict[str, dict]:
    """
    Load ALL sandbox outputs (stress + normal) for building stress heatmaps.

    Returns:
        {model_name: {
            "profile_comparison": {...},
            "profiles": {
                profile_name: {
                    "normal": BacktestResult dict or None,
                    "stress": {scenario_name: BacktestResult dict}
                }
            }
        }}
    """
    import json
    from pathlib import Path

    out = {}
    base = Path(sandbox_dir)
    if not base.exists():
        return out

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        ts_dirs = [d for d in sorted(model_dir.iterdir()) if d.is_dir()]
        if not ts_dirs:
            continue
        latest_ts = max(ts_dirs, key=lambda d: d.name)

        model_entry: dict = {"profiles": {}}
        comp_file = latest_ts / "profile_comparison.json"
        if comp_file.exists():
            model_entry["profile_comparison"] = json.loads(comp_file.read_text())

        for profile_dir in sorted(latest_ts.iterdir()):
            if not profile_dir.is_dir():
                continue
            pname = profile_dir.name
            if pname not in ("conservative", "balanced", "aggressive"):
                continue

            profile_entry: dict = {"normal": None, "stress": {}}

            # Normal
            normal_file = profile_dir / "normal" / "backtest_result.json"
            if normal_file.exists():
                normal_data: dict = json.loads(normal_file.read_text())
                nav_file = profile_dir / "normal" / "nav_curve.csv"
                if nav_file.exists():
                    nav_df = pd.read_csv(
                        nav_file, parse_dates=["date"], index_col="date"
                    )
                    normal_data["_nav_series"] = nav_df["nav"]
                profile_entry["normal"] = normal_data

            # Stress
            for child in sorted(profile_dir.iterdir()):
                if child.is_dir() and child.name.startswith("stress_"):
                    scenario = child.name[len("stress_") :]
                    res_file = child / "backtest_result.json"
                    if res_file.exists():
                        profile_entry["stress"][scenario] = json.loads(
                            res_file.read_text()
                        )

            model_entry["profiles"][pname] = profile_entry

        if model_entry["profiles"]:
            out[model_dir.name] = model_entry

    return out
