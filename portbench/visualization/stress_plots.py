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
    _HATCHES = ["", "//", "\\\\", "xx", "..", "++", "--", "||", "//", "oo"]

    bar_width = 0.7 / n_models
    scenario_positions = np.arange(n_scenarios)

    for i, model in enumerate(model_names):
        color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
        hatch_base = _HATCHES[i % len(_HATCHES)] if i >= len(MODEL_PALETTE) else ""
        offsets = scenario_positions + (i - n_models / 2 + 0.5) * bar_width
        for j in range(n_scenarios):
            score = ceps_matrix[i, j]
            passed = pass_matrix[i, j]
            ax.bar(
                offsets[j],
                score,
                width=bar_width * 0.9,
                color=color,
                hatch=hatch_base,
                alpha=0.85,
                edgecolor="gray",
            )
            if not passed:
                ax.text(
                    offsets[j],
                    score,
                    "✗",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="red",
                    fontweight="bold",
                    zorder=6,
                )

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
            edgecolor="gray",
            label=m,
        )
        for i, m in enumerate(model_names)
    ]
    handles.append(
        plt.Line2D(
            [0], [0], color="red", linestyle="--", label=f"Tolerance ({threshold:.0%})"
        )
    )
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="$✗$",
            color="red",
            linestyle="None",
            markersize=9,
            label="Failed",
        )
    )
    ax.legend(
        handles=handles, fontsize=8, loc="upper left", ncols=max(1, len(handles) // 4)
    )

    fig.tight_layout()
    return fig


def plot_stress_threshold_chart(
    data: dict[str, dict[str, dict[str, dict]]],
    title: str = "Stress Test Drawdown Score",
    thresholds: list = None,
    tier_labels: list = None,
    figsize: tuple = (18, 6),
) -> Figure:
    """
    Dot chart showing each model's continuous drawdown score against threshold levels.

    One subplot per stress scenario. Within each subplot:
      - Colored background bands per tier zone (D=red, C=orange, B=yellow, A=green)
      - X-axis: models sorted by worst-case dd_score (best left, worst right)
      - Y-axis: dd_score ∈ [0, 1]
      - Filled dot = passed stress gate; hollow ring = failed
      - Lollipop stems from y=0 to the dot
      - Worst-profile dot annotated with actual max_drawdown %

    Args:
        data: {model_key: {profile: {scenario: {dd_score, ceps_tier, passed, max_drawdown}}}}
        title:       Figure title.
        thresholds:  Y positions for tier boundaries (ascending).
        tier_labels: Labels for each tier band (same length as thresholds).
        figsize:     Figure size.

    Returns:
        matplotlib Figure.
    """
    if thresholds is None:
        thresholds = [0.1, 0.4, 0.6, 0.8]
    if tier_labels is None:
        tier_labels = ["D", "C", "B", "A"]

    apply_paper_style()

    # Tier band colors: D(red) C(orange) B(yellow) A(green)
    _TIER_BAND_COLORS = ["#fde8e8", "#fde8c8", "#fdfbe8", "#e8fde8"]

    # Collect scenario and model keys
    model_keys = list(data.keys())
    scenario_set: list[str] = []
    for profile_data in data.values():
        for sc_data in profile_data.values():
            for sc in sc_data:
                if sc not in scenario_set:
                    scenario_set.append(sc)
    scenario_set.sort()

    profile_order = ["conservative", "balanced", "aggressive"]
    # Frost-themed profile colors: dark navy / steel blue / gray
    profile_colors = {
        "conservative": "#1e3d6e",  # dark navy (AF darkened)
        "balanced":     "#4a6fa5",  # steel blue (AF1)
        "aggressive":   "#8a8a8a",  # dark silver-gray
    }
    profile_markers = {"conservative": "o", "balanced": "s", "aggressive": "^"}

    # Separate LLM models and baselines
    llm_keys = [mk for mk in model_keys if not mk.startswith("baseline/")]
    baseline_keys = [mk for mk in model_keys if mk.startswith("baseline/")]

    # Sort each group: primary=worst dd_score desc, secondary=mean dd_score desc
    def _sort_key(mk: str) -> tuple:
        scores = []
        for profile in profile_order:
            for sc in scenario_set:
                entry = data.get(mk, {}).get(profile, {}).get(sc, {})
                if entry:
                    scores.append(entry.get("dd_score", 0.0))
        if not scores:
            return (0.0, 0.0)
        return (min(scores), sum(scores) / len(scores))

    llm_keys = sorted(llm_keys, key=_sort_key, reverse=True)
    baseline_keys = sorted(baseline_keys, key=_sort_key, reverse=True)
    ordered_keys = llm_keys + baseline_keys

    n_scenarios = len(scenario_set)
    if n_scenarios == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No stress data", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, axes = plt.subplots(1, n_scenarios, figsize=figsize, sharey=True)
    if n_scenarios == 1:
        axes = [axes]

    x_pos = np.arange(len(ordered_keys))
    x_labels = [abbrev_model_name(mk) for mk in ordered_keys]
    n_profiles = len(profile_order)
    offsets = np.linspace(-0.18, 0.18, n_profiles)

    for ax_idx, sc_name in enumerate(scenario_set):
        ax = axes[ax_idx]

        # ── Tier background bands ──────────────────────────────────────────────
        band_bounds = [0.0] + thresholds + [1.05]
        for bi in range(len(band_bounds) - 1):
            color_idx = min(bi, len(_TIER_BAND_COLORS) - 1)
            ax.axhspan(
                band_bounds[bi],
                band_bounds[bi + 1],
                color=_TIER_BAND_COLORS[color_idx],
                alpha=0.18,
                zorder=0,
            )

        # Tier boundary lines + labels on the right spine
        for thresh, label in zip(thresholds, tier_labels):
            ax.axhline(thresh, color="#aab0b8", linestyle="--", linewidth=0.7, zorder=1)
            ax.text(
                len(ordered_keys) - 0.1,
                thresh + 0.015,
                f"Tier {label}",
                fontsize=6,
                color="#555",
                va="bottom",
                ha="right",
                zorder=5,
            )

        # ── Compute worst-profile dd_score per model (for annotation) ──────────
        worst_score_per_model: dict[str, tuple[float, float, int]] = {}
        for xi, mk in enumerate(ordered_keys):
            worst_dd_score = 1.1
            worst_dd_raw = 0.0
            worst_pidx = 0
            for pidx, profile in enumerate(profile_order):
                entry = data.get(mk, {}).get(profile, {}).get(sc_name, {})
                if not entry:
                    continue
                s = entry.get("dd_score", 0.0)
                if s < worst_dd_score:
                    worst_dd_score = s
                    worst_dd_raw = entry.get("max_drawdown", 0.0)
                    worst_pidx = pidx
            if worst_dd_score < 1.1:
                worst_score_per_model[mk] = (worst_dd_score, worst_dd_raw, worst_pidx)

        # ── Plot lollipops: color by profile ───────────────────────────────────
        for xi, mk in enumerate(ordered_keys):
            for pidx, profile in enumerate(profile_order):
                entry = data.get(mk, {}).get(profile, {}).get(sc_name, {})
                if not entry:
                    continue
                score = entry.get("dd_score", 0.0)
                passed = entry.get("passed", True)
                color = profile_colors[profile]
                marker = profile_markers[profile]
                xpos = x_pos[xi] + offsets[pidx]

                # Lollipop stem
                ax.plot([xpos, xpos], [0, score], color=color, linewidth=0.8, alpha=0.45, zorder=2)

                if passed:
                    ax.scatter(
                        [xpos], [score],
                        color=color, marker=marker, s=50,
                        zorder=4,
                        label="_nolegend_",
                        edgecolors="white", linewidths=0.5,
                    )
                else:
                    ax.scatter(
                        [xpos], [score],
                        facecolors="none", edgecolors=color, marker=marker, s=55,
                        linewidths=1.4, zorder=4,
                        label="_nolegend_",
                    )
                    ax.scatter(
                        [xpos], [score],
                        color=color, marker="x", s=25, linewidths=1.0, zorder=5,
                        label="_nolegend_",
                    )

        # ── Annotate only failed-gate points with actual drawdown % ──────────
        for xi, mk in enumerate(ordered_keys):
            for pidx, profile in enumerate(profile_order):
                entry = data.get(mk, {}).get(profile, {}).get(sc_name, {})
                if not entry or entry.get("passed", True):
                    continue
                score = entry.get("dd_score", 0.0)
                wd = entry.get("max_drawdown", 0.0)
                xpos = x_pos[xi] + offsets[pidx]
                dd_pct = f"{wd * 100:+.1f}%"
                ax.text(
                    xpos, score + 0.04, dd_pct,
                    ha="center", va="bottom", fontsize=6,
                    color="#c0392b", rotation=0, zorder=6,
                )

        # Scenario title — full descriptive name
        _SC_NAMES = {
            "2015_china_shock": "2015 China Stock Market Crisis",
            "2020_covid_flash_crash": "2020 COVID-19 Flash Crash",
            "2022_crypto_collapse": "2022 Crypto Market Collapse",
        }
        sc_key = sc_name.removeprefix("stress_")
        sc_label = _SC_NAMES.get(sc_key, sc_key.replace("_", " ").title())
        ax.set_title(sc_label, fontsize=10, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha="right")
        ax.set_ylim(-0.02, 1.12)
        ax.set_xlim(-0.6, len(ordered_keys) - 0.4)

        if ax_idx == 0:
            ax.set_ylabel("Drawdown Score  (1 = no loss → 0 = full loss)", fontsize=9)

        # Vertical separator between LLM models and baselines
        if llm_keys and baseline_keys:
            sep = len(llm_keys) - 0.5
            ax.axvline(sep, color="#bdc3c7", linestyle=":", linewidth=1.0)

    # ── Legend: profile colors + failed indicator ──────────────────────────────
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    legend_handles = []
    for profile in profile_order:
        c = profile_colors[profile]
        m = profile_markers[profile]
        legend_handles.append(
            mlines.Line2D([], [], color=c, marker=m, linestyle="None",
                          markersize=7, markerfacecolor=c, label=profile.capitalize())
        )
    legend_handles.append(
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                      markersize=7, markerfacecolor="none", markeredgewidth=1.5,
                      label="Failed gate")
    )
    axes[0].legend(handles=legend_handles, title="Profile", fontsize=8, title_fontsize=8,
                   loc="upper left", framealpha=0.9)

    fig.tight_layout()
    return fig


def plot_stress_continuous_heatmap(
    data: dict[str, dict[str, dict[str, dict]]],
    title: str = "Stress Test Continuous Score",
    figsize: tuple = (7.2, 5.2),
) -> Figure:
    """
    Heatmap: rows = models, columns = scenarios.

    Color encodes normalized drawdown score (darker = milder). Cell text shows
    raw max-drawdown % only; failed cells append × (no triple score/DD/X encoding).
    Worst case is taken as the minimum score across investor profiles.
    """
    apply_paper_style()

    _SC_NAMES = {
        "2015_china_shock":       "2015 China\nStock Crisis",
        "2020_covid_flash_crash": "2020 COVID\nFlash Crash",
        "2022_crypto_collapse":   "2022 Crypto\nCollapse",
    }

    # ── Collect ordered scenario and model keys ───────────────────────────────
    scenario_set: list[str] = []
    for pd_ in data.values():
        for sc_map in pd_.values():
            for sc in sc_map:
                if sc not in scenario_set:
                    scenario_set.append(sc)
    scenario_set.sort()
    n_sc = len(scenario_set)

    profile_order = ["conservative", "balanced", "aggressive"]
    model_keys = list(data.keys())

    def _mean_worst(mk):
        scores = [min((data[mk].get(p, {}).get(sc, {}).get("dd_score", 0.0)
                       for p in profile_order), default=0.0)
                  for sc in scenario_set]
        return -np.mean(scores)

    llm_keys      = sorted([m for m in model_keys if not m.startswith("baseline/")], key=_mean_worst)
    baseline_keys = sorted([m for m in model_keys if     m.startswith("baseline/")], key=_mean_worst)
    ordered_keys  = llm_keys + baseline_keys
    n_models      = len(ordered_keys)

    # ── Build matrices ────────────────────────────────────────────────────────
    score_mat  = np.zeros((n_models, n_sc))
    dd_mat     = np.zeros((n_models, n_sc))
    passed_mat = np.ones((n_models, n_sc), dtype=bool)

    for ri, mk in enumerate(ordered_keys):
        for ci, sc in enumerate(scenario_set):
            worst_score, worst_dd, any_fail = 1.1, 0.0, False
            for prof in profile_order:
                entry = data.get(mk, {}).get(prof, {}).get(sc, {})
                if not entry:
                    continue
                s = entry.get("dd_score", 0.0)
                if s < worst_score:
                    worst_score = s
                    worst_dd    = entry.get("max_drawdown", 0.0)
                if not entry.get("passed", True):
                    any_fail = True
            score_mat[ri, ci]  = worst_score if worst_score < 1.1 else 0.0
            dd_mat[ri, ci]     = worst_dd
            passed_mat[ri, ci] = not any_fail

    # ── Colormap: Blues (darker = higher / milder drawdown) ───────────────────
    cmap = plt.get_cmap("Blues").copy()

    w, h = figsize
    fig, ax = plt.subplots(figsize=(min(w, 7.5), h))
    im = ax.imshow(score_mat, cmap=cmap, vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")

    # ── Cell annotations + fail border/× ──────────────────────────────────────
    import matplotlib.patches as mpatches

    for ri in range(n_models):
        for ci in range(n_sc):
            sc_val = score_mat[ri, ci]
            dd_val = dd_mat[ri, ci]
            passed = passed_mat[ri, ci]
            txt_color = "#111111" if sc_val < 0.55 else "#ffffff"
            if passed:
                ax.text(
                    ci, ri, f"{dd_val * 100:.1f}%",
                    ha="center", va="center",
                    fontsize=8.5, color=txt_color, fontweight="medium",
                )
            else:
                # Red cell border so failures read at a glance
                ax.add_patch(
                    mpatches.Rectangle(
                        (ci - 0.48, ri - 0.48), 0.96, 0.96,
                        fill=False, edgecolor="#c0392b", linewidth=2.0, zorder=4,
                    )
                )
                ax.text(
                    ci, ri, f"{dd_val * 100:.1f}%  ×",
                    ha="center", va="center",
                    fontsize=8.5, color="#c0392b", fontweight="bold", zorder=5,
                )

    # ── Axes tick labels ──────────────────────────────────────────────────────
    sc_col_labels = [_SC_NAMES.get(sc.removeprefix("stress_"), sc.replace("_", " "))
                     for sc in scenario_set]
    ax.set_xticks(range(n_sc))
    ax.set_xticklabels(sc_col_labels, fontsize=9, fontweight="bold")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([abbrev_model_name(m) for m in ordered_keys], fontsize=8.5)
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, n_sc - 0.5)
    ax.set_ylim(n_models - 0.5, -0.5)

    # ── LLMs / Baselines separator ────────────────────────────────────────────
    if llm_keys and baseline_keys:
        sep = len(llm_keys) - 0.5
        ax.axhline(sep, color="#555555", linestyle="-", linewidth=1.1, alpha=0.7)
        ax.text(
            n_sc - 0.45, sep - 0.12, "LLMs",
            fontsize=7.5, color="#333333", ha="right", va="bottom", style="italic",
        )
        ax.text(
            n_sc - 0.45, sep + 0.12, "Baselines",
            fontsize=7.5, color="#333333", ha="right", va="top", style="italic",
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.set_title("DD score\n(milder↑)", fontsize=7, pad=3)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.tight_layout()
    return fig


def plot_stress_drawdown_bars(
    data: dict[str, dict[str, dict[str, dict]]],
    title: str = "",
    figsize: tuple = (7.2, 3.6),
    gate_tol: float = 0.10,
) -> Figure:
    """
    1×3 horizontal bar panels: MaxDD% per model (worst across profiles) for each
    stress regime. Vertical dashed line marks the conservative gate (−gate_tol).
    Failed bars (any profile exceeds tolerance) are red; LLM vs baseline separated.
    """
    apply_paper_style()

    _SC_NAMES = {
        "2015_china_shock": "2015 China Stock Crisis",
        "2020_covid_flash_crash": "2020 COVID Flash Crash",
        "2022_crypto_collapse": "2022 Crypto Collapse",
    }
    _SC_SHORT = {
        "2015_china_shock": "(a) 2015 China",
        "2020_covid_flash_crash": "(b) 2020 COVID",
        "2022_crypto_collapse": "(c) 2022 Crypto",
    }

    scenario_set: list[str] = []
    for pd_ in data.values():
        for sc_map in pd_.values():
            for sc in sc_map:
                if sc not in scenario_set:
                    scenario_set.append(sc)
    scenario_set.sort()

    profile_order = ["conservative", "balanced", "aggressive"]
    model_keys = list(data.keys())

    def _worst_entry(mk: str, sc: str) -> tuple[float, bool]:
        """Return (worst MaxDD across profiles, fail under conservative gate)."""
        worst_score, worst_dd = 1.1, 0.0
        for prof in profile_order:
            entry = data.get(mk, {}).get(prof, {}).get(sc, {})
            if not entry:
                continue
            s = float(entry.get("dd_score", 0.0))
            if s < worst_score:
                worst_score = s
                worst_dd = float(entry.get("max_drawdown", 0.0))
        # Gate coloring follows the conservative profile (matches −10% line)
        cons = data.get(mk, {}).get("conservative", {}).get(sc, {})
        if cons:
            failed = not bool(cons.get("passed", True))
            if "passed" not in cons:
                failed = abs(float(cons.get("max_drawdown", 0.0))) > gate_tol + 1e-9
        else:
            failed = abs(worst_dd) > gate_tol + 1e-9
        return (worst_dd if worst_score < 1.1 else 0.0, failed)

    def _mean_dd(mk: str) -> float:
        dds = [_worst_entry(mk, sc)[0] for sc in scenario_set]
        return float(np.mean(dds)) if dds else 0.0

    # Sort: more negative (worse) mean DD first within each group
    llm_keys = sorted(
        [m for m in model_keys if not m.startswith("baseline/")],
        key=_mean_dd,
    )
    baseline_keys = sorted(
        [m for m in model_keys if m.startswith("baseline/")],
        key=_mean_dd,
    )
    ordered = llm_keys + baseline_keys
    n_models = len(ordered)
    y = np.arange(n_models)
    labels = [abbrev_model_name(m) for m in ordered]

    llm_color = "#4a6fa5"
    base_color = "#7a8a99"
    fail_color = "#c0392b"
    gate_x = -100.0 * gate_tol

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for ax, sc in zip(axes, scenario_set):
        dds = []
        fails = []
        for mk in ordered:
            dd, failed = _worst_entry(mk, sc)
            dds.append(100.0 * dd)  # to percent
            fails.append(failed)

        colors = [
            fail_color if f else (llm_color if not mk.startswith("baseline/") else base_color)
            for mk, f in zip(ordered, fails)
        ]
        ax.barh(y, dds, height=0.72, color=colors, edgecolor="white", linewidth=0.3, zorder=3)
        ax.axvline(gate_x, color=fail_color, linestyle="--", linewidth=1.1, zorder=2, alpha=0.85)
        ax.axvline(0.0, color="#bbbbbb", linewidth=0.7, zorder=1)
        ax.set_xlabel("MaxDD (%)", fontsize=8)
        ax.tick_params(axis="x", labelsize=7.5)
        short = _SC_SHORT.get(sc.removeprefix("stress_"), _SC_NAMES.get(sc.removeprefix("stress_"), sc))
        # strip stress_ prefix variants
        key = sc.removeprefix("stress_")
        short = _SC_SHORT.get(key, short)
        ax.text(
            0.5, 1.02, short,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, fontweight="regular", color="#333333", clip_on=False,
        )
        ax.set_xlim(min(min(dds) - 1.5, gate_x - 2), 1.0)
        ax.grid(True, axis="x", alpha=0.35, linewidth=0.5)
        ax.set_ylim(n_models - 0.55, -0.55)

        # LLM / baseline separator
        if llm_keys and baseline_keys:
            sep = len(llm_keys) - 0.5
            ax.axhline(sep, color="#666666", linewidth=0.9, linestyle="-", alpha=0.6)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.5)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)

    if llm_keys and baseline_keys:
        sep = len(llm_keys) - 0.5
        axes[-1].text(
            1.02, sep - 0.15, "LLMs", transform=axes[-1].get_yaxis_transform(),
            fontsize=7, color="#333333", ha="left", va="bottom", style="italic",
        )
        axes[-1].text(
            1.02, sep + 0.15, "Baselines", transform=axes[-1].get_yaxis_transform(),
            fontsize=7, color="#333333", ha="left", va="top", style="italic",
        )

    handles = [
        mpatches.Patch(color=llm_color, label="LLM (pass)"),
        mpatches.Patch(color=base_color, label="Baseline (pass)"),
        mpatches.Patch(color=fail_color, label="Fail gate"),
        plt.Line2D([0], [0], color=fail_color, linestyle="--", linewidth=1.1,
                   label=f"Gate (−{gate_tol*100:.0f}%)"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, fontsize=7.5,
        frameon=True, bbox_to_anchor=(0.5, -0.04),
    )
    if title:
        fig.suptitle(title, fontsize=10, fontweight="normal", y=1.05)
    fig.subplots_adjust(left=0.14, right=0.92, top=0.88, bottom=0.22, wspace=0.12)
    return fig
