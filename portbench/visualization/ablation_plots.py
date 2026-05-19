"""
σ ablation visualization: how S3 scoring parameter σ affects model rankings.

Figure — plot_sigma_ablation:  3-panel subplot (one per investor profile).
    Each panel: x=σ value, y=mean CEPS, one line per model.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from .style import apply_paper_style, MODEL_PALETTE, abbrev_model_name

_PROFILE_KEYS = ["conservative", "balanced", "aggressive"]
_PROFILE_LABELS = ["Conservative", "Balanced", "Aggressive"]


def plot_sigma_ablation(
    results: dict,
    title: str = "σ Ablation — S3 Scoring Balance",
    figsize: tuple = (13, 4.5),
) -> Figure:
    """
    3-panel line chart: effect of σ on mean CEPS per investor profile.

    Args:
        results: dict with keys:
            - "sigma_values": list[float]
            - "models": {model_key: {profile: {sigma_str: mean_ceps}}}
        title:   Figure title.
        figsize: Figure size.

    Returns:
        matplotlib Figure with 3 subplots (conservative / balanced / aggressive).
        Panels share y-axis range. A rank-shift annotation is added when
        the Kendall τ between σ=0 and σ=1 rankings is < 0.7 for any model.
    """
    apply_paper_style()

    sigma_values = [float(s) for s in results.get("sigma_values", [])]
    models_data = results.get("models", {})
    model_keys = [m for m in models_data if models_data[m]]

    if not sigma_values or not model_keys:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    all_y: list[float] = []

    for ax, profile_key, profile_label in zip(axes, _PROFILE_KEYS, _PROFILE_LABELS):
        rank_at_sigma0: dict[str, float] = {}
        rank_at_sigma1: dict[str, float] = {}

        for i, mk in enumerate(model_keys):
            color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
            profile_data = models_data.get(mk, {}).get(profile_key, {})
            y = [profile_data.get(str(s), float("nan")) for s in sigma_values]
            valid = [v for v in y if not np.isnan(v)]
            all_y.extend(valid)

            label = abbrev_model_name(mk)
            ax.plot(sigma_values, y, marker="o", markersize=4,
                    linewidth=1.8, color=color, label=label)

            # Collect rank endpoints for annotation
            if not np.isnan(y[0]):
                rank_at_sigma0[mk] = y[0]
            if not np.isnan(y[-1]):
                rank_at_sigma1[mk] = y[-1]

        ax.set_title(profile_label, fontsize=10, fontweight="bold")
        ax.set_xlabel("σ", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Mean CEPS", fontsize=9)
        ax.set_xticks(sigma_values)
        ax.set_xticklabels([str(s) for s in sigma_values], fontsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

        # Rank-change annotation: label the model with largest rank flip
        if len(rank_at_sigma0) >= 2 and len(rank_at_sigma1) >= 2:
            ranked0 = sorted(rank_at_sigma0, key=rank_at_sigma0.get, reverse=True)
            ranked1 = sorted(rank_at_sigma1, key=rank_at_sigma1.get, reverse=True)
            max_shift = 0
            max_model = ""
            for mk in model_keys:
                if mk in ranked0 and mk in ranked1:
                    shift = abs(ranked0.index(mk) - ranked1.index(mk))
                    if shift > max_shift:
                        max_shift = shift
                        max_model = mk
            if max_shift > 0:
                ax.annotate(
                    f"Δrank={max_shift}\n{abbrev_model_name(max_model)}",
                    xy=(0.97, 0.03), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=7,
                    color="gray",
                )

    # Shared y range with small margin
    if all_y:
        y_min = max(0.0, min(all_y) - 0.05)
        y_max = min(1.0, max(all_y) + 0.05)
        for ax in axes:
            ax.set_ylim(y_min, y_max)

    # Shared legend below all subplots
    handles = [
        mpatches.Patch(color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
                       label=abbrev_model_name(mk))
        for i, mk in enumerate(model_keys)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncols=min(len(model_keys), 6),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig
