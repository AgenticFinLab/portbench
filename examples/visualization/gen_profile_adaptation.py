"""
Generate Figure for Section 5.4 — Profile Adaptation as LLM Value.

Scatter design (aligned with analysis_normal_vs_stress):
  X = AdaptScore, Y = PAS.
  Each model contributes three independent points (one per investor profile).
  Encoding:
    - profile → colour (solid fill, uniform white edge)
    - model   → marker shape
  All three profile points share the model AdaptScore (no stems, no dodge).

Output: figures/analysis_profile_adaptation.png
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.visualization.risk_return_plots import _MODEL_MARKERS
from portbench.visualization.style import abbrev_model_name, apply_paper_style

PROFILE_LABELS = ["Conservative", "Balanced", "Aggressive"]
PROFILE_KEYS = ["conservative", "balanced", "aggressive"]

# Editorial categorical: cool–mid–warm, muted chroma, print-safe.
# Hue separation ≈ blue–teal–copper (CVD-friendly); matched mid-value
# so marker shape, not brightness, carries identity.
PROFILE_COLORS = {
    "conservative": "#2C4A6E",  # slate navy
    "balanced":     "#4A8B7F",  # muted teal
    "aggressive":   "#B87A45",  # antique copper
}

# Matplotlib marker area is not perceptually equal across shapes.
# Scale `s` so diamonds/stars/pluses read as the same size as circles.
_MARKER_SIZE_SCALE = {
    "o": 1.00,
    "s": 0.85,
    "^": 0.95,
    "D": 0.45,
    "*": 2.05,
    "H": 0.78,
    "v": 0.95,
    "<": 0.95,
    ">": 0.95,
    "P": 0.95,
    "X": 0.95,
    "p": 0.90,
    "h": 0.78,
}
_BASE_SCATTER_S = 90.0
_BASE_LEGEND_MS = 6.5

# User-annotated legend anchors in data coordinates (AdaptScore×100, PAS).
# PROFILE box ≈ (0–4.8, 0.79–0.865); MODEL box ≈ (11.2–17.5, 0.69–0.865).
_PROFILE_ANCHOR = (0.8, 0.870)   # upper-left of left empty band
# Upper-right of Model box: below the GLM-5.1 balanced (pink) star
# (~13.3, 0.88), inset from x_hi=18 so the frame is not clipped.
_MODEL_ANCHOR = (17.95, 0.832)


def _place_data_legend(
    ax: plt.Axes,
    handles: list,
    ncol: int,
    title: str,
    anchor_data: tuple[float, float],
    loc: str,
    legend_kw: dict,
):
    """
    Place an axes legend at a data-coordinate anchor.

    The legend packer is centre-aligned so the short title sits over the
    wider entry block; each entry's own TextArea stays left-aligned.
    Returns the legend artist.
    """
    leg = ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=anchor_data,
        bbox_transform=ax.transData,
        ncol=ncol,
        title=title,
        title_fontproperties={"size": 6.5, "weight": "normal"},
        **legend_kw,
    )
    # Centre title over the entry block (entries remain left-aligned inside).
    leg._legend_box.align = "center"
    leg._legend_box.sep = 1.0
    title_obj = leg.get_title()
    title_obj.set_fontweight("normal")
    title_obj.set_ha("center")
    leg.set_clip_on(False)
    ax.add_artist(leg)
    return leg


def load_pas_data(experiments_dir: str) -> dict[str, dict[str, float]]:
    """Extract mean_profile_score from backtest_result.json files (normal period)."""
    results: dict[str, dict[str, float]] = {}
    base = Path(experiments_dir).resolve()

    for root, dirs, files in os.walk(str(base)):
        for f in files:
            if f == "backtest_result.json" and "normal" in root:
                rel = Path(root).relative_to(base)
                parts = rel.parts
                if len(parts) < 4:
                    continue
                model_dir = parts[1]
                profile = parts[3]
                model_name = re.sub(r"-\d{6,8}$", "", model_dir)
                model_name = re.sub(r"-260215$", "", model_name)

                with open(os.path.join(root, f)) as fh:
                    data = json.load(fh)

                pas = data.get("mean_profile_score")
                if model_name not in results:
                    results[model_name] = {}
                if pas is not None:
                    results[model_name][profile] = pas

    return results


def make_figure(pas_data: dict, output_path: str) -> plt.Figure:
    """Scatter: AdaptScore (x) vs PAS (y), three profiles per model."""
    apply_paper_style()

    llm_models = {
        m: d for m, d in pas_data.items()
        if any(v > 0 for v in d.values()) and len(d) == 3
    }

    model_keys = sorted(llm_models.keys(), key=abbrev_model_name)
    model_meta: dict[str, dict] = {}
    for i, mk in enumerate(model_keys):
        scores = [llm_models[mk][k] for k in PROFILE_KEYS]
        model_meta[mk] = {
            "marker": _MODEL_MARKERS[i % len(_MODEL_MARKERS)],
            "short": abbrev_model_name(mk),
            "adapt": float(np.std(scores, ddof=0)),
            "pas": {k: float(llm_models[mk][k]) for k in PROFILE_KEYS},
        }

    adapts = [model_meta[m]["adapt"] for m in model_keys]
    all_pas = [model_meta[m]["pas"][k] for m in model_keys for k in PROFILE_KEYS]
    # Plot AdaptScore ×100 so the axis is 0/5/10/15 instead of 0.00/0.05/...
    scale = 100.0
    x_pad = 1.2
    y_pad = 0.03
    x_lo = max(0.0, min(adapts) * scale - x_pad)
    # Extra right pad so the Model legend (upper-right anchored) is not clipped
    x_hi = max(max(adapts) * scale + x_pad * 1.5, 18.0)
    y_lo = max(0.55, min(all_pas) - y_pad)
    y_hi = 1.02  # tiny pad so markers at PAS=1.0 are not clipped

    # Separate near-colliding AdaptScores so model groups stay readable
    x_pos: dict[str, float] = {m: model_meta[m]["adapt"] * scale for m in model_keys}
    sorted_by_adapt = sorted(model_keys, key=lambda m: model_meta[m]["adapt"])
    for a, b in zip(sorted_by_adapt, sorted_by_adapt[1:]):
        if abs(x_pos[b] - x_pos[a]) < 0.40:
            x_pos[a] -= 0.25
            x_pos[b] += 0.25

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    ax.set_facecolor("white")
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.35, color="#b0c0d0")

    # Draw aggressive → balanced → conservative so darker points sit on top when close
    draw_order = ["aggressive", "balanced", "conservative"]
    for mk in model_keys:
        meta = model_meta[mk]
        msize = _BASE_SCATTER_S * _MARKER_SIZE_SCALE.get(meta["marker"], 1.0)
        for key in draw_order:
            ax.scatter(
                x_pos[mk],
                meta["pas"][key],
                marker=meta["marker"],
                s=msize,
                c=PROFILE_COLORS[key],
                edgecolors="white",
                linewidths=0.85,
                alpha=0.95,
                zorder=5,
            )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel(r"AdaptScore ($\times 100$)", fontsize=9)
    ax.set_ylabel("Profile Alignment Score (PAS)", fontsize=9)

    # Profile colour legend
    profile_handles = [
        mlines.Line2D(
            [], [],
            color=PROFILE_COLORS[k],
            marker="o",
            linestyle="None",
            markersize=_BASE_LEGEND_MS,
            markerfacecolor=PROFILE_COLORS[k],
            markeredgecolor="white",
            markeredgewidth=0.85,
            label=lab,
        )
        for k, lab in zip(PROFILE_KEYS, PROFILE_LABELS)
    ]
    # Compact legend labels (long Qwen names were overflowing the axes)
    _LEGEND_SHORT = {
        "Qwen3.6-35b": "Q3.6-35b",
        "Qwen3.6-Plus": "Q3.6-Plus",
        "Qwen3.7-Max": "Q3.7-Max",
        "Doubao-Lite": "DB-Lite",
        "Doubao-Pro": "DB-Pro",
    }
    # Model shape legend (neutral fill so colour stays reserved for profile)
    model_handles = [
        mlines.Line2D(
            [], [],
            color="#555555",
            marker=model_meta[mk]["marker"],
            linestyle="None",
            markersize=_BASE_LEGEND_MS * np.sqrt(
                _MARKER_SIZE_SCALE.get(model_meta[mk]["marker"], 1.0)
            ),
            markerfacecolor="#888888",
            markeredgecolor="white",
            markeredgewidth=0.85,
            label=_LEGEND_SHORT.get(model_meta[mk]["short"], model_meta[mk]["short"]),
        )
        for mk in model_keys
    ]

    legend_kw = dict(
        fontsize=5.8,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        framealpha=0.95,
        borderpad=0.30,
        labelspacing=0.22,
        columnspacing=0.55,
        handletextpad=0.28,
    )

    fig.tight_layout()

    # Fixed anchors from user annotation (red boxes)
    leg_left = _place_data_legend(
        ax, profile_handles, ncol=1, title="Profile",
        anchor_data=_PROFILE_ANCHOR, loc="upper left", legend_kw=legend_kw,
    )
    leg_right = _place_data_legend(
        ax, model_handles, ncol=2, title="Model",
        anchor_data=_MODEL_ANCHOR, loc="upper right", legend_kw=legend_kw,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out, dpi=300, bbox_inches="tight",
        bbox_extra_artists=(leg_left, leg_right),
        pad_inches=0.03,
    )
    print(f"Saved → {out}")
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-dir",
        default="EXPERIMENTS_rebuttal_lookback/monthly",
        help="Experiment tree with backtest_result.json files",
    )
    parser.add_argument(
        "--output",
        default="figures/analysis_profile_adaptation.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    pas_data = load_pas_data(args.experiments_dir)
    print(f"Loaded PAS data for {len(pas_data)} models")
    for model, scores in sorted(pas_data.items()):
        vals = [f"{scores.get(k, 0):.4f}" for k in PROFILE_KEYS]
        print(f"  {model:25s}  {vals}")

    make_figure(pas_data, args.output)
