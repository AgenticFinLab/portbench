"""
Generate Figure for Section 5.4 — Profile Adaptation as LLM Value.

Model-centric grouped bar chart: each model has 3 bars (Conservative, Balanced,
Aggressive), sorted left-to-right by adaptation variance (descending). Models
with visibly uneven bar heights genuinely adapt their portfolios to different
risk profiles; models with near-flat bars apply a uniform policy.

Output: figures/analysis_profile_adaptation.png
"""

import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.visualization.style import apply_paper_style, PAPER_COLORS

# ---------------------------------------------------------------------------
# Model display names (matching paper conventions)
# ---------------------------------------------------------------------------
MODEL_DISPLAY = {
    "deepseek-v4-flash":    "DS-V4-\nFlash",
    "deepseek-v4-pro":      "DS-V4-\nPro",
    "qwen3.7-max":          "Qwen3.7-\nMax",
    "qwen3.6-plus":         "Qwen3.6-\nPlus",
    "qwen3.6-35b-a3b":      "Qwen3.6-\n35B-A3B",
    "glm-5.1":              "GLM-\n5.1",
    "doubao-seed-2-0-lite": "DB-2.0-\nLite",
    "doubao-seed-2-0-pro":  "DB-2.0-\nPro",
    "hy3-preview":          "HY3-\nPreview",
    "kimi-k2.6":            "Kimi-\nK2.6",
}

PROFILE_LABELS = ["Conservative", "Balanced", "Aggressive"]
PROFILE_KEYS = ["conservative", "balanced", "aggressive"]

# Frost palette: dark navy, steel blue, light steel
PROFILE_COLORS = ["#1e3d6e", "#4a6fa5", "#7a9fc5"]


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
    """Generate the profile adaptation bar chart."""
    apply_paper_style()

    # Filter to LLM models only
    llm_models = {
        m: d for m, d in pas_data.items()
        if any(v > 0 for v in d.values()) and len(d) == 3
    }

    # Sort by adaptation std (descending) — higher std = more adaptation
    def adaptation_std(item):
        scores = [item[1].get(k, 0.0) for k in PROFILE_KEYS]
        return np.std(scores)

    sorted_models = sorted(llm_models.items(), key=adaptation_std, reverse=True)
    model_names = [m for m, _ in sorted_models]
    display_names = [MODEL_DISPLAY.get(m, m) for m in model_names]
    n_models = len(model_names)
    n_profiles = len(PROFILE_KEYS)

    # Build score matrix
    score_matrix = np.zeros((n_models, n_profiles))
    for i, (_, scores) in enumerate(sorted_models):
        for j, key in enumerate(PROFILE_KEYS):
            score_matrix[i, j] = scores.get(key, 0.0)

    model_stds = score_matrix.std(axis=1, ddof=0)

    # Plot
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    bar_width = 0.22
    x = np.arange(n_models)

    for j in range(n_profiles):
        offset = (j - n_profiles / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            score_matrix[:, j],
            width=bar_width * 0.92,
            color=PROFILE_COLORS[j],
            alpha=0.90,
            edgecolor="white",
            linewidth=0.3,
            label=PROFILE_LABELS[j],
            zorder=3,
        )

    # Axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=7.5, linespacing=1.2)
    ax.set_ylabel("Profile Alignment Score (PAS)", fontsize=10)
    ax.set_ylim(0, 1.13)
    ax.set_xlim(-0.6, n_models - 0.4)

    # Perfect alignment reference
    ax.axhline(1.0, color=PAPER_COLORS["failed"], linestyle="--",
               linewidth=0.8, alpha=0.6, zorder=1)

    # Adaptation std annotation above each model group
    for i, std_val in enumerate(model_stds):
        ax.text(
            i, 1.06, f"σ={std_val:.3f}", ha="center", va="bottom",
            fontsize=6.5, color="#555555", style="italic",
        )

    # Legend — profiles only, bottom-right to avoid overlapping σ annotations
    ax.legend(
        fontsize=8, loc="lower right", framealpha=0.9,
        edgecolor=PAPER_COLORS["neutral"], ncol=3,
    )

    # Light grid
    ax.yaxis.grid(True, alpha=0.4, linestyle="-", linewidth=0.5,
                  color=PAPER_COLORS["neutral"])
    ax.set_axisbelow(True)

    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"Saved → {out}")
    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", default="EXPERIMENTS/monthly")
    parser.add_argument("--output", default="figures/analysis_profile_adaptation.png")
    args = parser.parse_args()

    pas_data = load_pas_data(args.experiments_dir)
    print(f"Loaded PAS data for {len(pas_data)} models")
    for model, scores in sorted(pas_data.items()):
        vals = [f"{scores.get(k, 0):.4f}" for k in PROFILE_KEYS]
        print(f"  {model:25s}  {vals}")

    make_figure(pas_data, args.output)
