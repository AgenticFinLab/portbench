"""
Generate side-by-side theme preview images for all three PortBench visual themes:
  - paper      : original serif publication style
  - minimalist : Modern Minimalist (clean white, sans-serif, muted palette)
  - galaxy     : Midnight Galaxy (dark background, vivid neon accents)

Usage:
    python examples/visualization/preview_themes.py

Output (written to outputs/theme_preview/):
    theme_minimalist.png
    theme_galaxy.png

Each image shows five panels:
  1. CEPS radar chart
  2. CEPS error propagation heatmap
  3. CEPS violin / distribution
  4. NAV curve backtest
  5. CEPS vs PnL scatter
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.visualization.style import (
    apply_style, STAGE_IDS, STAGE_LABELS,
)

# ---------------------------------------------------------------------------
# Synthetic demo data (same across both themes for fair comparison)
# ---------------------------------------------------------------------------

MODELS = ["GPT-4o", "Claude-3.5", "Qwen-Plus", "Equal-Weight"]

DEMO_STAGE_SCORES = {
    "GPT-4o":       {"S1": 0.82, "S2": 0.78, "S3": 0.71, "S4": 0.95, "S5": 0.90},
    "Claude-3.5":   {"S1": 0.88, "S2": 0.85, "S3": 0.80, "S4": 0.96, "S5": 0.92},
    "Qwen-Plus":    {"S1": 0.74, "S2": 0.70, "S3": 0.55, "S4": 0.95, "S5": 0.88},
    "Equal-Weight": {"S1": 0.50, "S2": 0.50, "S3": 0.65, "S4": 0.97, "S5": 0.91},
}

DEMO_CEPS_TOTALS = {
    "GPT-4o": 0.82, "Claude-3.5": 0.87, "Qwen-Plus": 0.73, "Equal-Weight": 0.66,
}

rng = np.random.default_rng(42)
DEMO_EPISODE_SCORES = {
    m: (np.clip(rng.normal(v, 0.08, 30), 0, 1).tolist())
    for m, v in DEMO_CEPS_TOTALS.items()
}

# NAV curves
dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")
DEMO_NAV = {}
for i, m in enumerate(MODELS):
    drift = [0.0006, 0.0008, 0.0003, 0.0004][i]
    sigma = [0.012, 0.010, 0.015, 0.011][i]
    rets = rng.normal(drift, sigma, len(dates))
    nav = 1_000_000 * np.cumprod(1 + rets)
    DEMO_NAV[m] = pd.Series(nav, index=dates)

DEMO_CEPS_VS_PNL = [
    {"model_name": "GPT-4o",       "mean_ceps": 0.82, "total_return": 0.143, "risk_gate_passed": True},
    {"model_name": "Claude-3.5",   "mean_ceps": 0.87, "total_return": 0.179, "risk_gate_passed": True},
    {"model_name": "Qwen-Plus",    "mean_ceps": 0.73, "total_return": 0.082, "risk_gate_passed": True},
    {"model_name": "Equal-Weight", "mean_ceps": 0.66, "total_return": 0.094, "risk_gate_passed": False},
]


# ---------------------------------------------------------------------------
# Individual panel drawing functions (theme-aware)
# ---------------------------------------------------------------------------

def _draw_radar(ax, colors, stage_colors, regime_colors, model_palette):
    n = len(STAGE_IDS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(STAGE_LABELS, size=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=6)

    for i, (model, scores) in enumerate(DEMO_STAGE_SCORES.items()):
        vals = [scores[s] for s in STAGE_IDS] + [scores[STAGE_IDS[0]]]
        c = model_palette[i % len(model_palette)]
        ax.plot(angles, vals, color=c, linewidth=1.8)
        ax.fill(angles, vals, color=c, alpha=0.12)

    ax.set_title("Per-Stage Capability (Radar)", fontsize=9, fontweight="bold", pad=12)


def _draw_heatmap(ax, colors, stage_colors, regime_colors, model_palette, cmap="RdBu"):
    model_names = list(DEMO_STAGE_SCORES.keys())
    col_ids = STAGE_IDS + ["CEPS"]
    col_labels = ["S1", "S2", "S3", "S4", "S5", "CEPS"]
    data = np.array([
        [DEMO_STAGE_SCORES[m][s] for s in STAGE_IDS] + [DEMO_CEPS_TOTALS[m]]
        for m in model_names
    ])

    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.axvline(x=len(STAGE_IDS) - 0.5, color=colors["neutral"], linewidth=1.5)

    for r in range(len(model_names)):
        for c in range(len(col_ids)):
            v = data[r, c]
            tc = "black" if 0.35 < v < 0.85 else "white"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color=tc, fontweight="bold")

    ax.set_xticks(range(len(col_ids)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=8)
    ax.set_title("Error Propagation Heatmap (CEPS)", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)


def _draw_violin(ax, colors, stage_colors, regime_colors, model_palette, median_color="#36454f"):
    model_names = list(DEMO_EPISODE_SCORES.keys())
    data = [DEMO_EPISODE_SCORES[m] for m in model_names]

    parts = ax.violinplot(data, positions=range(len(model_names)),
                          showmedians=True, showextrema=True, widths=0.65)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(model_palette[i % len(model_palette)])
        pc.set_alpha(0.65)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color(median_color)
            parts[key].set_linewidth(1.2)

    rng2 = np.random.default_rng(7)
    for i, scores in enumerate(data):
        jitter = rng2.uniform(-0.1, 0.1, size=len(scores))
        ax.scatter(np.full(len(scores), i) + jitter, scores,
                   s=8, color=model_palette[i % len(model_palette)], alpha=0.55, zorder=3)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("CEPS Score", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color=colors["failed"], linestyle="--", linewidth=1, alpha=0.7, label="Pass threshold")
    ax.legend(fontsize=7)
    ax.set_title("CEPS Distribution (Violin)", fontsize=9, fontweight="bold")


def _draw_nav(ax, colors, stage_colors, regime_colors, model_palette):
    linestyles = ["-", "--", "-.", ":"]
    for i, (name, nav) in enumerate(DEMO_NAV.items()):
        c = model_palette[i % len(model_palette)]
        ls = linestyles[i % len(linestyles)]
        nav_norm = nav / nav.iloc[0] * 100
        ax.plot(nav_norm.index, nav_norm.values, label=name, color=c,
                linewidth=1.8, linestyle=ls)

    ax.axhline(100, color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.5, label="Start (100)")
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Normalized NAV", fontsize=8)
    ax.set_title("Portfolio NAV Curves", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.figure.autofmt_xdate()


def _draw_scatter(ax, colors, stage_colors, regime_colors, model_palette):
    for i, entry in enumerate(DEMO_CEPS_VS_PNL):
        c = model_palette[i % len(model_palette)] if entry["risk_gate_passed"] else colors["neutral"]
        mk = "o" if entry["risk_gate_passed"] else "x"
        ax.scatter(entry["mean_ceps"], entry["total_return"], color=c,
                   marker=mk, s=70, zorder=3)
        ax.annotate(entry["model_name"], (entry["mean_ceps"], entry["total_return"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7, color=c)

    xs = np.array([e["mean_ceps"] for e in DEMO_CEPS_VS_PNL])
    ys = np.array([e["total_return"] for e in DEMO_CEPS_VS_PNL])
    m_coef, b = np.polyfit(xs, ys, 1)
    x_line = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(x_line, m_coef * x_line + b, "--", color="#aaaaaa", linewidth=1, alpha=0.6)

    ax.axhline(0, color="#aaaaaa", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("Mean CEPS Score", fontsize=8)
    ax.set_ylabel("Total Return", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.set_title("CEPS Score vs. Realized Return", fontsize=9, fontweight="bold")


# ---------------------------------------------------------------------------
# Compose a full 5-panel preview figure for one theme
# ---------------------------------------------------------------------------

THEME_META = {
    "minimalist": {"cmap": "Greys",   "median": "#36454f", "label": "Modern Minimalist"},
    "galaxy":     {"cmap": "Purples", "median": "#2b1e3e", "label": "Midnight Galaxy"},
    "frost":      {"cmap": "Blues",   "median": "#1e3d6e", "label": "Arctic Frost"},
}

def generate_theme_preview(theme: str, out_path: Path) -> None:
    colors, stage_colors, regime_colors, model_palette = apply_style(theme)

    meta = THEME_META[theme]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"PortBench — {meta['label']} Theme",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    ax_radar   = fig.add_subplot(gs[0, 0], polar=True)
    ax_heat    = fig.add_subplot(gs[0, 1])
    ax_violin  = fig.add_subplot(gs[0, 2])
    ax_nav     = fig.add_subplot(gs[1, 0:2])
    ax_scatter = fig.add_subplot(gs[1, 2])

    kw = dict(colors=colors, stage_colors=stage_colors,
              regime_colors=regime_colors, model_palette=model_palette)

    _draw_radar(ax_radar, **kw)
    _draw_heatmap(ax_heat, **kw, cmap=meta["cmap"])
    _draw_violin(ax_violin, **kw, median_color=meta["median"])
    _draw_nav(ax_nav, **kw)
    _draw_scatter(ax_scatter, **kw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = Path("figures/theme_preview")

    print("Generating theme previews...")
    generate_theme_preview("minimalist", out_dir / "theme_minimalist.png")
    generate_theme_preview("galaxy",     out_dir / "theme_galaxy.png")
    generate_theme_preview("frost",      out_dir / "theme_frost.png")
    print(f"\nDone. Preview images are in {out_dir}/")
    print("  theme_minimalist.png  — Modern Minimalist")
    print("  theme_galaxy.png      — Midnight Galaxy")
    print("  theme_frost.png       — Arctic Frost")
