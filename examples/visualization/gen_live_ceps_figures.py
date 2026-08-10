"""
Live-eval figures for the paper appendix (LLM models only).

Reads outputs/live/daily_*/*/balanced/range_summary.json and writes:
  - live_stage_scores.png  (mean lookback S1--S5; not redundant with CEPS table)
  - live_ceps_daily.png
into the paper figures/ directory (or --out-dir).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.visualization.style import apply_paper_style

MODEL_DISPLAY = {
    "hy3-preview": "HY3-Preview",
    "qwen3.7-max": "Qwen3.7-Max",
    "qwen3.6-plus": "Qwen3.6-Plus",
    "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B",
    "glm-5.1": "GLM-5.1",
    "kimi-k2.6": "Kimi-K2.6",
}

# Rank by mean lookback CEPS (descending), matching Table tab:live_ceps
MODEL_ORDER = [
    "kimi-k2.6",
    "glm-5.1",
    "qwen3.6-plus",
    "qwen3.6-35b-a3b",
    "qwen3.7-max",
    "hy3-preview",
]

STAGES = ["S1", "S2", "S3", "S4", "S5"]
# Okabe–Ito / print-friendly categorical (colorblind-safe)
LINE_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # amber
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
]
LINE_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _load_llm_summaries(live_root: Path) -> list[dict]:
    rows = []
    for prov_dir in sorted(live_root.iterdir()):
        if not prov_dir.is_dir() or prov_dir.name == "baseline":
            continue
        for model_dir in sorted(prov_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            summary_path = model_dir / "balanced" / "range_summary.json"
            if not summary_path.exists():
                continue
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(s)
    return rows


def _mean_stage_matrix(summaries: list[dict], models: list[str]) -> np.ndarray:
    """models × stages mean lookback stage scores over episodes."""
    by_model = {s["model"]: s for s in summaries}
    mat = np.zeros((len(models), len(STAGES)))
    for i, m in enumerate(models):
        eps = by_model[m].get("episodes") or []
        for j, stage in enumerate(STAGES):
            vals = [
                float(e["stage_scores_lookback"][stage])
                for e in eps
                if e.get("stage_scores_lookback")
                and stage in e["stage_scores_lookback"]
            ]
            mat[i, j] = float(np.mean(vals)) if vals else np.nan
    return mat


def plot_stage_heatmap(summaries: list[dict], out_path: Path) -> None:
    apply_paper_style()
    by_model = {s["model"]: s for s in summaries}
    models = [m for m in MODEL_ORDER if m in by_model]
    labels = [MODEL_DISPLAY.get(m, m) for m in models]
    mat = _mean_stage_matrix(summaries, models)

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(STAGES)))
    ax.set_xticklabels(STAGES)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Pipeline stage (lookback oracle)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            # Dark cells → white text
            color = "white" if val >= 0.55 else "#1a1a1a"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean stage score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_daily(summaries: list[dict], out_path: Path) -> None:
    apply_paper_style()
    by_model = {s["model"]: s for s in summaries}
    models = [m for m in MODEL_ORDER if m in by_model]
    ink = "#1A1A1A"

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "axes.grid": False,
        "axes.linewidth": 1.05,
        "axes.edgecolor": ink,
    }):
        fig, ax = plt.subplots(figsize=(7.0, 3.7))
        for i, m in enumerate(models):
            eps = by_model[m].get("episodes") or []
            dates = [e["decision_date"][5:] for e in eps]  # MM-DD
            vals = [float(e["ceps_lookback"]) for e in eps]
            color = LINE_COLORS[i % len(LINE_COLORS)]
            marker = LINE_MARKERS[i % len(LINE_MARKERS)]
            ax.plot(
                dates,
                vals,
                marker=marker,
                markersize=5.5,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.9,
                linewidth=2.0,
                color=color,
                label=MODEL_DISPLAY.get(m, m),
                zorder=3 + i * 0.01,
            )
        ax.set_xlabel("Decision date (2026)", fontsize=11.0, color="black")
        ax.set_ylabel("CEPS (lookback)", fontsize=11.0, color="black")
        ax.tick_params(axis="x", rotation=35, labelsize=9.5, length=3.5, pad=2)
        ax.tick_params(axis="y", labelsize=9.5, length=3.5, pad=2)
        ax.yaxis.grid(True, linestyle="-", linewidth=0.35, alpha=0.25, color="#B8B8B8", zorder=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(1.05)
            ax.spines[spine].set_color(ink)

        # Legend below the axes (frameless) so it does not cover trajectories
        ax.legend(
            frameon=False,
            ncol=3,
            fontsize=9.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            handlelength=1.8,
            columnspacing=1.15,
            handletextpad=0.4,
        )
        fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.28)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(f"wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--live-root",
        default="outputs/live/daily_2026-07-16_2026-07-31",
    )
    p.add_argument(
        "--out-dir",
        default=r"D:\GitHub\yuxuan-emnlp26-portbench\figures",
    )
    args = p.parse_args()
    live_root = Path(args.live_root)
    out_dir = Path(args.out_dir)
    summaries = _load_llm_summaries(live_root)
    if not summaries:
        raise SystemExit(f"No LLM range_summary.json under {live_root}")
    plot_stage_heatmap(summaries, out_dir / "live_stage_scores.png")
    plot_daily(summaries, out_dir / "live_ceps_daily.png")
    # Remove obsolete bars figure if present
    old = out_dir / "live_ceps_bars.png"
    if old.exists():
        old.unlink()
        print(f"removed {old}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
