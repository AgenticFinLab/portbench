"""
Live-eval CEPS figures for the paper appendix (LLM models only).

Reads outputs/live/daily_*/*/balanced/range_summary.json and writes:
  - live_ceps_bars.png
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

# Match short labels used in main tables where helpful
MODEL_ORDER = [
    "glm-5.1",
    "kimi-k2.6",
    "qwen3.6-plus",
    "qwen3.6-35b-a3b",
    "qwen3.7-max",
    "hy3-preview",
]

LOOKBACK_COLOR = "#1e3d6e"
EXPOST_COLOR = "#7a9fc5"
LINE_COLORS = [
    "#1e3d6e",
    "#4a6fa5",
    "#c44e52",
    "#55a868",
    "#8172b3",
    "#ccb974",
]


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


def plot_bars(summaries: list[dict], out_path: Path) -> None:
    apply_paper_style()
    by_model = {s["model"]: s for s in summaries}
    models = [m for m in MODEL_ORDER if m in by_model]
    labels = [MODEL_DISPLAY.get(m, m) for m in models]
    lb = [float(by_model[m]["mean_ceps_lookback"]) for m in models]
    ep = [float(by_model[m]["mean_ceps_ex_post"]) for m in models]

    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.bar(x - width / 2, lb, width, label="Lookback", color=LOOKBACK_COLOR, edgecolor="none")
    ax.bar(x + width / 2, ep, width, label="Ex-post", color=EXPOST_COLOR, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean CEPS")
    ax.set_ylim(0, max(max(lb), max(ep)) * 1.18)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_daily(summaries: list[dict], out_path: Path) -> None:
    apply_paper_style()
    by_model = {s["model"]: s for s in summaries}
    models = [m for m in MODEL_ORDER if m in by_model]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for i, m in enumerate(models):
        eps = by_model[m].get("episodes") or []
        dates = [e["decision_date"][5:] for e in eps]  # MM-DD
        vals = [float(e["ceps_lookback"]) for e in eps]
        ax.plot(
            dates,
            vals,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            color=LINE_COLORS[i % len(LINE_COLORS)],
            label=MODEL_DISPLAY.get(m, m),
        )
    ax.set_xlabel("Decision date (2026)")
    ax.set_ylabel("CEPS (lookback)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
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
    plot_bars(summaries, out_dir / "live_ceps_bars.png")
    plot_daily(summaries, out_dir / "live_ceps_daily.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
