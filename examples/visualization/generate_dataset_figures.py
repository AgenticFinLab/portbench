"""
Generate QA dataset visualizations into datasets/visualization/.

Produces:
  datasets/visualization/
    fig_dataset_overview.png        — 3×2 comprehensive stats panel
    fig_dataset_regime_heatmap.png  — template × regime count heatmap
    fig_qa_samples.png              — 4×2 card grid (one example per template)
    fig_qa_sample_T1.png … T7.png   — standalone cards for each template

Usage:
    python examples/visualization/generate_dataset_figures.py
    python examples/visualization/generate_dataset_figures.py \\
        --qa-dir outputs/qa_dataset \\
        --output-dir datasets/visualization \\
        --format both
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.visualization import (
    plot_dataset_overview,
    plot_regime_heatmap,
    plot_qa_sample_cards,
    plot_single_card,
    save_figure,
)


_TEMPLATE_IDS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]


def load_one_sample_per_template(qa_dir: Path) -> dict[str, dict]:
    """
    Scan all_pairs.jsonl and return the first QA pair found for each template.
    Prefers samples with a non-empty context_summary and reasonable question length.
    """
    samples: dict[str, dict] = {}
    candidates: dict[str, list[dict]] = {t: [] for t in _TEMPLATE_IDS}

    pairs_file = qa_dir / "all_pairs.jsonl"
    if not pairs_file.exists():
        print(f"  WARNING: {pairs_file} not found — sample cards will be empty")
        return {}

    print(f"  Scanning {pairs_file} for representative samples…")
    with open(pairs_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            tmpl = item.get("template")
            if tmpl not in candidates:
                continue
            # Collect up to 20 candidates per template
            if len(candidates[tmpl]) < 50:
                candidates[tmpl].append(item)

    # Pick the best candidate per template:
    # prefer non-empty context_summary + short question (cleaner for display)
    for tmpl, cands in candidates.items():
        if not cands:
            continue
        scored = []
        for c in cands:
            has_news = "Recent filing/news:" in c.get("question", "")
            ctx_ok   = 1 if c.get("context_summary") else 0
            q_len    = len(c.get("question", ""))
            exp_ok   = 1 if c.get("explanation") else 0
            # Strongly prefer samples with embedded news/filing text
            score    = has_news * 500 + ctx_ok * 100 + exp_ok * 50 - max(0, q_len - 400) // 10
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        samples[tmpl] = scored[0][1]
        print(f"    {tmpl}: selected id={samples[tmpl].get('id', '?')} "
              f"(regime={samples[tmpl].get('market_regime')}, "
              f"date={samples[tmpl].get('decision_date')})")

    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Generate QA dataset figures")
    parser.add_argument("--qa-dir", default="datasets/qa_dataset",
                        help="Directory containing all_pairs.jsonl and stats.json")
    parser.add_argument("--output-dir", default="figures/datasets",
                        help="Output directory for figures")
    parser.add_argument("--format", default="png", choices=["pdf", "png", "both"],
                        help="Output format(s)")
    parser.add_argument("--standalone-cards", action="store_true",
                        help="Also generate individual card figures for T1–T7")
    return parser.parse_args()


def main():
    args = parse_args()
    fmt = ("pdf", "png") if args.format == "both" else (args.format,)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    qa_dir = Path(args.qa_dir)

    print("PortBench Dataset Figure Generator")
    print(f"  QA dir     : {qa_dir}")
    print(f"  Output dir : {out.resolve()}")
    print(f"  Format     : {fmt}")
    print("=" * 50)

    # ---- Load stats ----
    stats_file = qa_dir / "stats.json"
    if not stats_file.exists():
        print(f"  ERROR: stats.json not found at {stats_file}")
        sys.exit(1)
    stats = json.loads(stats_file.read_text(encoding="utf-8"))

    # ---- Figure A: Dataset overview (3×2 panel) ----
    print("\n  [A] Dataset overview panel…")
    fig = plot_dataset_overview(stats)
    save_figure(fig, str(out / "fig_dataset_overview"), formats=fmt)
    print(f"    Saved → fig_dataset_overview.{fmt[0]}")

    # ---- Figure B: Regime heatmap ----
    print("\n  [B] Template × regime heatmap…")
    fig = plot_regime_heatmap(stats)
    save_figure(fig, str(out / "fig_dataset_regime_heatmap"), formats=fmt)
    print(f"    Saved → fig_dataset_regime_heatmap.{fmt[0]}")

    # ---- Load samples ----
    samples = load_one_sample_per_template(qa_dir)

    # ---- Figure C: 4×2 sample card grid ----
    if samples:
        print("\n  [C] QA sample card grid (7 templates)…")
        fig = plot_qa_sample_cards(samples)
        save_figure(fig, str(out / "fig_qa_samples"), formats=fmt)
        print(f"    Saved → fig_qa_samples.{fmt[0]}")

        # ---- Figure D (optional): Standalone cards ----
        if args.standalone_cards:
            print("\n  [D] Standalone cards per template…")
            for tmpl, sample in sorted(samples.items()):
                fig = plot_single_card(sample)
                save_figure(fig, str(out / f"fig_qa_sample_{tmpl}"), formats=fmt)
                print(f"    Saved → fig_qa_sample_{tmpl}.{fmt[0]}")
    else:
        print("\n  [C/D] Skipped — no samples loaded")

    print(f"\nDone. Figures written to {out.resolve()}")


if __name__ == "__main__":
    main()
