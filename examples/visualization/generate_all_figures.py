"""
Generate all PortBench paper figures from evaluation outputs.

Usage:
    python examples/visualization/generate_all_figures.py
    python examples/visualization/generate_all_figures.py --results-dir outputs/eval_results
    python examples/visualization/generate_all_figures.py --figures 1,2,4 --format png

Output:
    figures/
        fig1_ceps_radar.{pdf,png}
        fig2_ceps_heatmap.{pdf,png}
        fig3_ceps_violin.{pdf,png}
        fig4_stress_gate.{pdf,png}
        fig5_risk_ranking.{pdf,png}
        fig6_dataset_stats.{pdf,png}
        fig7_regime_distributions.{pdf,png}
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.visualization import (
    plot_ceps_radar, plot_ceps_heatmap, plot_ceps_violin,
    plot_stress_gate, plot_risk_ranking,
    plot_dataset_stats,
    plot_regime_distributions, build_regime_data_from_mock,
    save_figure,
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_model_results(results_dir: Path) -> dict:
    """
    Scan results_dir for model subdirectories and load their JSON outputs.

    Returns:
        {
          "per_stage": {model_name: {"S1": float, ...}},
          "ceps_totals": {model_name: float},
          "episode_ceps": {model_name: [float, ...]},
          "stress": {model_name: [stress_result_dict, ...]},
          "ranking": [ranking_entry_dict, ...],
        }
    """
    per_stage    = {}
    ceps_totals  = {}
    episode_ceps = {}
    stress       = {}
    ranking      = []

    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"  Results dir not found: {results_dir}")
        return {}

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        # Per-stage scores
        ps_file = model_dir / "per_stage_scores.json"
        if ps_file.exists():
            data = json.loads(ps_file.read_text())
            per_stage[model_name] = data.get("mean", {})

        # CEPS scores
        ceps_file = model_dir / "ceps_scores.json"
        if ceps_file.exists():
            data = json.loads(ceps_file.read_text())
            ceps_totals[model_name] = data.get("mean_ceps", 0.0)

        # Per-episode CEPS computed from per_stage_scores (individual_results are objects)
        if ps_file.exists():
            ps_data   = json.loads(ps_file.read_text())
            per_ep    = ps_data.get("per_episode", {})
            stage_ids = ["S1", "S2", "S3", "S4", "S5"]
            n_ep = len(next(iter(per_ep.values()), []))
            ep_scores = []
            for i in range(n_ep):
                scores = [per_ep.get(s, [0.0] * n_ep)[i] for s in stage_ids if s in per_ep]
                if not scores:
                    continue
                avg = sum(scores) / len(scores)
                # Apply propagation penalty (weight=0.1, same as CEPS class)
                drops = sum(max(scores[j] - scores[j+1], 0) for j in range(len(scores)-1))
                ep_scores.append(max(0.0, min(1.0, avg - 0.1 * drops)))
            if ep_scores:
                episode_ceps[model_name] = ep_scores

        # Stress test results
        stress_file = model_dir / "stress_test_results.json"
        if stress_file.exists():
            stress[model_name] = json.loads(stress_file.read_text())

        # Risk-first ranking
        rank_file = model_dir / "risk_first_ranking.json"
        if rank_file.exists():
            entry = json.loads(rank_file.read_text())
            ranking.append(entry)

    return {
        "per_stage":    per_stage,
        "ceps_totals":  ceps_totals,
        "episode_ceps": episode_ceps,
        "stress":       stress,
        "ranking":      ranking,
    }


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def gen_fig1(data: dict, out: Path, fmt: tuple) -> bool:
    if not data.get("per_stage"):
        print("  [Fig 1] Skipped — no per_stage data")
        return False
    fig = plot_ceps_radar(data["per_stage"])
    save_figure(fig, str(out / "fig1_ceps_radar"), formats=fmt)
    return True


def gen_fig2(data: dict, out: Path, fmt: tuple) -> bool:
    if not data.get("per_stage"):
        print("  [Fig 2] Skipped — no per_stage data")
        return False
    fig = plot_ceps_heatmap(data["per_stage"], ceps_totals=data.get("ceps_totals"))
    save_figure(fig, str(out / "fig2_ceps_heatmap"), formats=fmt)
    return True


def gen_fig3(data: dict, out: Path, fmt: tuple) -> bool:
    ep = data.get("episode_ceps")
    if not ep:
        print("  [Fig 3] Skipped — no per-episode CEPS data (individual_results missing from ceps_scores.json)")
        return False
    fig = plot_ceps_violin(ep)
    save_figure(fig, str(out / "fig3_ceps_violin"), formats=fmt)
    return True


def gen_fig4(data: dict, out: Path, fmt: tuple) -> bool:
    if not data.get("stress"):
        print("  [Fig 4] Skipped — no stress test data")
        return False
    fig = plot_stress_gate(data["stress"])
    save_figure(fig, str(out / "fig4_stress_gate"), formats=fmt)
    return True


def gen_fig5(data: dict, out: Path, fmt: tuple) -> bool:
    if not data.get("ranking"):
        print("  [Fig 5] Skipped — no ranking data")
        return False
    fig = plot_risk_ranking(data["ranking"])
    save_figure(fig, str(out / "fig5_risk_ranking"), formats=fmt)
    return True


def gen_fig6(qa_stats_path: str, out: Path, fmt: tuple) -> bool:
    stats_path = Path(qa_stats_path)
    if not stats_path.exists():
        print(f"  [Fig 6] Skipped — QA stats file not found: {stats_path}")
        return False
    stats = json.loads(stats_path.read_text())
    fig = plot_dataset_stats(stats)
    save_figure(fig, str(out / "fig6_dataset_stats"), formats=fmt)
    return True


def gen_fig7(out: Path, fmt: tuple, seed: int = 42) -> bool:
    print("  [Fig 7] Building regime data from MockDataProvider…")
    regime_data = build_regime_data_from_mock(n_days=400, seed=seed)
    has_data = any(len(v) > 0 for v in regime_data.values())
    if not has_data:
        print("  [Fig 7] Skipped — no regime data generated")
        return False
    fig = plot_regime_distributions(regime_data)
    save_figure(fig, str(out / "fig7_regime_distributions"), formats=fmt)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FIGURE_MAP = {
    1: ("CEPS Radar Chart",          gen_fig1),
    2: ("CEPS Error Propagation Heatmap", gen_fig2),
    3: ("CEPS Violin Distribution",  gen_fig3),
    4: ("Stress Test Risk Gate",     gen_fig4),
    5: ("Risk-First Ranking",        gen_fig5),
    6: ("QA Dataset Statistics",     gen_fig6),
    7: ("Regime Return Distributions", gen_fig7),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PortBench paper figures")
    parser.add_argument("--results-dir", default="outputs/eval_results",
                        help="Directory containing per-model eval result subdirs")
    parser.add_argument("--qa-stats", default="outputs/qa_dataset/stats.json",
                        help="Path to QA dataset stats.json (for Fig 6)")
    parser.add_argument("--output-dir", default="figures",
                        help="Directory to write figures into")
    parser.add_argument("--format", default="both", choices=["pdf", "png", "both"],
                        help="Output format(s)")
    parser.add_argument("--figures", default="all",
                        help="Comma-separated figure numbers to generate (e.g. '1,2,5') or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    fmt = ("pdf", "png") if args.format == "both" else (args.format,)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.figures == "all":
        fig_ids = list(FIGURE_MAP.keys())
    else:
        fig_ids = [int(x.strip()) for x in args.figures.split(",")]

    print(f"PortBench Figure Generator")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Output dir  : {out.resolve()}")
    print(f"  Formats     : {fmt}")
    print(f"  Figures     : {fig_ids}")
    print("=" * 50)

    # Load eval results once
    data = load_model_results(args.results_dir)
    n_models = len(data.get("per_stage", {}))
    print(f"  Loaded results for {n_models} model(s): {list(data.get('per_stage', {}).keys())}")

    generated = []
    for fig_id in fig_ids:
        if fig_id not in FIGURE_MAP:
            print(f"  [Fig {fig_id}] Unknown figure ID, skipping")
            continue
        label, fn = FIGURE_MAP[fig_id]
        print(f"\n  Fig {fig_id}: {label}")
        try:
            # Fig 6 and 7 have special signatures
            if fig_id == 6:
                ok = fn(args.qa_stats, out, fmt)
            elif fig_id == 7:
                ok = fn(out, fmt, seed=args.seed)
            else:
                ok = fn(data, out, fmt)
            if ok:
                files = [str(out / f"fig{fig_id}_*.{f}") for f in fmt]
                print(f"    Saved → {', '.join(f'fig{fig_id}*.{f}' for f in fmt)}")
                generated.append(fig_id)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Generated {len(generated)}/{len(fig_ids)} figures in {out.resolve()}")


if __name__ == "__main__":
    main()
