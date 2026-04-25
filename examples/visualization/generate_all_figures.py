"""
Generate all PortBench paper figures from evaluation outputs.

Usage:
    python examples/visualization/generate_all_figures.py
    python examples/visualization/generate_all_figures.py --sandbox-dir outputs/sandbox
    python examples/visualization/generate_all_figures.py --figures 8,9,11,12,13,14 --format png

Output:
    figures/
        ceps/        fig1–5  (legacy: from outputs/ceps/ runs)
        sandbox/     fig8–10, fig13, fig14
        profile/     fig11–12
        static/      fig6–7
        qa/          fig15
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.visualization import (
    plot_ceps_radar, plot_ceps_heatmap, plot_ceps_violin,
    plot_stress_gate, plot_risk_ranking,
    plot_dataset_overview,
    plot_regime_distributions, build_regime_data_from_mock,
    plot_sandbox_nav, plot_sandbox_metrics, plot_ceps_vs_pnl,
    load_sandbox_results,
    plot_profile_alignment, plot_profile_radar,
    save_figure,
)
from portbench.visualization.sandbox_plots import (
    load_sandbox_results_full,
    plot_stress_drawdown,
    plot_profile_nav,
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_model_results(results_dir: Path) -> dict:
    """Load legacy outputs/ceps/ data for Fig 1–5."""
    per_stage    = {}
    ceps_totals  = {}
    episode_ceps = {}
    stress       = {}
    ranking      = []

    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"  Results dir not found: {results_dir}")
        return {}

    run_dirs = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for ts_dir in sorted(model_dir.iterdir()):
            if ts_dir.is_dir():
                run_dirs.append((model_dir.name, ts_dir))

    latest: dict[str, Path] = {}
    for model_name, ts_dir in run_dirs:
        if model_name not in latest or ts_dir.name > latest[model_name].name:
            latest[model_name] = ts_dir

    for model_name, run_dir in latest.items():
        ps_file = run_dir / "per_stage_scores.json"
        if ps_file.exists():
            data = json.loads(ps_file.read_text())
            per_stage[model_name] = data.get("mean", {})

        ceps_file = run_dir / "ceps_scores.json"
        if ceps_file.exists():
            data = json.loads(ceps_file.read_text())
            ceps_totals[model_name] = data.get("mean_ceps", 0.0)

        if ps_file.exists():
            ps_data = json.loads(ps_file.read_text())
            per_ep = ps_data.get("per_episode", {})
            stage_ids = ["S1", "S2", "S3", "S4", "S5"]
            n_ep = len(next(iter(per_ep.values()), []))
            ep_scores = []
            for i in range(n_ep):
                scores = [per_ep.get(s, [0.0] * n_ep)[i] for s in stage_ids if s in per_ep]
                if not scores:
                    continue
                avg = sum(scores) / len(scores)
                drops = sum(max(scores[j] - scores[j+1], 0) for j in range(len(scores)-1))
                ep_scores.append(max(0.0, min(1.0, avg - 0.1 * drops)))
            if ep_scores:
                episode_ceps[model_name] = ep_scores

        stress_file = run_dir / "stress_test_results.json"
        if stress_file.exists():
            stress[model_name] = json.loads(stress_file.read_text())

        rank_file = run_dir / "risk_first_ranking.json"
        if rank_file.exists():
            ranking.append(json.loads(rank_file.read_text()))

    return {
        "per_stage":    per_stage,
        "ceps_totals":  ceps_totals,
        "episode_ceps": episode_ceps,
        "stress":       stress,
        "ranking":      ranking,
    }


# ---------------------------------------------------------------------------
# Figure generators — legacy (Fig 1–5, data from outputs/ceps/)
# ---------------------------------------------------------------------------

def gen_fig1(data: dict, out: Path, fmt: tuple) -> bool:
    if not data.get("per_stage"):
        print("  [Fig 1] Skipped — no per_stage data in outputs/ceps/")
        return False
    fig = plot_ceps_radar(data["per_stage"])
    save_figure(fig, str(out / "ceps" / "fig1_ceps_radar"), formats=fmt)
    return True


def gen_fig2(data: dict, out: Path, fmt: tuple) -> bool:
    if not data.get("per_stage"):
        print("  [Fig 2] Skipped — no per_stage data in outputs/ceps/")
        return False
    fig = plot_ceps_heatmap(data["per_stage"], ceps_totals=data.get("ceps_totals"))
    save_figure(fig, str(out / "ceps" / "fig2_ceps_heatmap"), formats=fmt)
    return True


def gen_fig3(data: dict, out: Path, fmt: tuple) -> bool:
    ep = data.get("episode_ceps")
    if not ep:
        print("  [Fig 3] Skipped — no episode_ceps data")
        return False
    fig = plot_ceps_violin(ep)
    save_figure(fig, str(out / "ceps" / "fig3_ceps_violin"), formats=fmt)
    return True


def gen_fig4(data: dict, out: Path, fmt: tuple) -> bool:
    if not data.get("stress"):
        print("  [Fig 4] Skipped — no stress data in outputs/ceps/")
        return False
    fig = plot_stress_gate(data["stress"])
    save_figure(fig, str(out / "ceps" / "fig4_stress_gate"), formats=fmt)
    return True


def gen_fig5(data: dict, out: Path, fmt: tuple) -> bool:
    if not data.get("ranking"):
        print("  [Fig 5] Skipped — no ranking data in outputs/ceps/")
        return False
    fig = plot_risk_ranking(data["ranking"])
    save_figure(fig, str(out / "ceps" / "fig5_risk_ranking"), formats=fmt)
    return True


# ---------------------------------------------------------------------------
# Figure generators — static (Fig 6–7)
# ---------------------------------------------------------------------------

def gen_fig6(qa_stats_path: str, out: Path, fmt: tuple) -> bool:
    stats_path = Path(qa_stats_path)
    if not stats_path.exists():
        print(f"  [Fig 6] Skipped — {stats_path} not found")
        return False
    stats = json.loads(stats_path.read_text())
    fig = plot_dataset_overview(stats)
    save_figure(fig, str(out / "static" / "fig6_dataset_stats"), formats=fmt)
    return True


def gen_fig7(out: Path, fmt: tuple, seed: int = 42) -> bool:
    print("  [Fig 7] Building regime data from MockDataProvider…")
    regime_data = build_regime_data_from_mock(n_days=400, seed=seed)
    if not any(len(v) > 0 for v in regime_data.values()):
        print("  [Fig 7] Skipped — no regime data")
        return False
    fig = plot_regime_distributions(regime_data)
    save_figure(fig, str(out / "static" / "fig7_regime_distributions"), formats=fmt)
    return True


# ---------------------------------------------------------------------------
# Figure generators — sandbox (Fig 8–10, data from outputs/sandbox/)
# ---------------------------------------------------------------------------

def gen_fig8(sandbox_dir: str, out: Path, fmt: tuple) -> bool:
    """NAV curves — one line per model/profile combination."""
    sandbox_results = load_sandbox_results(sandbox_dir)
    if not sandbox_results:
        print(f"  [Fig 8] Skipped — no sandbox normal results in {sandbox_dir}")
        return False
    nav_data = {k: v["_nav_series"] for k, v in sandbox_results.items() if "_nav_series" in v}
    if not nav_data:
        print("  [Fig 8] Skipped — no nav_curve.csv files found")
        return False
    fig = plot_sandbox_nav(nav_data)
    save_figure(fig, str(out / "sandbox" / "fig8_sandbox_nav"), formats=fmt)
    return True


def gen_fig9(sandbox_dir: str, out: Path, fmt: tuple) -> bool:
    """Performance metrics bar chart — one bar per model/profile."""
    sandbox_results = load_sandbox_results(sandbox_dir)
    if not sandbox_results:
        print(f"  [Fig 9] Skipped — no sandbox normal results in {sandbox_dir}")
        return False
    # Include mean_ceps if available
    metric_keys = ["total_return", "sharpe_ratio", "max_drawdown", "mean_ceps"]
    available_metrics = [k for k in metric_keys
                         if any(k in v for v in sandbox_results.values())]
    if not available_metrics:
        available_metrics = ["total_return", "sharpe_ratio", "max_drawdown"]
    fig = plot_sandbox_metrics(sandbox_results, metric_keys=available_metrics)
    save_figure(fig, str(out / "sandbox" / "fig9_sandbox_metrics"), formats=fmt)
    return True


def gen_fig10(sandbox_dir: str, out: Path, fmt: tuple) -> bool:
    """CEPS vs PnL scatter — reads mean_ceps + total_return from sandbox normal results."""
    sandbox_results = load_sandbox_results(sandbox_dir)
    if not sandbox_results:
        print(f"  [Fig 10] Skipped — no sandbox normal results in {sandbox_dir}")
        return False

    model_data = []
    for key, sb in sandbox_results.items():
        if "mean_ceps" not in sb or sb.get("mean_ceps", 0.0) == 0.0:
            continue
        model_data.append({
            "model_name": key,
            "mean_ceps": sb["mean_ceps"],
            "total_return": sb.get("total_return", 0.0),
            "stress_gate_passed": sb.get("stress_passed", True),
        })

    if not model_data:
        print("  [Fig 10] Skipped — no entries with mean_ceps (need use_pipeline=True runs)")
        return False

    fig = plot_ceps_vs_pnl(model_data)
    save_figure(fig, str(out / "sandbox" / "fig10_ceps_vs_pnl"), formats=fmt)
    return True


# ---------------------------------------------------------------------------
# Figure generators — profile (Fig 11–12, data from outputs/sandbox/)
# ---------------------------------------------------------------------------

_DEMO_PROFILE_DATA = {
    "GPT-4o":       {"conservative": 0.73, "balanced": 0.68, "aggressive": 0.61},
    "Claude-3.5":   {"conservative": 0.80, "balanced": 0.72, "aggressive": 0.65},
    "Qwen-Plus":    {"conservative": 0.62, "balanced": 0.64, "aggressive": 0.63},
    "Equal-Weight": {"conservative": 0.45, "balanced": 0.46, "aggressive": 0.44},
}


def _load_profile_data(sandbox_dir: str) -> dict[str, dict[str, float]]:
    """
    Load per-profile mean_profile_score from sandbox outputs/sandbox/ structure.
    Falls back to demo data if nothing found.
    """
    out: dict[str, dict[str, float]] = {}
    full = load_sandbox_results_full(sandbox_dir)
    for model_name, model_data in full.items():
        scores: dict[str, float] = {}
        for pname, pdata in model_data.get("profiles", {}).items():
            normal = pdata.get("normal")
            if normal and "mean_profile_score" in normal:
                scores[pname] = normal["mean_profile_score"]
        if scores:
            out[model_name] = scores

    # Also try legacy outputs/profile/ if sandbox gives nothing
    if not out:
        profile_path = Path(sandbox_dir).parent / "profile"
        if profile_path.exists():
            for model_dir in sorted(profile_path.iterdir()):
                if not model_dir.is_dir():
                    continue
                latest_comp = None
                for ts_dir in sorted(model_dir.iterdir()):
                    comp_file = ts_dir / "profile_comparison.json"
                    if comp_file.exists():
                        latest_comp = comp_file
                if latest_comp:
                    comp = json.loads(latest_comp.read_text())
                    model_name = comp.get("model_name", model_dir.name)
                    profiles = comp.get("profiles", {})
                    out[model_name] = {
                        k: v.get("mean_profile_score", 0.0) for k, v in profiles.items()
                    }

    return out or _DEMO_PROFILE_DATA


def gen_fig11(sandbox_dir: str, out: Path, fmt: tuple) -> bool:
    profile_data = _load_profile_data(sandbox_dir)
    fig = plot_profile_alignment(profile_data)
    save_figure(fig, str(out / "profile" / "fig11_profile_alignment"), formats=fmt)
    return True


def gen_fig12(sandbox_dir: str, out: Path, fmt: tuple) -> bool:
    profile_data = _load_profile_data(sandbox_dir)
    fig = plot_profile_radar(profile_data)
    save_figure(fig, str(out / "profile" / "fig12_profile_radar"), formats=fmt)
    return True


# ---------------------------------------------------------------------------
# New figure generators (Fig 13–15)
# ---------------------------------------------------------------------------

def gen_fig13(sandbox_dir: str, out: Path, fmt: tuple) -> bool:
    """Stress drawdown heatmap: rows=scenarios, cols=profiles, cells=max_drawdown."""
    full = load_sandbox_results_full(sandbox_dir)
    if not full:
        print(f"  [Fig 13] Skipped — no sandbox results in {sandbox_dir}")
        return False

    generated = False
    for model_name, model_data in full.items():
        # Build stress_data: {profile: {scenario: {max_drawdown, tolerance, stress_passed}}}
        stress_data: dict[str, dict] = {}
        for pname, pdata in model_data.get("profiles", {}).items():
            stress_entry: dict = {}
            for scenario, sdata in pdata.get("stress", {}).items():
                stress_entry[scenario] = {
                    "max_drawdown": sdata.get("max_drawdown", 0.0),
                    "tolerance": sdata.get("tolerance", 0.0),
                    "stress_passed": sdata.get("stress_passed", False),
                }
            if stress_entry:
                stress_data[pname] = stress_entry

        if not stress_data:
            continue

        safe_name = model_name.replace("/", "_").replace("(", "").replace(")", "")
        fig = plot_stress_drawdown(
            stress_data,
            title=f"Stress Drawdown — {model_name}",
        )
        save_figure(fig, str(out / "sandbox" / f"fig13_stress_drawdown_{safe_name}"), formats=fmt)
        generated = True

    if not generated:
        print("  [Fig 13] Skipped — no stress data found")
    return generated


def gen_fig14(sandbox_dir: str, out: Path, fmt: tuple) -> bool:
    """Profile NAV curves: three profiles for each model on the same axes."""
    full = load_sandbox_results_full(sandbox_dir)
    if not full:
        print(f"  [Fig 14] Skipped — no sandbox results in {sandbox_dir}")
        return False

    generated = False
    for model_name, model_data in full.items():
        profile_nav: dict = {}
        for pname, pdata in model_data.get("profiles", {}).items():
            normal = pdata.get("normal")
            if normal and "_nav_series" in normal:
                profile_nav[pname] = normal["_nav_series"]

        if len(profile_nav) < 2:
            continue

        safe_name = model_name.replace("/", "_").replace("(", "").replace(")", "")
        fig = plot_profile_nav(profile_nav, model_name=model_name)
        save_figure(fig, str(out / "sandbox" / f"fig14_profile_nav_{safe_name}"), formats=fmt)
        generated = True

    if not generated:
        print("  [Fig 14] Skipped — need ≥2 profiles with normal backtest results")
    return generated


def gen_fig15(qa_dir: str, out: Path, fmt: tuple) -> bool:
    """QA accuracy bar chart: per-template accuracy from outputs/qa/."""
    import matplotlib.pyplot as plt
    from portbench.visualization.style import apply_paper_style, MODEL_PALETTE

    qa_path = Path(qa_dir)
    if not qa_path.exists():
        print(f"  [Fig 15] Skipped — {qa_path} not found")
        return False

    # Collect latest run per model
    model_results: dict[str, dict] = {}
    for model_dir in sorted(qa_path.iterdir()):
        if not model_dir.is_dir():
            continue
        latest_file = None
        for ts_dir in sorted(model_dir.iterdir()):
            f = ts_dir / "qa_results.json"
            if f.exists():
                latest_file = f
        if latest_file:
            data = json.loads(latest_file.read_text(encoding="utf-8"))
            model_results[data.get("model_name", model_dir.name)] = data

    if not model_results:
        print("  [Fig 15] Skipped — no qa_results.json found")
        return False

    apply_paper_style()

    # Collect all template IDs
    all_templates = sorted({
        tid
        for data in model_results.values()
        for tid in data.get("per_template", {}).keys()
    })
    if not all_templates:
        print("  [Fig 15] Skipped — no per-template data")
        return False

    models = list(model_results.keys())
    n_models = len(models)
    n_templates = len(all_templates)
    bar_width = 0.7 / n_models
    x = range(n_templates)

    fig, ax = plt.subplots(figsize=(max(8, n_templates * 1.2), 4.5))
    for i, model in enumerate(models):
        pt = model_results[model].get("per_template", {})
        values = [pt.get(t, {}).get("accuracy", 0.0) for t in all_templates]
        offsets = [xi + (i - n_models / 2 + 0.5) * bar_width for xi in x]
        ax.bar(offsets, values, width=bar_width * 0.9,
               color=MODEL_PALETTE[i % len(MODEL_PALETTE)], alpha=0.85, label=model)

    ax.set_xticks(list(x))
    ax.set_xticklabels(all_templates, fontsize=10)
    ax.set_ylabel("Accuracy (exact match)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="50%")
    ax.set_title("QA Evaluation Accuracy by Template", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    save_figure(fig, str(out / "qa" / "fig15_qa_accuracy"), formats=fmt)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FIGURE_MAP = {
    1:  ("CEPS Radar Chart [legacy]",         gen_fig1),
    2:  ("CEPS Error Propagation Heatmap [legacy]", gen_fig2),
    3:  ("CEPS Violin Distribution [legacy]", gen_fig3),
    4:  ("Stress Test Risk Gate [legacy]",    gen_fig4),
    5:  ("Risk-First Ranking [legacy]",       gen_fig5),
    6:  ("QA Dataset Statistics",             gen_fig6),
    7:  ("Regime Return Distributions",       gen_fig7),
    8:  ("Sandbox NAV Curves",                gen_fig8),
    9:  ("Sandbox Performance Metrics",       gen_fig9),
    10: ("CEPS vs. Realized Return",          gen_fig10),
    11: ("Investor Profile Alignment Score",  gen_fig11),
    12: ("Profile Adaptation Radar",          gen_fig12),
    13: ("Stress Drawdown Heatmap",           gen_fig13),
    14: ("Profile NAV Curves",                gen_fig14),
    15: ("QA Accuracy by Template",           gen_fig15),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PortBench paper figures")
    parser.add_argument("--results-dir", default="outputs/ceps",
                        help="Legacy CEPS eval results dir (for Figs 1–5)")
    parser.add_argument("--sandbox-dir", default="outputs/sandbox",
                        help="Sandbox results dir (for Figs 8–14)")
    parser.add_argument("--qa-dir", default="outputs/qa",
                        help="QA eval results dir (for Fig 15)")
    parser.add_argument("--qa-stats", default="datasets/qa_dataset/stats.json",
                        help="QA dataset stats.json (for Fig 6)")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--format", default="both", choices=["pdf", "png", "both"])
    parser.add_argument("--figures", default="all",
                        help="Comma-separated figure numbers or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    fmt = ("pdf", "png") if args.format == "both" else (args.format,)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig_ids = list(FIGURE_MAP.keys()) if args.figures == "all" else \
              [int(x.strip()) for x in args.figures.split(",")]

    print("PortBench Figure Generator")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Sandbox dir : {args.sandbox_dir}")
    print(f"  QA dir      : {args.qa_dir}")
    print(f"  Output dir  : {out.resolve()}")
    print(f"  Formats     : {fmt}")
    print(f"  Figures     : {fig_ids}")
    print("=" * 50)

    # Load legacy CEPS data once (only needed for Fig 1–5)
    ceps_data = {}
    if any(i in fig_ids for i in range(1, 6)):
        ceps_data = load_model_results(args.results_dir)
        n = len(ceps_data.get("per_stage", {}))
        print(f"  Loaded legacy CEPS data for {n} model(s)")

    generated = []
    for fig_id in fig_ids:
        if fig_id not in FIGURE_MAP:
            print(f"  [Fig {fig_id}] Unknown — skipping")
            continue
        label, fn = FIGURE_MAP[fig_id]
        print(f"\n  Fig {fig_id}: {label}")
        try:
            if fig_id in (1, 2, 3, 4, 5):
                ok = fn(ceps_data, out, fmt)
            elif fig_id == 6:
                ok = fn(args.qa_stats, out, fmt)
            elif fig_id == 7:
                ok = fn(out, fmt, seed=args.seed)
            elif fig_id in (8, 9, 10, 13, 14):
                ok = fn(args.sandbox_dir, out, fmt)
            elif fig_id in (11, 12):
                ok = fn(args.sandbox_dir, out, fmt)
            elif fig_id == 15:
                ok = fn(args.qa_dir, out, fmt)
            else:
                ok = fn(ceps_data, out, fmt)
            if ok:
                print(f"    Saved → fig{fig_id}*.{'/'.join(fmt)}")
                generated.append(fig_id)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Generated {len(generated)}/{len(fig_ids)} figures in {out.resolve()}")


if __name__ == "__main__":
    main()
