"""
Regenerate paper-facing PNGs into the EMNLP paper figures/ directory.

Usage (from portbench repo root):
  python examples/visualization/export_paper_figures.py \
      --experiments-dir EXPERIMENTS_rebuttal_lookback/monthly \
      --out-dir D:/GitHub/yuxuan-emnlp26-portbench/figures
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from portbench.visualization.style import save_figure
from portbench.visualization.stress_plots import (
    plot_stress_continuous_heatmap,
    plot_stress_drawdown_bars,
)
from portbench.visualization.cross_period_plots import (
    plot_cross_period_vs_ew,
    DEFAULT_MODEL_ORDER,
)
from portbench.visualization.normal_vs_stress_plots import plot_normal_vs_stress_scatter
from portbench.agent_eval.investor_profiles import PROFILES

# Profile adaptation helpers live alongside this script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_profile_adaptation import load_pas_data, make_figure  # noqa: E402

PERIOD_DIRS = {
    "2015-16": "stress_2015_china_shock",
    "2020": "stress_2020_covid_flash_crash",
    "2022": "stress_2022_crypto_collapse",
    "2024": "normal_bull_2024",
}
PROFILES_ORDER = ["conservative", "balanced", "aggressive"]


def _strip_date(name: str) -> str:
    return re.sub(r"-\d{6,8}$", "", name)


def _iter_model_runs(exp_dir: Path):
    """Yield (provider, model_name, run_dir)."""
    for provider_dir in sorted(exp_dir.iterdir()):
        if not provider_dir.is_dir() or provider_dir.name.startswith(("_", ".")):
            continue
        if provider_dir.name in {"comparison_figures", "lambda_sweep"}:
            continue
        for model_dir in sorted(provider_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            runs = sorted(
                [d for d in model_dir.iterdir() if d.is_dir()],
                key=lambda p: p.name,
                reverse=True,
            )
            if not runs:
                continue
            yield provider_dir.name, model_dir.name, runs[0]


def load_stress_continuous(exp_dir: Path) -> dict:
    """Build continuous_data for plot_stress_continuous_heatmap."""
    continuous: dict[str, dict[str, dict[str, dict]]] = {}
    for provider, model_name, run_dir in _iter_model_runs(exp_dir):
        mlabel = f"{provider}/{model_name}"
        model_entry: dict[str, dict[str, dict]] = {}
        for profile in PROFILES_ORDER:
            p_dir = run_dir / profile
            if not p_dir.is_dir():
                continue
            tol = PROFILES[profile].max_drawdown_tolerance
            sc_entry: dict[str, dict] = {}
            for sc_dir in p_dir.iterdir():
                if not sc_dir.is_dir() or not sc_dir.name.startswith("stress_"):
                    continue
                br_path = sc_dir / "backtest_result.json"
                if not br_path.exists():
                    continue
                payload = json.loads(br_path.read_text(encoding="utf-8"))
                dd = float(payload.get("max_drawdown", 0.0))
                stored = payload.get("dd_score")
                if stored is None or stored == 0.0:
                    dd_score = max(0.0, 1.0 - abs(dd) / max(tol, 1e-6))
                else:
                    dd_score = float(stored)
                sc_key = sc_dir.name.removeprefix("stress_")
                passed = bool(payload.get("stress_passed", abs(dd) <= tol + 1e-9))
                sc_entry[sc_key] = {
                    "dd_score": dd_score,
                    "passed": passed,
                    "max_drawdown": dd,
                }
            if sc_entry:
                model_entry[profile] = sc_entry
        if model_entry:
            continuous[mlabel] = model_entry
    return continuous


def load_cross_period(exp_dir: Path):
    """Return (data, eqw_sharpe) for plot_cross_period_vs_ew."""
    data: dict[str, dict[str, dict[str, float]]] = {}
    eqw: dict[str, float] = {}

    for provider, model_name, run_dir in _iter_model_runs(exp_dir):
        mlabel = f"{provider}/{_strip_date(model_name)}"
        is_eqw = provider == "baseline" and "equal_weight" in model_name
        for profile in PROFILES_ORDER:
            p_dir = run_dir / profile
            if not p_dir.is_dir():
                continue
            for per, dirname in PERIOD_DIRS.items():
                # normal may be normal_bull_2024 or just normal_*
                candidates = [p_dir / dirname]
                if per == "2024":
                    candidates.extend(sorted(p_dir.glob("normal*")))
                br = None
                for c in candidates:
                    if c.is_dir() and (c / "backtest_result.json").exists():
                        br = json.loads((c / "backtest_result.json").read_text(encoding="utf-8"))
                        break
                if br is None:
                    continue
                sharpe = float(br.get("sharpe_ratio", 0.0))
                if is_eqw:
                    # EqW is profile-independent; keep one value per period
                    eqw[per] = sharpe
                elif not provider == "baseline":
                    data.setdefault(mlabel, {}).setdefault(per, {})[profile] = sharpe
    return data, eqw


def load_normal_vs_stress(exp_dir: Path) -> list[dict]:
    points = []
    for provider, model_name, run_dir in _iter_model_runs(exp_dir):
        if provider == "baseline":
            continue
        mlabel = f"{provider}/{model_name}"
        p_dir = run_dir / "conservative"
        if not p_dir.is_dir():
            continue
        normal_dirs = sorted(p_dir.glob("normal*"))
        if not normal_dirs:
            continue
        normal_br_path = normal_dirs[0] / "backtest_result.json"
        crypto_br_path = p_dir / "stress_2022_crypto_collapse" / "backtest_result.json"
        if not normal_br_path.exists() or not crypto_br_path.exists():
            continue
        normal_br = json.loads(normal_br_path.read_text(encoding="utf-8"))
        crypto_br = json.loads(crypto_br_path.read_text(encoding="utf-8"))
        ceps_n = float(normal_br.get("mean_ceps") or 0.0)
        ceps_c = float(crypto_br.get("mean_ceps") or 0.0)
        if ceps_n <= 0 or ceps_c <= 0:
            continue
        tol = float(crypto_br.get("drawdown_tolerance") or PROFILES["conservative"].max_drawdown_tolerance)
        dd = float(crypto_br.get("max_drawdown") or 0.0)
        # Global gate: fail if any stress scenario under any profile fails
        # For the scatter we use conservative-profile gate (all 3 scenarios under cons)
        gate_ok = True
        for sc in (
            "stress_2015_china_shock",
            "stress_2020_covid_flash_crash",
            "stress_2022_crypto_collapse",
        ):
            sc_path = p_dir / sc / "backtest_result.json"
            if not sc_path.exists():
                continue
            sc_br = json.loads(sc_path.read_text(encoding="utf-8"))
            sc_dd = float(sc_br.get("max_drawdown") or 0.0)
            sc_tol = float(sc_br.get("drawdown_tolerance") or tol)
            if abs(sc_dd) > sc_tol + 1e-9:
                gate_ok = False
                break
        points.append({
            "model": mlabel,
            "ceps_normal": ceps_n,
            "ceps_crypto": ceps_c,
            "stress_gate_passed": gate_ok,
        })
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-dir",
        default="EXPERIMENTS_rebuttal_lookback/monthly",
    )
    parser.add_argument(
        "--out-dir",
        default=r"D:\GitHub\yuxuan-emnlp26-portbench\figures",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiments_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiments: {exp_dir}")
    print(f"Output:      {out_dir}")

    # 1. Stress drawdown bars (main text); keep heatmap for appendix/archive
    continuous = load_stress_continuous(exp_dir)
    print(f"Stress models: {len(continuous)}")
    fig = plot_stress_drawdown_bars(continuous)
    save_figure(fig, str(out_dir / "exp_stress_drawdown.png"), formats=("png",))
    print("Wrote exp_stress_drawdown.png")
    fig = plot_stress_continuous_heatmap(continuous)
    save_figure(fig, str(out_dir / "exp_stress_drawdown_heatmap.png"), formats=("png",))
    print("Wrote exp_stress_drawdown_heatmap.png (appendix archive)")

    # 2. Cross-period vs EqW
    data, eqw = load_cross_period(exp_dir)
    print(f"Cross-period models: {len(data)}; EqW periods: {list(eqw)}")
    # Align DEFAULT_MODEL_ORDER keys (may include provider/)
    fig = plot_cross_period_vs_ew(data, eqw, model_order=DEFAULT_MODEL_ORDER)
    save_figure(fig, str(out_dir / "exp_cross_period_vs_ew.png"), formats=("png",))
    # Also keep a copy under comparison_figures
    save_figure(fig, str(exp_dir / "comparison_figures" / "cross_period_vs_ew.png"), formats=("png",))
    print("Wrote exp_cross_period_vs_ew.png")

    # 3. Normal vs stress
    ns_points = load_normal_vs_stress(exp_dir)
    print(f"Normal-vs-stress points: {len(ns_points)}")
    fig = plot_normal_vs_stress_scatter(ns_points)
    save_figure(fig, str(out_dir / "analysis_normal_vs_stress.png"), formats=("png",))
    print("Wrote analysis_normal_vs_stress.png")

    # 4. Profile adaptation
    pas = load_pas_data(str(exp_dir))
    make_figure(pas, str(out_dir / "analysis_profile_adaptation.png"))
    print("Wrote analysis_profile_adaptation.png")

    print("Done.")


if __name__ == "__main__":
    main()
