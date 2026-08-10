"""
Export polished NAV figures into the paper figures/ directory (NAV-only, no CEPS panel).

Usage:
  python examples/visualization/gen_paper_nav_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from portbench.experiments.figures import (  # noqa: E402
    _load_normal_nav,
    _load_stress_nav,
    render_batch_comparison_figures,
)
from portbench.experiments import paths as _paths  # noqa: E402
from portbench.visualization.sandbox_plots import plot_sandbox_nav  # noqa: E402
from portbench.visualization.style import save_figure  # noqa: E402

PAPER_FIG = Path(r"D:\GitHub\yuxuan-emnlp26-portbench\figures")
EXP_ROOT = "EXPERIMENTS_rebuttal_lookback"
REBALANCE = "monthly"

# paper filename -> (profile, stress_scenario|None, legend_loc, legend_bbox,
#                    legend_col_align, figsize, events)
_DEFAULT_FIGSIZE = (7.2, 4.15)
EXPORTS = [
    ("exp_nav_conservative.png", "conservative", None, "upper left", None, "top", _DEFAULT_FIGSIZE, None),
    ("exp_nav_balanced.png", "balanced", None, "upper left", None, "top", _DEFAULT_FIGSIZE, None),
    ("exp_nav_aggressive.png", "aggressive", None, "upper left", None, "top", _DEFAULT_FIGSIZE, None),
    ("exp_nav_stress_china_cons.png", "conservative", "2015_china_shock", "upper left", None, "top", _DEFAULT_FIGSIZE, None),
    ("exp_nav_stress_covid_bal.png", "balanced", "2020_covid_flash_crash", "upper left", None, "top", _DEFAULT_FIGSIZE, None),
    (
        "exp_nav_stress_crypto_cons.png",
        "conservative",
        "2022_crypto_collapse",
        "lower left",
        None,
        "bottom",
        (7.2, 3.75),
        # Early / early-mid / mid / late: crypto ignition → credit contagion
        # → macro repricing → exchange collapse
        [
            ("2022-05-09", "Terra/LUNA"),
            ("2022-06-18", "3AC / Celsius"),
            ("2022-08-26", "Jackson Hole"),
            ("2022-11-11", "FTX"),
        ],
    ),
]


def _discover_timestamps(rebal_dir: Path) -> dict[str, str]:
    run_timestamps: dict[str, str] = {}
    for prov_dir in sorted(rebal_dir.iterdir()):
        if (
            not prov_dir.is_dir()
            or prov_dir.name.startswith("_")
            or prov_dir.name == "comparison_figures"
        ):
            continue
        for m_dir in sorted(prov_dir.iterdir()):
            if not m_dir.is_dir():
                continue
            key = f"{prov_dir.name}/{m_dir.name}"
            ts = _paths.find_best_run(EXP_ROOT, REBALANCE, prov_dir.name, m_dir.name, [])
            if ts:
                run_timestamps[key] = ts
    return run_timestamps


def _collect_nav(
    run_timestamps: dict[str, str],
    profile: str,
    scenario: str | None,
) -> dict[str, pd.Series]:
    nav_map: dict[str, pd.Series] = {}
    for model_key, timestamp in run_timestamps.items():
        prov, model = model_key.split("/", 1)
        r_dir = _paths.run_dir(EXP_ROOT, REBALANCE, prov, model, timestamp)
        p_dir = r_dir / profile
        if not p_dir.is_dir():
            continue
        if scenario is None:
            nav = _load_normal_nav(p_dir)
        else:
            stress = _load_stress_nav(p_dir) or {}
            nav = stress.get(scenario)
        if nav is not None and len(nav) > 1:
            # Strip date suffix from model folder for stable abbrev keys
            import re
            clean = re.sub(r"-\d{6,8}$", "", model)
            label = f"{prov}/{clean}"
            nav_map[label] = nav
    return nav_map


def main() -> int:
    rebal = Path(EXP_ROOT) / REBALANCE
    ts = _discover_timestamps(rebal)
    print(f"models: {len(ts)}")
    PAPER_FIG.mkdir(parents=True, exist_ok=True)

    for (
        fname, profile, scenario, legend_loc, legend_bbox,
        legend_col_align, figsize, events,
    ) in EXPORTS:
        nav_map = _collect_nav(ts, profile, scenario)
        if not nav_map:
            print(f"SKIP {fname}: no data")
            continue
        fig = plot_sandbox_nav(
            nav_map,
            title="",
            figsize=figsize,
            ceps_data=None,
            legend_loc=legend_loc,
            legend_bbox=legend_bbox,
            legend_col_align=legend_col_align,
            show_title=False,
            event_markers=events,
        )
        out = PAPER_FIG / fname
        save_figure(fig, str(out), formats=("png",))
        print(f"wrote {out.name}  (n={len(nav_map)}, legend={legend_loc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
