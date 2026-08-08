"""
Generate exp_cross_period_vs_ew.png from an experiment tree.

Usage:
  python examples/visualization/gen_cross_period_vs_ew.py \
      --experiments-dir EXPERIMENTS_rebuttal_lookback/monthly \
      --output D:/GitHub/yuxuan-emnlp26-portbench/figures/exp_cross_period_vs_ew.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from portbench.visualization.style import save_figure
from portbench.visualization.cross_period_plots import (
    plot_cross_period_vs_ew,
    DEFAULT_MODEL_ORDER,
)
from export_paper_figures import load_cross_period


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-dir",
        default="EXPERIMENTS_rebuttal_lookback/monthly",
    )
    parser.add_argument(
        "--output",
        default=r"D:\GitHub\yuxuan-emnlp26-portbench\figures\exp_cross_period_vs_ew.png",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiments_dir).resolve()
    data, eqw = load_cross_period(exp_dir)
    print(f"Models: {len(data)}; EqW: {eqw}")
    fig = plot_cross_period_vs_ew(data, eqw, model_order=DEFAULT_MODEL_ORDER)
    save_figure(fig, args.output, formats=("png",))
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
