"""
Pure-Python entry point for batch experiments. Equivalent to:
    python -m portbench.experiments --config configs/experiments/default.yaml

Useful for IDE debugging and programmatic config construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.experiments import BatchRunner, ExperimentConfig


def main():
    cfg_path = Path("configs/experiments/default.yaml")
    raw = cfg_path.read_text(encoding="utf-8")
    cfg = ExperimentConfig.from_yaml(cfg_path)
    BatchRunner(cfg, raw_yaml=raw).run()


if __name__ == "__main__":
    main()
