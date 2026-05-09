"""
PortBench batch experiment framework.

Usage (CLI):
    python -m portbench.experiments --config configs/experiments/default.yaml
    python -m portbench.experiments --config x.yaml --dry-run

Usage (Python):
    from portbench.experiments import BatchRunner, ExperimentConfig
    cfg = ExperimentConfig.from_yaml("configs/experiments/default.yaml")
    BatchRunner(cfg).run()
"""

from .config import ExperimentConfig, ModelSpec, LoggingConfig, NormalPeriod
from .providers import (
    PROVIDER_REGISTRY,
    BASELINE_REGISTRY,
    build_adapter,
    build_baseline,
    build_mock,
    model_label,
)
from .runner import BatchRunner

__all__ = [
    "ExperimentConfig", "ModelSpec", "LoggingConfig", "NormalPeriod",
    "PROVIDER_REGISTRY", "BASELINE_REGISTRY",
    "build_adapter", "build_baseline", "build_mock", "model_label",
    "BatchRunner",
]
