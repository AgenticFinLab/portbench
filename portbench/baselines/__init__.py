"""
Baseline portfolio strategies for non-LLM comparison in PortBench.

These baselines serve as comparison points for LLM agent evaluation:
  - EqualWeightBaseline:    Classic 1/N portfolio — lowest complexity reference
  - SixtyFortyBaseline:     Traditional 60% equity / 40% bond allocation
  - RiskParityBaseline:     Weights inversely proportional to asset volatility

All baselines implement AgentAdapter so they can be dropped into EvalPipeline
as a direct replacement for any LLM adapter.
"""

from .base import BaselineStrategy, BaselineResult
from .equal_weight import EqualWeightBaseline
from .sixty_forty import SixtyFortyBaseline
from .risk_parity import RiskParityBaseline

__all__ = [
    "BaselineStrategy",
    "BaselineResult",
    "EqualWeightBaseline",
    "SixtyFortyBaseline",
    "RiskParityBaseline",
]
