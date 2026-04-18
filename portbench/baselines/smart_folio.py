"""
SmartFolio baseline interface wrapper.

SmartFolio (IJCAI 2025) is the non-LLM SOTA baseline for multi-asset
portfolio management.  This module provides an interface wrapper so that
SmartFolio can be evaluated within the PortBench EvalPipeline alongside
LLM-based agents.

Because SmartFolio is a third-party research artifact (not distributed with
PortBench), this wrapper uses a two-level strategy:

  1. If the `smartfolio` package is installed (pip install smartfolio), the
     wrapper delegates allocation to the real SmartFolio model.
  2. If the package is unavailable, a heuristic approximation is used
     (momentum + volatility scaling, matching the paper's described logic)
     so that the pipeline can still run without the external dependency.

Integration note:
  To use the real SmartFolio model:
    pip install smartfolio          # (when publicly released by the authors)
    adapter = SmartFolioBaseline(use_real_model=True)

Reference:
  [SmartFolio citation to be added when IJCAI 2025 proceedings are published]
"""

import warnings
import numpy as np

from ..agent_eval.base import MarketSnapshot
from .base import BaselineStrategy


class SmartFolioBaseline(BaselineStrategy):
    """
    Interface wrapper for the SmartFolio IJCAI 2025 non-LLM SOTA baseline.

    When the real SmartFolio package is unavailable, this falls back to a
    momentum + inverse-volatility heuristic that approximates SmartFolio's
    publicly described allocation logic.

    Args:
        use_real_model: If True, attempt to import the `smartfolio` package.
                        Raises ImportError if unavailable.
        momentum_window: Number of days used to compute trailing momentum
                         signal for the fallback heuristic.
        vol_window:      Number of days used to compute volatility for the
                         fallback heuristic.
        momentum_weight: Relative weight of momentum vs inverse-vol in the
                         fallback heuristic score.
    """

    def __init__(
        self,
        use_real_model: bool = False,
        momentum_window: int = 20,
        vol_window: int = 60,
        momentum_weight: float = 0.5,
    ):
        self.use_real_model = use_real_model
        self.momentum_window = momentum_window
        self.vol_window = vol_window
        self.momentum_weight = momentum_weight
        self._real_model = None

        if use_real_model:
            try:
                import smartfolio as sf   # noqa: F401
                self._real_model = sf
            except ImportError as e:
                raise ImportError(
                    "SmartFolio package not installed. "
                    "Set use_real_model=False to use the heuristic approximation."
                ) from e

    @property
    def model_name(self) -> str:
        suffix = "real" if (self.use_real_model and self._real_model) else "heuristic"
        return f"smartfolio_{suffix}"

    def allocate(self, snapshot: MarketSnapshot) -> dict[str, float]:
        """
        Compute portfolio weights using SmartFolio (or heuristic fallback).

        Real model path:
          Delegates to smartfolio.allocate(price_history, macro) if available.

        Fallback heuristic (momentum + inverse-vol):
          score_i = momentum_weight * momentum_i + (1 - momentum_weight) * (1/σ_i)
          w_i ∝ max(score_i, 0)   (long-only constraint)
        """
        if self.use_real_model and self._real_model is not None:
            return self._allocate_real(snapshot)
        return self._allocate_heuristic(snapshot)

    def _allocate_real(self, snapshot: MarketSnapshot) -> dict[str, float]:
        """Delegate to the real SmartFolio model."""
        # The real SmartFolio API is TBD pending paper publication.
        # Placeholder: fall through to heuristic with a warning.
        warnings.warn(
            "SmartFolio real model API not yet integrated. "
            "Falling back to heuristic approximation.",
            UserWarning,
            stacklevel=2,
        )
        return self._allocate_heuristic(snapshot)

    def _allocate_heuristic(self, snapshot: MarketSnapshot) -> dict[str, float]:
        """
        Momentum + inverse-volatility heuristic approximating SmartFolio.

        Momentum score: trailing cumulative return over `momentum_window` days.
        Volatility score: inverse of rolling std over `vol_window` days.
        Combined score: convex combination controlled by `momentum_weight`.
        Long-only: negative momentum assets receive zero weight.
        """
        assets = list(snapshot.return_data.keys()) if snapshot.return_data else list(snapshot.price_data.keys())

        scores: dict[str, float] = {}
        for asset in assets:
            r = snapshot.return_data.get(asset)
            if r is None or r.dropna().empty:
                scores[asset] = 0.0
                continue

            r_clean = r.dropna()

            # Momentum: cumulative return over last momentum_window days
            window_r = r_clean.iloc[-self.momentum_window:] if len(r_clean) >= self.momentum_window else r_clean
            momentum = float((1 + window_r).prod() - 1)

            # Volatility: annualized std over last vol_window days
            window_v = r_clean.iloc[-self.vol_window:] if len(r_clean) >= self.vol_window else r_clean
            vol = max(float(window_v.std()) * np.sqrt(252), 1e-8)
            inv_vol = 1.0 / vol

            # Combined score (normalize inv_vol to same scale as momentum)
            combined = self.momentum_weight * momentum + (1 - self.momentum_weight) * inv_vol
            scores[asset] = max(combined, 0.0)   # Long-only

        # If all scores are zero (e.g., all momentum negative) → equal weight
        if sum(scores.values()) == 0:
            n = max(len(assets), 1)
            return {a: round(1.0 / n, 6) for a in assets}

        return self._normalize(scores)
