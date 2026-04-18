"""
Equal-weight (1/N) baseline strategy.

The simplest possible allocation: divide total portfolio weight equally
across all assets.  Despite its simplicity, equal-weighting is a
surprisingly hard benchmark to beat in practice (DeMiguel et al., 2009).

In PortBench this serves as the lowest bar — any model claiming to add
value must outperform 1/N allocation.
"""

from ..agent_eval.base import MarketSnapshot
from .base import BaselineStrategy


class EqualWeightBaseline(BaselineStrategy):
    """
    Equal-weight (1/N) portfolio strategy.

    No parameters — simply divides 1.0 equally across all assets present
    in snapshot.current_weights or snapshot.price_data.

    Args:
        asset_universe: Optional explicit list of assets.  If None, the
                        assets present in each snapshot are used.
    """

    def __init__(self, asset_universe: list[str] = None):
        self._universe = asset_universe

    @property
    def model_name(self) -> str:
        return "equal_weight_1_over_N"

    def allocate(self, snapshot: MarketSnapshot) -> dict[str, float]:
        """
        Return equal weights across the asset universe.

        Universe is determined by (in order of priority):
          1. Explicit asset_universe passed to __init__
          2. Assets in snapshot.current_weights
          3. Assets in snapshot.price_data
        """
        if self._universe:
            assets = self._universe
        elif snapshot.current_weights:
            assets = list(snapshot.current_weights.keys())
        else:
            assets = list(snapshot.price_data.keys())

        n = max(len(assets), 1)
        weights = {a: round(1.0 / n, 6) for a in assets}
        return weights
