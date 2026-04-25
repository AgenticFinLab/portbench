"""
60/40 (sixty-forty) baseline strategy.

The traditional 60% equity / 40% bond allocation used by institutional
investors for decades.  Included as a widely-recognized industry reference
point.  In PortBench's six-asset-class universe the weights are applied to
asset-class buckets rather than individual securities.

Asset class mapping (configurable):
  EQUITY_CLASSES  → collectively receive 60% weight (split equally within)
  BOND_CLASSES    → collectively receive 40% weight (split equally within)
  OTHER_CLASSES   → receive 0% weight (can be enabled via include_alternatives)

When include_alternatives=True, alternative assets (commodities, real estate,
crypto) are funded by scaling down equities proportionally.
"""

from ..agent_eval.base import MarketSnapshot
from .base import BaselineStrategy


# Default asset-class membership heuristics (substring matching on ticker/name)
_EQUITY_KEYWORDS = {"SPY", "QQQ", "IWM", "VTI", "EEM", "ACWI", "equity", "stock"}
_BOND_KEYWORDS = {"AGG", "TLT", "IEF", "BIL", "LQD", "HYG", "bond", "treasury"}
_COMMODITY_KEYWORDS = {"GLD", "USO", "DJP", "commodity", "gold", "oil"}
_REALESTATE_KEYWORDS = {"VNQ", "IYR", "REIT", "real_estate", "realestate"}
_CRYPTO_KEYWORDS = {"BTC", "ETH", "crypto", "bitcoin"}


def _classify(asset: str) -> str:
    """Map an asset ticker/name to a broad asset class string."""
    a = asset.upper()
    if any(k in a for k in _EQUITY_KEYWORDS):
        return "equity"
    if any(k in a for k in _BOND_KEYWORDS):
        return "bond"
    if any(k in a for k in _COMMODITY_KEYWORDS):
        return "commodity"
    if any(k in a for k in _REALESTATE_KEYWORDS):
        return "real_estate"
    if any(k in a for k in _CRYPTO_KEYWORDS):
        return "crypto"
    return "unknown"


class SixtyFortyBaseline(BaselineStrategy):
    """
    Traditional 60/40 equity-bond allocation.

    Args:
        equity_fraction:       Weight allocated to equity assets (default 0.60).
        bond_fraction:         Weight allocated to bond assets (default 0.40).
        include_alternatives:  If True, assign a small allocation to other
                               asset classes (commodities, real estate, crypto)
                               and reduce equity/bond proportionally.
        alt_fraction:          Total weight for alternatives when enabled (0.10).
    """

    def __init__(
        self,
        equity_fraction: float = 0.60,
        bond_fraction: float = 0.40,
        include_alternatives: bool = False,
        alt_fraction: float = 0.10,
    ):
        assert (
            abs(equity_fraction + bond_fraction - 1.0) < 1e-6 or include_alternatives
        ), "equity_fraction + bond_fraction must equal 1.0 when include_alternatives=False"
        self.equity_fraction = equity_fraction
        self.bond_fraction = bond_fraction
        self.include_alternatives = include_alternatives
        self.alt_fraction = alt_fraction

    @property
    def model_name(self) -> str:
        return f"sixty_forty_eq{self.equity_fraction:.0%}_bd{self.bond_fraction:.0%}"

    def allocate(self, snapshot: MarketSnapshot) -> dict[str, float]:
        """
        Assign weights by asset class.

        Assets whose class cannot be determined receive 0 weight; their
        budget is redistributed to the equity bucket.
        """
        if snapshot.current_weights:
            assets = list(snapshot.current_weights.keys())
        else:
            assets = list(snapshot.price_data.keys())

        # Bucket assets by class
        equities, bonds, alts = [], [], []
        for a in assets:
            cls = _classify(a)
            if cls == "equity":
                equities.append(a)
            elif cls == "bond":
                bonds.append(a)
            else:
                alts.append(a)

        weights: dict[str, float] = {}

        if self.include_alternatives and alts:
            # Scale down eq/bond to make room for alternatives
            scale = 1.0 - self.alt_fraction
            eq_frac = self.equity_fraction * scale
            bd_frac = self.bond_fraction * scale
            alt_frac = self.alt_fraction / max(len(alts), 1)
            for a in alts:
                weights[a] = round(alt_frac, 6)
        else:
            eq_frac = self.equity_fraction
            bd_frac = self.bond_fraction

        # Distribute equity budget equally within equity bucket
        if equities:
            per_eq = round(eq_frac / len(equities), 6)
            for a in equities:
                weights[a] = per_eq
        else:
            # No equity assets found — add their budget to bonds
            bd_frac += eq_frac

        # Distribute bond budget equally within bond bucket
        if bonds:
            per_bd = round(bd_frac / len(bonds), 6)
            for a in bonds:
                weights[a] = per_bd
        elif not equities:
            # Neither equity nor bond found — fall back to equal weight
            n = max(len(assets), 1)
            return {a: round(1.0 / n, 6) for a in assets}

        return self._normalize(weights)
