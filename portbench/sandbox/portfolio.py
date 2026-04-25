"""
PortfolioState: stateful portfolio tracker for the Sandbox backtest engine.

Maintains NAV, weights, and trade history across rebalance steps.
Transaction cost model (slippage + commission) mirrors S4ExecutionSimulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class PortfolioState:
    """
    Tracks portfolio state across time steps in the backtest loop.

    Attributes:
        nav:          Current net asset value (dollar amount).
        weights:      Current portfolio weights after drift (sum ≈ 1.0).
        nav_history:  List of (date, nav) tuples for NAV curve construction.
        trade_history: List of rebalance records with cost detail.
    """

    nav: float
    weights: dict[str, float]
    nav_history: list[tuple[date, float]] = field(default_factory=list)
    trade_history: list[dict] = field(default_factory=list)

    SLIPPAGE_RATE: float = 0.0010   # 10 bps linear market impact
    COMMISSION_RATE: float = 0.0005  # 5 bps commission per trade value

    def rebalance(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
        d: date,
    ) -> dict:
        """
        Execute a rebalance to target_weights, apply transaction costs, and
        update internal state.

        Args:
            target_weights: Desired portfolio weights (must sum to ~1.0).
            prices:         Current last prices per asset (used for slippage direction).
            d:              Rebalance date (recorded in trade_history).

        Returns:
            Trade record dict with cost breakdown.
        """
        current = self.weights
        total_cost = 0.0
        total_turnover = 0.0
        orders = []

        all_assets = set(current) | set(target_weights)
        for asset in all_assets:
            curr_w = current.get(asset, 0.0)
            targ_w = target_weights.get(asset, 0.0)
            delta = targ_w - curr_w
            if abs(delta) < 1e-6:
                continue

            direction = "buy" if delta > 0 else "sell"
            trade_value = abs(delta) * self.nav
            slippage = self.SLIPPAGE_RATE * (1 if direction == "buy" else -1)
            commission = trade_value * self.COMMISSION_RATE
            cost = commission + trade_value * abs(slippage)
            total_cost += cost
            total_turnover += abs(delta)
            orders.append({
                "asset": asset,
                "direction": direction,
                "delta_weight": round(delta, 6),
                "trade_value": round(trade_value, 2),
                "slippage_bps": round(slippage * 10000, 2),
                "commission": round(commission, 4),
                "cost": round(cost, 4),
            })

        cost_drag = total_cost / self.nav
        self.nav = self.nav * (1 - cost_drag)

        # Normalize target weights to sum to 1.0
        total_w = sum(target_weights.values())
        if total_w > 0:
            self.weights = {a: w / total_w for a, w in target_weights.items()}
        else:
            self.weights = dict(target_weights)

        record = {
            "date": str(d),
            "total_cost": round(total_cost, 4),
            "total_turnover": round(total_turnover, 4),
            "nav_after": round(self.nav, 2),
            "orders": orders,
        }
        self.trade_history.append(record)
        self.nav_history.append((d, self.nav))
        return record

    def mark_to_market(self, returns: dict[str, float], d: date) -> None:
        """
        Update NAV and weights using per-asset daily returns (no rebalance).

        Args:
            returns: Dict of asset → daily simple return for date d.
            d:       Date to record in nav_history.
        """
        # Compute weighted portfolio return
        port_return = sum(
            self.weights.get(asset, 0.0) * r
            for asset, r in returns.items()
        )
        self.nav = self.nav * (1 + port_return)

        # Update weights to reflect price drift (weight drift)
        new_weights = {}
        total = 0.0
        for asset, w in self.weights.items():
            asset_return = returns.get(asset, 0.0)
            new_w = w * (1 + asset_return)
            new_weights[asset] = new_w
            total += new_w
        if total > 0:
            self.weights = {a: w / total for a, w in new_weights.items()}

        self.nav_history.append((d, self.nav))
