"""
Stress scenario definitions and injector for risk-first evaluation.

Pre-defined historical stress periods are sliced from market data and injected
into the evaluation pipeline as dedicated test episodes. Models must achieve
minimum scores on all stress scenarios before entering the main performance ranking.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from .base import MarketSnapshot


@dataclass
class StressScenario:
    """
    Definition of a stress test scenario.

    Attributes:
        name:           Scenario identifier (used in result filenames).
        start:          Start date of the stress period (primary, real-data window).
        end:            End date of the stress period (primary).
        description:    Human-readable description of the event.
        min_pass_score: Minimum CEPS required to "pass" this scenario.
        fallback_start: Alternative start date used when primary window has no data
                        coverage (e.g., mock data that only goes back to 2015).
        fallback_end:   Alternative end date for the fallback window.
        fallback_description: Description of the fallback event.
    """

    name: str
    start: date
    end: date
    description: str
    min_pass_score: float = 0.50
    fallback_start: Optional[date] = None
    fallback_end: Optional[date] = None
    fallback_description: str = ""


# ---------------------------------------------------------------------------
# Canonical PortBench stress scenarios
# All three windows are fully covered by datasets/processed/ (2015-01-02 to 2025-12-31).
# ---------------------------------------------------------------------------

STRESS_SCENARIOS: list[StressScenario] = [
    StressScenario(
        name="2015_china_shock",
        start=date(2015, 8, 1),
        end=date(2016, 2, 29),
        description=(
            "China currency devaluation shock (Aug 2015) + oil price collapse. "
            "S&P 500 fell ~12% in two weeks; broad cross-asset sell-off with "
            "elevated VIX and credit spread widening."
        ),
        min_pass_score=0.40,
    ),
    StressScenario(
        name="2020_covid_flash_crash",
        start=date(2020, 2, 1),
        end=date(2020, 5, 31),
        description=(
            "COVID-19 pandemic shock. S&P 500 fell 34% in 33 days (fastest bear market on record). "
            "Cross-asset correlations spiked; even gold and Treasuries briefly sold off in "
            "the March 2020 liquidity crisis."
        ),
        min_pass_score=0.45,
    ),
    StressScenario(
        name="2022_crypto_collapse",
        start=date(2022, 5, 1),
        end=date(2022, 12, 31),
        description=(
            "Cryptocurrency market collapse: Bitcoin fell ~75% from peak, Ethereum fell ~80%. "
            "Terra/LUNA collapse (May 2022) and FTX bankruptcy (Nov 2022) were catalysts. "
            "Concurrent with Fed rate hike cycle creating broad risk-off environment."
        ),
        min_pass_score=0.50,
    ),
]


class ScenarioInjector:
    """
    Slices historical data for a stress scenario and generates MarketSnapshot
    objects for each trading day within the scenario window.

    If real data is unavailable, falls back to mock data automatically.

    Args:
        provider:     A DataProvider (or MockDataProvider) for fetching price data.
        assets:       List of asset identifiers to include in snapshots.
        lookback_days: Number of prior trading days to include as context.
    """

    def __init__(self, provider, assets: list[str], lookback_days: int = 60):
        self.provider = provider
        self.assets = assets
        self.lookback_days = lookback_days

    def generate_snapshots(
        self,
        scenario: StressScenario,
        step_days: int = 5,   # Generate one snapshot every 5 trading days
    ) -> list[MarketSnapshot]:
        """
        Generate a list of MarketSnapshot objects for the stress scenario window.

        Args:
            scenario:  The StressScenario to run.
            step_days: Spacing between consecutive decision dates.

        Returns:
            List of MarketSnapshot objects, one per sampled decision date.
        """
        import pandas as pd
        from datetime import timedelta

        snapshots = []
        dates = pd.bdate_range(start=scenario.start, end=scenario.end, freq=f"{step_days}B")

        for d in dates:
            decision_date = d.date()
            try:
                price_data = {}
                return_data = {}
                end = decision_date
                # Approximate start for lookback
                from datetime import timedelta as td
                start = end - td(days=int(self.lookback_days * 1.5))

                for asset in self.assets:
                    prices = self.provider.get_price_series(asset, start, end)
                    prices = prices.iloc[-self.lookback_days:]
                    returns = self.provider.get_return_series(asset, start, end)
                    returns = returns.iloc[-self.lookback_days:]
                    # Skip assets with no data in this window — otherwise an
                    # empty snapshot looks valid and the pipeline scores it as
                    # actual=gt=0 (false pass).
                    if prices.empty or returns.empty:
                        continue
                    price_data[asset] = prices
                    return_data[asset] = returns

                # Require at least 2 assets with usable data to build a snapshot
                if len(price_data) < 2:
                    continue

                macro = self.provider.get_macro(end)
                regime = self.provider.get_regime(end).value

                # Equal-weight initial portfolio
                n = len(self.assets)
                current_weights = {a: round(1.0 / n, 4) for a in self.assets}

                # Pull news text from the first asset that has any
                news_text = ""
                for asset in self.assets:
                    try:
                        txt = self.provider.get_news(asset, decision_date)
                        if txt:
                            news_text = txt
                            break
                    except Exception:
                        continue

                snapshots.append(MarketSnapshot(
                    decision_date=decision_date,
                    price_data=price_data,
                    return_data=return_data,
                    macro_data=macro,
                    current_weights=current_weights,
                    market_regime=regime,
                    news_text=news_text,
                ))
            except Exception:
                continue   # Skip dates where data is unavailable

        return snapshots

    def run_stress_test(
        self,
        scenario: StressScenario,
        pipeline,           # EvalPipeline instance
        step_days: int = 5,
    ) -> dict:
        """
        Run the full stress test for a scenario and return pass/fail result.

        Args:
            scenario:  StressScenario to test.
            pipeline:  EvalPipeline instance (already initialized with stages).
            step_days: Spacing between decision dates in the scenario window.

        Returns:
            Dict with keys: scenario_name, passed, mean_score, per_stage_mean,
            n_episodes, min_pass_score.
        """
        from ..metrics.ceps import CEPS

        snapshots = self.generate_snapshots(scenario, step_days=step_days)
        fallback_used = False
        if not snapshots and scenario.fallback_start and scenario.fallback_end:
            # Primary window has no data coverage — try the fallback window
            fallback = StressScenario(
                name=scenario.name,
                start=scenario.fallback_start,
                end=scenario.fallback_end,
                description=scenario.fallback_description or scenario.description,
                min_pass_score=scenario.min_pass_score,
            )
            snapshots = self.generate_snapshots(fallback, step_days=step_days)
            fallback_used = True

        if not snapshots:
            # No usable data for this scenario window — fail the risk gate
            # rather than silently pass on empty episodes.
            return {
                "scenario_name": scenario.name,
                "passed": False,
                "mean_ceps": 0.0,
                "per_stage_mean": {},
                "n_episodes": 0,
                "min_pass_score": scenario.min_pass_score,
                "description": scenario.description,
                "error": "no snapshots generated (data coverage missing for window)",
            }

        episodes = []
        for snap in snapshots:
            result = pipeline.run_episode(snap)
            episodes.append(result.to_stage_score_list())

        ceps_result = CEPS().compute_batch(episodes)
        mean_score = ceps_result["mean_ceps"]
        passed = mean_score >= scenario.min_pass_score

        result = {
            "scenario_name": scenario.name,
            "passed": passed,
            "mean_ceps": mean_score,
            "per_stage_mean": ceps_result["per_stage_mean"],
            "n_episodes": ceps_result["n_episodes"],
            "min_pass_score": scenario.min_pass_score,
            "description": scenario.description,
        }
        if fallback_used:
            result["fallback_window"] = (
                f"{scenario.fallback_start} → {scenario.fallback_end}"
            )
            result["fallback_description"] = scenario.fallback_description
        return result
