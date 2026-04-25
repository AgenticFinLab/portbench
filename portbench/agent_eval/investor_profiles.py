"""
Personalized investor profile evaluation for PortBench.

Defines three investor risk profiles (conservative / balanced / aggressive)
and provides two classes:

  ProfileAlignmentScorer  — post-computes a [0, 1] score from an EpisodeResult
                            measuring how well the S3 weights satisfy the profile's
                            asset-class constraints and S5 VaR limit.

  ProfiledPipeline        — thin wrapper around EvalPipeline that prepends the
                            investor profile description to snapshot.news_text
                            before calling run_episode(), then returns both the
                            EpisodeResult and the alignment score.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import (
        EpisodeResult,
        EvalPipeline,
        MarketSnapshot,
    )


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvestorProfile:
    """
    Risk profile parameters for a single investor archetype.

    Attributes:
        name:                  "conservative" | "balanced" | "aggressive"
        max_equity_weight:     Maximum total weight in equities + cryptocurrency.
        min_bond_cash_weight:  Minimum total weight in bonds + cash.
        max_drawdown_tolerance: Maximum acceptable portfolio drawdown (positive fraction).
        var_limit:             Maximum 1-day VaR at 95% (negative fraction, e.g. -0.01).
        description:           Plain-text description injected into the LLM prompt.
    """

    name: str
    max_equity_weight: float
    min_bond_cash_weight: float
    max_drawdown_tolerance: float
    var_limit: float
    description: str


PROFILES: dict[str, InvestorProfile] = {
    "conservative": InvestorProfile(
        name="conservative",
        max_equity_weight=0.40,
        min_bond_cash_weight=0.40,
        max_drawdown_tolerance=0.10,
        var_limit=-0.01,
        description=(
            "This investor is CONSERVATIVE: allocate at most 40% to equities and "
            "cryptocurrency combined; allocate at least 40% to bonds and cash; "
            "the portfolio must not exceed a 10% maximum drawdown."
        ),
    ),
    "balanced": InvestorProfile(
        name="balanced",
        max_equity_weight=0.65,
        min_bond_cash_weight=0.20,
        max_drawdown_tolerance=0.20,
        var_limit=-0.02,
        description=(
            "This investor is BALANCED: allocate at most 65% to equities and "
            "cryptocurrency combined; allocate at least 20% to bonds and cash; "
            "the portfolio may accept up to a 20% maximum drawdown."
        ),
    ),
    "aggressive": InvestorProfile(
        name="aggressive",
        max_equity_weight=0.90,
        min_bond_cash_weight=0.05,
        max_drawdown_tolerance=0.35,
        var_limit=-0.04,
        description=(
            "This investor is AGGRESSIVE: up to 90% may be allocated to equities and "
            "cryptocurrency; only 5% minimum in bonds and cash; "
            "drawdown tolerance is up to 35%."
        ),
    ),
}

# Asset classes that count as high-risk (equity-side)
_EQUITY_CLASSES = {"equities", "cryptocurrency", "real_estate"}
# Asset classes that count as defensive (bond-side)
_BOND_CLASSES = {"bonds", "cash"}


# ---------------------------------------------------------------------------
# Profile alignment scorer
# ---------------------------------------------------------------------------


class ProfileAlignmentScorer:
    """
    Scores how well an EpisodeResult's S3 weights satisfy an InvestorProfile.

    The score has three components, each in [0, 1], averaged equally:
      1. Equity constraint: whether the equity+crypto weight is within the profile limit.
      2. Bond constraint: whether the bond+cash weight meets the profile minimum.
      3. VaR constraint: whether S5 portfolio_var is within the profile's VaR limit.

    Args:
        asset_class_map: Dict mapping ticker → asset class string.
                         Build with: {a: cls for cls in ALL_CLASSES
                                      for a in provider.list_assets(cls)}
    """

    def __init__(self, asset_class_map: dict[str, str]) -> None:
        self._class_map = asset_class_map

    def score(self, episode: "EpisodeResult", profile: InvestorProfile) -> float:
        """Return a [0, 1] alignment score for the episode vs the profile."""
        from .base import S3Output, S5Output, StageID

        # --- S3 weights ---
        s3_out = episode.stage_outputs.get(StageID.S3_WEIGHT_OPTIMIZATION)
        weights: dict[str, float] = {}
        if s3_out is not None and hasattr(s3_out, "weights"):
            weights = s3_out.weights or {}

        equity_w = sum(
            w
            for a, w in weights.items()
            if self._class_map.get(a, "") in _EQUITY_CLASSES
        )
        bond_w = sum(
            w for a, w in weights.items() if self._class_map.get(a, "") in _BOND_CLASSES
        )

        # Equity constraint: penalise proportionally if over limit
        if profile.max_equity_weight > 0:
            excess = max(equity_w - profile.max_equity_weight, 0.0)
            equity_score = max(1.0 - excess / profile.max_equity_weight, 0.0)
        else:
            equity_score = 1.0 if equity_w == 0 else 0.0

        # Bond constraint: credit proportionally if above minimum
        if profile.min_bond_cash_weight > 0:
            bond_score = min(bond_w / profile.min_bond_cash_weight, 1.0)
        else:
            bond_score = 1.0

        # --- S5 VaR ---
        s5_out = episode.stage_outputs.get(StageID.S5_RISK_MONITORING)
        var_score = 1.0
        if s5_out is not None and hasattr(s5_out, "portfolio_var"):
            pvar = s5_out.portfolio_var or 0.0
            # profile.var_limit is negative; more negative = tighter
            if profile.var_limit < 0:
                excess_var = min(pvar - profile.var_limit, 0.0)  # negative if breached
                var_score = max(1.0 + excess_var / abs(profile.var_limit), 0.0)

        return round((equity_score + bond_score + var_score) / 3.0, 6)


# ---------------------------------------------------------------------------
# Profiled pipeline wrapper
# ---------------------------------------------------------------------------


class ProfiledPipeline:
    """
    Thin wrapper around EvalPipeline that injects investor profile context.

    For each call to run_episode(), the profile's description is prepended to
    snapshot.news_text so that S1/S2/S3 LLM prompts receive the constraint.
    The alignment score is then post-computed from the returned EpisodeResult.

    Args:
        pipeline:        An existing EvalPipeline instance.
        asset_class_map: Dict mapping ticker → asset class string.
    """

    def __init__(
        self, pipeline: "EvalPipeline", asset_class_map: dict[str, str]
    ) -> None:
        self._pipeline = pipeline
        self._scorer = ProfileAlignmentScorer(asset_class_map)

    def run_episode(
        self,
        snapshot: "MarketSnapshot",
        profile: InvestorProfile,
    ) -> tuple["EpisodeResult", float]:
        """
        Run the full pipeline with investor profile context injected.

        Returns:
            (EpisodeResult, alignment_score)  where alignment_score ∈ [0, 1].
        """
        profile_prefix = f"[INVESTOR PROFILE] {profile.description}\n\n"
        patched = dataclasses.replace(
            snapshot,
            news_text=profile_prefix + snapshot.news_text,
        )
        episode = self._pipeline.run_episode(patched)
        alignment = self._scorer.score(episode, profile)
        return episode, alignment
