"""Unit tests for T3/T4 redesign switch (no dataset regeneration required)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from portbench.qa_builder.base import ContextWindow, MarketRegime, QAConfig, Split
from portbench.qa_builder.t3_position_sizing import T3PositionSizing
from portbench.qa_builder.t4_pairwise_allocation import T4PairwiseAllocation
from portbench.qa_eval.scorer import score_response


class _TinyProvider:
    def list_assets(self, asset_class=None):
        return ["SPY", "TLT"]

    def get_return_series(self, asset, start, end):
        idx = pd.bdate_range(start, end)
        rng = np.random.default_rng(abs(hash(asset)) % (2**31))
        return pd.Series(rng.normal(0.0, 0.01, size=len(idx)), index=idx)


def _context(assets=("SPY", "TLT")) -> ContextWindow:
    d = date(2024, 6, 3)
    idx = pd.bdate_range("2024-01-02", "2024-05-31")
    rng = np.random.default_rng(7)
    returns = {
        a: pd.Series(
            rng.normal(0.0005 if a == "SPY" else 0.0001, 0.012, len(idx)), index=idx
        )
        for a in assets
    }
    return ContextWindow(
        decision_date=d,
        assets=list(assets),
        price_history={a: (1 + returns[a]).cumprod() * 100 for a in assets},
        returns_history=returns,
        market_regime=MarketRegime.BULL,
        news_text="",
    )


@pytest.fixture
def qa_config():
    return QAConfig(
        lookback_days=60,
        horizon_days=21,
        samples_per_template=10,
    )


def test_legacy_t3_keeps_var_in_context(qa_config):
    pair = T3PositionSizing(_TinyProvider(), qa_config, redesign=False).build_one(
        _context(("SPY",)), 0
    )
    assert "VaR(99%)" in pair.context_summary
    assert pair.metadata.get("t3t4_redesign") is False


def test_redesign_t3_strips_var_and_adds_keypoints(qa_config):
    pair = T3PositionSizing(_TinyProvider(), qa_config, redesign=True).build_one(
        _context(("SPY",)), 1
    )
    assert "VaR(99%)" not in pair.context_summary
    assert "VaR(99%)" not in pair.question
    assert pair.metadata.get("t3t4_redesign") is True
    assert pair.metadata.get("explanation_keypoints")
    assert "explanation" in pair.question.lower()


def test_legacy_t4_exposes_covariance(qa_config):
    pair = T4PairwiseAllocation(_TinyProvider(), qa_config, redesign=False).build_one(
        _context(("SPY", "TLT")), 0
    )
    assert "Covariance(" in pair.question
    assert pair.metadata.get("t3t4_redesign") is False


def test_redesign_t4_hides_covariance(qa_config):
    pair = T4PairwiseAllocation(_TinyProvider(), qa_config, redesign=True).build_one(
        _context(("SPY", "TLT")), 0
    )
    assert "Covariance(" not in pair.question
    assert "Minimum required portfolio return" in pair.question
    assert pair.metadata.get("constraint_binding") is not None
    assert pair.metadata.get("explanation_keypoints")


def test_composite_score_penalizes_bad_explanation():
    gt = "0.2500"
    good = '{"answer": 0.25, "explanation": "Use VaR downside and drawdown limit; position size capped."}'
    bad = '{"answer": 0.25, "explanation": "Because the chart looks nice."}'
    keypoints = ["var", "drawdown", "capped"]
    s_good = score_response(
        "T3",
        gt,
        good,
        answer_numeric=0.25,
        redesign=True,
        explanation_keypoints=keypoints,
    )
    s_bad = score_response(
        "T3",
        gt,
        bad,
        answer_numeric=0.25,
        redesign=True,
        explanation_keypoints=keypoints,
    )
    assert s_good > s_bad
    assert s_good > 0.7  # strong numeric + partial/full explanation
