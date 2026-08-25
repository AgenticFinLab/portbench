"""Tests for prompt-exact archived S1-S3 reuse."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from portbench.agent_eval.base import MarketSnapshot, S1Output, S2Output, StageID
from portbench.agent_eval.legacy_stage_reuse import LegacyStageReuseStore
from portbench.agent_eval.prompts import build_s1_prompt, build_s2_prompt, build_s3_prompt
from portbench.agent_eval.stages import (
    _format_correlation,
    _format_macro_context,
    _format_price_context,
)


def _snapshot(current_weights: dict[str, float] | None = None) -> MarketSnapshot:
    """Build one small Point-in-Time snapshot for legacy-reuse tests."""
    index = pd.date_range("2024-01-25", periods=6, freq="B")
    prices = pd.Series([100, 101, 102, 103, 104, 105], index=index)
    returns = prices.pct_change().fillna(0.0)
    return MarketSnapshot(
        decision_date=date(2024, 2, 1),
        price_data={"AAA": prices},
        return_data={"AAA": returns},
        current_weights=current_weights or {"AAA": 1.0},
        portfolio_value=100_000.0,
        market_regime="sideways",
    )


def _prompts(snapshot: MarketSnapshot) -> tuple[str, str, str]:
    """Build the archived prompt chain for one deterministic test episode."""
    s1 = S1Output(
        asset_views={"AAA": 0.3},
        macro_summary="stable",
        detected_regime="sideways",
        confidence=0.7,
    )
    s2 = S2Output(
        signals={"AAA": "buy"},
        strengths={"AAA": 0.3},
        reasoning="positive view",
    )
    s1_prompt = build_s1_prompt(
        snapshot=snapshot,
        assets=["AAA"],
        price_context=_format_price_context(snapshot),
        macro_block=_format_macro_context(snapshot),
        corr_block=_format_correlation(snapshot),
        trailing_days=6,
        use_tools=False,
    )
    s2_prompt = build_s2_prompt(snapshot, s1, ["AAA"], use_tools=False)
    s3_prompt = build_s3_prompt(
        snapshot,
        s2,
        ["AAA"],
        _format_correlation(snapshot),
        use_tools=False,
    )
    return s1_prompt, s2_prompt, s3_prompt


def _write_source(root, snapshot: MarketSnapshot) -> None:
    """Write one prompt-complete legacy episode and its source configuration."""
    config_path = root / "monthly" / "_last_run_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "generation:\n  temperature: 0.0\n  max_tokens: 4096\n",
        encoding="utf-8",
    )
    s1_prompt, s2_prompt, s3_prompt = _prompts(snapshot)
    episode = {
        "decision_date": "2024-02-01",
        "stages": [
            {
                "stage_id": "S1",
                "prompt": s1_prompt,
                "raw_response": "{\"asset_views\": {\"AAA\": 0.3}}",
                "parsed_output": {
                    "asset_views": {"AAA": 0.3},
                    "macro_summary": "stable",
                    "detected_regime": "sideways",
                    "confidence": 0.7,
                },
            },
            {
                "stage_id": "S2",
                "prompt": s2_prompt,
                "raw_response": "{\"signals\": {\"AAA\": \"buy\"}}",
                "parsed_output": {
                    "signals": {"AAA": "buy"},
                    "strengths": {"AAA": 0.3},
                    "reasoning": "positive view",
                },
            },
            {
                "stage_id": "S3",
                "prompt": s3_prompt,
                "raw_response": "{\"weights\": {\"AAA\": 1.0}}",
                "parsed_output": {
                    "weights": {"AAA": 1.0},
                    "expected_return": 0.08,
                    "expected_vol": 0.12,
                    "sharpe_estimate": 0.67,
                },
            },
        ],
    }
    episode_path = root / "monthly" / "demo" / "model" / "run" / "balanced"
    episode_path = episode_path / "normal_2024" / "pipeline_logs" / "log" / "episodes"
    episode_path.mkdir(parents=True)
    (episode_path / "2024-02-01_0001.json").write_text(
        json.dumps(episode), encoding="utf-8"
    )


def test_reuses_all_stages_when_every_prompt_matches(tmp_path):
    source_root = tmp_path / "source"
    snapshot = _snapshot()
    _write_source(source_root, snapshot)
    store = LegacyStageReuseStore(source_root, temperature=0.0, max_tokens=4096)

    resolution = store.resolve(
        snapshot,
        provider="demo",
        model="model",
        profile="balanced",
    )

    assert set(resolution.outputs) == {
        StageID.S1_MARKET_INTERPRETATION,
        StageID.S2_SIGNAL_GENERATION,
        StageID.S3_WEIGHT_OPTIMIZATION,
    }
    assert resolution.decisions["S3"]["prompt_matches"] == 1
    assert len(resolution.provenance) == 3


def test_reuses_s1_s2_but_not_s3_when_current_weights_change(tmp_path):
    source_root = tmp_path / "source"
    _write_source(source_root, _snapshot())
    store = LegacyStageReuseStore(source_root, temperature=0.0, max_tokens=4096)

    resolution = store.resolve(
        _snapshot(current_weights={"AAA": 0.4, "CASH": 0.6}),
        provider="demo",
        model="model",
        profile="balanced",
    )

    assert set(resolution.outputs) == {
        StageID.S1_MARKET_INTERPRETATION,
        StageID.S2_SIGNAL_GENERATION,
    }
    assert resolution.decisions["S3"]["prompt_matches"] == 0


def test_rejects_generation_configuration_mismatch(tmp_path):
    source_root = tmp_path / "source"
    _write_source(source_root, _snapshot())

    with pytest.raises(RuntimeError, match="max_tokens"):
        LegacyStageReuseStore(source_root, temperature=0.0, max_tokens=8192)
