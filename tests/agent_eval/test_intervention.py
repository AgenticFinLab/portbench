"""Intervention operators on toy episode; future data blocked."""

from __future__ import annotations

import pytest

from portbench.agent_eval.contracts import ProvenanceSource, ResultProvenance
from portbench.agent_eval.intervention import (
    apply_perturb,
    apply_repair,
    compute_descriptive_delta,
    store_intervention_result,
)
from portbench.agent_eval.replay_adapter import INTERVENTION_NAMESPACE, ReplayAdapter


LOOKBACK = {"SPY": [0.01, -0.02, 0.005], "TLT": [0.0, 0.001, -0.001]}


def test_repair_blocks_future():
    with pytest.raises(PermissionError):
        apply_repair("S1", {"lookback_returns": LOOKBACK, "future_return_data": {"SPY": [1.0]}})


def test_perturb_and_delta_on_toy():
    factual = apply_repair("S1", lookback_returns=LOOKBACK)

    def bump(output):
        out = dict(output)
        views = dict(out["asset_views"])
        views["SPY"] *= 1.5
        out["asset_views"] = views
        return out

    intervened = apply_perturb("S1", factual, perturb_fn=bump)
    delta = compute_descriptive_delta(
        {"view": factual["asset_views"]["SPY"]},
        {"view": intervened["asset_views"]["SPY"]},
        keys=["view"],
    )
    assert "view" in delta
    assert isinstance(delta["view"], float)


def test_store_uses_intervention_namespace():
    adapter = ReplayAdapter()
    prov = ResultProvenance(source=ProvenanceSource.CURRENT_CACHE.value, cache_key="k1")
    store_intervention_result(adapter, "k1", {"ok": True}, prov)
    assert adapter.lookup("k1", namespace=f"{INTERVENTION_NAMESPACE}__default") is not None
    assert adapter.lookup("k1") is None
