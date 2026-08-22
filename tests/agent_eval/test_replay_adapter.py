"""ReplayAdapter: cache hit skips call_fn; namespace isolation; provenance."""

import pytest

from portbench.agent_eval.contracts import ProvenanceSource, ResultProvenance
from portbench.agent_eval.result_gates import validate_provenance
from portbench.agent_eval.replay_adapter import (
    FACTUAL_NAMESPACE,
    INTERVENTION_NAMESPACE,
    ReplayAdapter,
)


def test_same_key_zero_second_call_fn_invocations():
    adapter = ReplayAdapter()
    calls = {"n": 0}

    def call_fn():
        calls["n"] += 1
        return {"weights": {"SPY": 1.0}}

    out1, prov1 = adapter.complete_or_call("key-a", call_fn, namespace=FACTUAL_NAMESPACE)
    out2, prov2 = adapter.complete_or_call("key-a", call_fn, namespace=FACTUAL_NAMESPACE)
    assert calls["n"] == 1
    assert out1 == out2
    assert prov1.source == ProvenanceSource.NEW_API_CALL.value
    assert prov2.cache_key == "key-a"


def test_namespace_isolation():
    adapter = ReplayAdapter()
    calls = {"n": 0}

    def call_fn():
        calls["n"] += 1
        return {"x": calls["n"]}

    adapter.complete_or_call("shared-key", call_fn, namespace=FACTUAL_NAMESPACE)
    adapter.complete_or_call("shared-key", call_fn, namespace=INTERVENTION_NAMESPACE)
    assert calls["n"] == 2
    f_out, _ = adapter.complete_from_cache("shared-key", namespace=FACTUAL_NAMESPACE)
    i_out, _ = adapter.complete_from_cache("shared-key", namespace=INTERVENTION_NAMESPACE)
    assert f_out != i_out


def test_persistent_cache_survives_adapter_restart(tmp_path):
    first = ReplayAdapter(tmp_path)
    provenance = ResultProvenance(
        source=ProvenanceSource.NEW_API_CALL.value,
        cache_key="persisted",
        request_count=1,
    )
    first.put("persisted", {"weights": {"SPY": 1.0}}, provenance)

    second = ReplayAdapter(tmp_path)
    calls = {"n": 0}

    def must_not_call():
        calls["n"] += 1
        return {"wrong": True}

    output, hit_provenance = second.complete_or_call("persisted", must_not_call)
    assert calls["n"] == 0
    assert output == {"weights": {"SPY": 1.0}}
    assert hit_provenance.source == ProvenanceSource.CURRENT_CACHE.value


def test_missing_provenance_rejected():
    with pytest.raises(ValueError):
        validate_provenance(None)
    with pytest.raises(ValueError):
        ResultProvenance(source="not-a-real-source")
    with pytest.raises(ValueError):
        validate_provenance({"source": "bogus", "cache_key": "k"})

    # Putting without provenance object is not allowed via API; simulate bad cache entry.
    adapter = ReplayAdapter()
    bad = type("Bad", (), {"output": 1, "provenance": None, "schema_version": "pipeline-v1"})()
    adapter._stores[FACTUAL_NAMESPACE]["bad"] = bad  # type: ignore[assignment]
    with pytest.raises(ValueError, match="provenance"):
        adapter.complete_from_cache("bad", namespace=FACTUAL_NAMESPACE)
