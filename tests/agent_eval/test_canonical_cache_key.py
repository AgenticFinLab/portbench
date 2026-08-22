"""Canonical cache key: lossless normalization and field sensitivity."""

import pandas as pd
import pytest

from portbench.agent_eval.canonical import (
    build_stage_cache_key,
    canonical_json,
    sha256_hex,
)


def _base_fields(**overrides):
    fields = dict(
        model="m1",
        model_revision="rev1",
        provider="p1",
        stage_id="S3",
        stage_schema_version="pipeline-v1",
        system_prompt_hash="sys",
        user_prompt_hash="usr",
        structured_stage_input_hash=sha256_hex(canonical_json({"b": 2, "a": 1})),
        generation_config={"temperature": 0.0, "max_tokens": 128},
        toolset_hash="tools",
        tool_result_hash="tool-results",
        memory_state_hash="mem",
        market_snapshot_hash="mkt",
        profile="balanced",
        decision_date="2024-01-15",
        data_version="dv1",
        code_commit="commit1",
        architecture_id="SA",
        memory_enabled=False,
        tools_enabled=False,
        intervention_id="factual",
        agent_id="optimizer",
        agent_protocol_version="collab-protocol-v1",
        collaboration_round="proposal",
        inbox_hash="inbox",
        message_bundle_hash="messages",
    )
    fields.update(overrides)
    return fields


def test_map_key_order_does_not_change_structured_hash():
    h1 = sha256_hex(canonical_json({"a": 1, "b": {"y": 2, "x": 1}}))
    h2 = sha256_hex(canonical_json({"b": {"x": 1, "y": 2}, "a": 1}))
    assert h1 == h2


def test_dataframe_values_and_labels_affect_hash():
    first = pd.DataFrame({"SPY": [0.1, 0.2]}, index=["d1", "d2"])
    changed_value = pd.DataFrame({"SPY": [0.1, 0.3]}, index=["d1", "d2"])
    changed_label = pd.DataFrame({"SPY": [0.1, 0.2]}, index=["d1", "d3"])
    first_hash = sha256_hex(canonical_json(first))
    assert sha256_hex(canonical_json(changed_value)) != first_hash
    assert sha256_hex(canonical_json(changed_label)) != first_hash


def test_cache_key_order_independence_via_canonical_inputs():
    # structured hash identical => overall key identical when other fields match
    structured_a = sha256_hex(canonical_json({"z": 9, "a": [1, 2]}))
    structured_b = sha256_hex(canonical_json({"a": [1, 2], "z": 9}))
    assert structured_a == structured_b
    k1 = build_stage_cache_key(**_base_fields(structured_stage_input_hash=structured_a))
    k2 = build_stage_cache_key(**_base_fields(structured_stage_input_hash=structured_b))
    assert k1 == k2


def test_generation_config_key_order_independent():
    k1 = build_stage_cache_key(
        **_base_fields(generation_config={"temperature": 0.0, "max_tokens": 128})
    )
    k2 = build_stage_cache_key(
        **_base_fields(generation_config={"max_tokens": 128, "temperature": 0.0})
    )
    assert k1 == k2


def test_any_field_change_changes_key():
    base = build_stage_cache_key(**_base_fields())
    mutants = [
        _base_fields(model="m2"),
        _base_fields(model_revision="rev2"),
        _base_fields(provider="p2"),
        _base_fields(stage_id="S4"),
        _base_fields(stage_schema_version="pipeline-v2-agentic"),
        _base_fields(system_prompt_hash="sys2"),
        _base_fields(user_prompt_hash="usr2"),
        _base_fields(structured_stage_input_hash="different"),
        _base_fields(generation_config={"temperature": 0.1, "max_tokens": 128}),
        _base_fields(toolset_hash="tools2"),
        _base_fields(tool_result_hash="tool-results2"),
        _base_fields(memory_state_hash="mem2"),
        _base_fields(market_snapshot_hash="mkt2"),
        _base_fields(profile="conservative"),
        _base_fields(decision_date="2024-01-16"),
        _base_fields(data_version="dv2"),
        _base_fields(code_commit="commit2"),
        _base_fields(architecture_id="MA"),
        _base_fields(memory_enabled=True),
        _base_fields(tools_enabled=True),
        _base_fields(intervention_id="repair-S2"),
        _base_fields(agent_id="risk"),
        _base_fields(agent_protocol_version="collab-protocol-v2"),
        _base_fields(collaboration_round="revision"),
        _base_fields(inbox_hash="inbox2"),
        _base_fields(message_bundle_hash="messages2"),
    ]
    for fields in mutants:
        assert build_stage_cache_key(**fields) != base


def test_missing_or_unknown_cache_key_field_is_rejected():
    missing = _base_fields()
    missing.pop("tool_result_hash")
    with pytest.raises(ValueError, match="missing cache-key fields"):
        build_stage_cache_key(**missing)
    with pytest.raises(ValueError, match="unknown cache-key fields"):
        build_stage_cache_key(**_base_fields(extra="x"))
