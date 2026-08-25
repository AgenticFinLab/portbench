"""Tests for durable, parser-aware LLM call artifacts."""

from __future__ import annotations

import json

import pytest

from portbench.agent_eval.call_artifacts import (
    CallArtifactStore,
    CallRequest,
    TerminalCallFailure,
)


def _request(**overrides):
    """Build one minimal SA-only request for artifact tests."""
    payload = {
        "provider": "test",
        "model": "model-a",
        "model_revision": "revision-a",
        "stage_id": "S4",
        "system_prompt": "system",
        "user_prompt": "user",
        "response_schema": {"type": "object"},
        "generation_config": {"temperature": 0.0, "max_tokens": 128},
        "visible_input": {"weights": {"A": 1.0}},
    }
    payload.update(overrides)
    return CallRequest(**payload)


def test_valid_call_reuses_raw_parse_and_score_without_provider(tmp_path):
    store = CallArtifactStore(tmp_path)
    request = _request()
    calls = {"count": 0}

    def call():
        calls["count"] += 1
        return '{"answer": 0.5}'

    parse = lambda raw: json.loads(raw)
    parsed, artifact, hit = store.complete_or_call(
        request,
        parser_version="schema-v1",
        parse=parse,
        call_fn=call,
        backoff_seconds=(0.0, 0.0),
    )
    assert parsed == {"answer": 0.5}
    assert not hit
    assert artifact.raw_response == '{"answer": 0.5}'
    records = store.attempts(request)
    assert records[0]["event"] == "request_recorded"
    assert records[0]["request_hash"] == request.call_key
    assert records[0]["request"]["system_prompt"] == request.system_prompt
    assert records[0]["request"]["user_prompt"] == request.user_prompt

    parsed_again, _, hit_again = store.complete_or_call(
        request,
        parser_version="schema-v1",
        parse=parse,
        call_fn=call,
        backoff_seconds=(0.0, 0.0),
    )
    assert hit_again
    assert parsed_again == parsed
    assert calls["count"] == 1

    score = store.score(
        request,
        "schema-v1",
        "metric-v1",
        {"answer": 0.5},
        lambda output: {"score": float(output["answer"] == 0.5)},
    )
    assert score.score_payload == {"score": 1.0}
    assert store.score(
        request,
        "schema-v1",
        "metric-v1",
        {"answer": 0.5},
        lambda output: {"score": 0.0},
    ).score_payload == {"score": 1.0}


def test_parser_upgrade_reuses_validated_raw_response(tmp_path):
    store = CallArtifactStore(tmp_path)
    request = _request()
    calls = {"count": 0}

    def call():
        calls["count"] += 1
        return '{"answer": "0.5"}'

    store.complete_or_call(
        request,
        parser_version="schema-v1",
        parse=lambda raw: json.loads(raw),
        call_fn=call,
        backoff_seconds=(0.0, 0.0),
    )
    parsed, _, hit = store.complete_or_call(
        request,
        parser_version="schema-v2",
        parse=lambda raw: {"answer": float(json.loads(raw)["answer"])},
        call_fn=call,
        backoff_seconds=(0.0, 0.0),
    )
    assert hit
    assert parsed == {"answer": 0.5}
    assert calls["count"] == 1


def test_invalid_attempts_stay_in_ledger_and_require_explicit_retry(tmp_path):
    store = CallArtifactStore(tmp_path)
    request = _request()
    responses = iter(["not-json", "still-not-json", "bad", '{"answer": 1}'])

    with pytest.raises(TerminalCallFailure):
        store.complete_or_call(
            request,
            parser_version="schema-v1",
            parse=lambda raw: json.loads(raw),
            call_fn=lambda: next(responses),
            backoff_seconds=(0.0, 0.0),
        )
    assert store.load_call(request) is None
    assert len([item for item in store.attempts(request) if item["event"] == "received"]) == 3

    parsed, _, hit = store.complete_or_call(
        request,
        parser_version="schema-v1",
        parse=lambda raw: json.loads(raw),
        call_fn=lambda: next(responses),
        backoff_seconds=(0.0, 0.0),
        retry_failed=True,
    )
    assert not hit
    assert parsed == {"answer": 1}


def test_call_key_ignores_provenance_but_tracks_behavior_fields():
    request = _request()
    assert request.call_key == _request().call_key
    assert request.call_key != _request(user_prompt="changed").call_key
    assert request.call_key != _request(generation_config={"temperature": 0.1}).call_key
    assert request.call_key != _request(namespace="intervention__S3_repair").call_key

    with pytest.raises(ValueError, match="SA-only"):
        _request(architecture_id="MA")


def test_provider_errors_count_toward_the_persisted_attempt_limit(tmp_path):
    store = CallArtifactStore(tmp_path)
    request = _request()

    with pytest.raises(TerminalCallFailure):
        store.complete_or_call(
            request,
            parser_version="schema-v1",
            parse=lambda raw: json.loads(raw),
            call_fn=lambda: (_ for _ in ()).throw(ConnectionError("offline")),
            backoff_seconds=(0.0, 0.0),
        )
    attempts = store.attempts(request)
    assert len([item for item in attempts if item["event"] == "provider_error"]) == 3
