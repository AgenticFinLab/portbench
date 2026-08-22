"""Preregistration expansion blocked when treatment results seen."""

import pytest

from portbench.experiments.preregistration import (
    SA_FACTUAL_ONLY,
    PreregistrationManifest,
    PreregistrationWriter,
)


def test_expansion_blocked_when_treatment_results_seen(tmp_path):
    path = tmp_path / "manifest.jsonl"
    writer = PreregistrationWriter(path)
    record = PreregistrationManifest(
        record_type="expansion",
        window_id="w1",
        original_window={"start": "2020-01-01", "end": "2020-03-01"},
        new_window={"start": "2020-01-01", "end": "2020-06-01"},
        reason="MDE undersized",
        based_on=SA_FACTUAL_ONLY,
        treatment_results_seen=True,
    )
    with pytest.raises(PermissionError, match="treatment_results_seen"):
        writer.append(record)
    assert writer.read_all() == []


def test_expansion_allowed_from_sa_factual(tmp_path):
    path = tmp_path / "manifest.jsonl"
    writer = PreregistrationWriter(path)
    record = PreregistrationManifest(
        record_type="expansion",
        window_id="w1",
        original_window={"start": "2020-01-01", "end": "2020-03-01"},
        new_window={"start": "2020-01-01", "end": "2020-06-01"},
        reason="MDE undersized",
        based_on=SA_FACTUAL_ONLY,
        treatment_results_seen=False,
    )
    writer.append(record)
    rows = writer.read_all()
    assert len(rows) == 1
    assert rows[0]["based_on"] == SA_FACTUAL_ONLY


def test_expansion_requires_based_on_sa_factual(tmp_path):
    path = tmp_path / "manifest.jsonl"
    writer = PreregistrationWriter(path)
    record = PreregistrationManifest(
        record_type="expansion",
        window_id="w1",
        based_on="MA_treatment",
        treatment_results_seen=False,
    )
    with pytest.raises(ValueError, match="based_on"):
        writer.append(record)
