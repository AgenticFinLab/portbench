"""Legacy cache scanner tests."""

from __future__ import annotations

from portbench.experiments.legacy_cache import scan_legacy_cache, summarize
from tests.experiments.artifact_factory import write_legacy_episode


def test_scan_fixture_grades(tmp_path):
    # Generate an auditable Grade-A episode without repository-owned fixtures.
    fixture_root = tmp_path / "mini_experiments"
    write_legacy_episode(fixture_root)
    records = scan_legacy_cache(fixture_root)
    assert len(records) >= 1
    grades = {r.episode_id: r.grade for r in records}
    assert grades.get("ep_toy") == "A"
    summary = summarize(records)
    assert summary["episode_count"] == len(records)
    assert sum(summary["grade_histogram"].values()) == len(records)
