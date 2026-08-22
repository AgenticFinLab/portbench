"""Source signature tests for experiment cache provenance."""

from portbench.experiments.source_version import source_tree_hash


def test_source_hash_changes_with_content_and_path(tmp_path):
    source_dir = tmp_path / "portbench"
    source_dir.mkdir()
    module = source_dir / "model.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")

    original = source_tree_hash(tmp_path)
    module.write_text("VALUE = 2\n", encoding="utf-8")
    changed_content = source_tree_hash(tmp_path)
    module.rename(source_dir / "agent.py")
    changed_path = source_tree_hash(tmp_path)

    assert changed_content != original
    assert changed_path != changed_content


def test_source_hash_ignores_local_reports(tmp_path):
    source_dir = tmp_path / "portbench"
    source_dir.mkdir()
    (source_dir / "model.py").write_text("VALUE = 1\n", encoding="utf-8")

    original = source_tree_hash(tmp_path)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "validation.md").write_text("local evidence\n", encoding="utf-8")

    assert source_tree_hash(tmp_path) == original
