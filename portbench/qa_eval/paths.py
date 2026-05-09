"""QA evaluation path helpers — mirrors portbench/experiments/paths.py."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def qa_root(output_root: str, batch_id: str) -> Path:
    return Path(output_root) / batch_id / "qa_eval"


def qa_model_dir(output_root: str, batch_id: str, model_label: str) -> Path:
    return qa_root(output_root, batch_id) / model_label


def qa_template_dir(
    output_root: str, batch_id: str, model_label: str, template_id: str
) -> Path:
    return qa_model_dir(output_root, batch_id, model_label) / template_id


def qa_checkpoint_file(
    output_root: str, batch_id: str, model_label: str
) -> Path:
    return qa_model_dir(output_root, batch_id, model_label) / "qa_checkpoint.json"


def qa_figures_dir(output_root: str, batch_id: str, model_label: str) -> Path:
    return qa_model_dir(output_root, batch_id, model_label) / "figures"


def load_checkpoint(path: Path) -> set[str]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("completed", []))
    return set()


def write_checkpoint(
    path: Path, completed_keys: set[str], batch_id: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "completed": sorted(completed_keys),
                "updated_at": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def append_result(out_dir: Path, record: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
