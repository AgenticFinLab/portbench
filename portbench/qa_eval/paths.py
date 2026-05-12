"""QA evaluation path helpers.

Layout (independent of rebalance frequency and run timestamp):

  EXPERIMENTS/
    qa_eval/
      {provider}/          e.g. tencent | deepseek | ark
        {model}/           e.g. hunyuan-turbos | deepseek-v4-pro
          qa_checkpoint.json
          qa_model_summary.json
          figures/
          {template}/      e.g. T1 | T2 | ... | T7
            results.jsonl
            summary.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def qa_root(output_root: str) -> Path:
    return Path(output_root) / "qa_eval"


def qa_model_dir(output_root: str, provider: str, model_name: str) -> Path:
    return qa_root(output_root) / provider / model_name


def qa_template_dir(
    output_root: str, provider: str, model_name: str, template_id: str
) -> Path:
    return qa_model_dir(output_root, provider, model_name) / template_id


def qa_checkpoint_file(output_root: str, provider: str, model_name: str) -> Path:
    return qa_model_dir(output_root, provider, model_name) / "qa_checkpoint.json"


def qa_figures_dir(output_root: str, provider: str, model_name: str) -> Path:
    return qa_model_dir(output_root, provider, model_name) / "figures"


def load_checkpoint(path: Path) -> set[str]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("completed", []))
    return set()


def write_checkpoint(path: Path, completed_keys: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
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
