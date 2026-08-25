"""Offline statistics for the locked constraint-v2 QA evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


CONSTRAINT_TEMPLATES = ("T3", "T4")


def load_constraint_v2_records(root: str | Path) -> list[dict[str, Any]]:
    """Load successful constraint-v2 results from one frozen QA output root."""
    qa_root = Path(root) / "qa_eval"
    records: list[dict[str, Any]] = []
    for path in sorted(qa_root.glob("*/*/T[34]/results.jsonl")):
        provider, model, template = path.relative_to(qa_root).parts[:3]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("template_version") != "constraint-v2" or record.get("error"):
                continue
            records.append(
                {
                    "model": f"{provider}/{model}",
                    "template": template,
                    "qa_id": str(record.get("qa_id", "")),
                    "score": float(record.get("score", 0.0)),
                }
            )
    return records


def _bootstrap(values: list[float], *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    """Compute an item-bootstrap confidence interval for one score collection."""
    if not values:
        return {"mean": None, "ci95": [None, None], "n": 0}
    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sample), size=(n_bootstrap, len(sample)))
    draws = sample[indices].mean(axis=1)
    return {
        "mean": float(sample.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "n": int(len(sample)),
    }


def summarize_constraint_v2(
    records: Iterable[Mapping[str, Any]],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize model-template and pooled scores with item bootstrap intervals."""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        template = str(record.get("template", ""))
        if template in CONSTRAINT_TEMPLATES:
            grouped[str(record["model"])][template].append(float(record["score"]))
    per_model: dict[str, dict[str, Any]] = {}
    pooled_by_template: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    for model_index, model in enumerate(sorted(grouped)):
        per_template: dict[str, Any] = {}
        for template_index, template in enumerate(CONSTRAINT_TEMPLATES):
            values = grouped[model].get(template, [])
            per_template[template] = _bootstrap(
                values,
                n_bootstrap=n_bootstrap,
                seed=seed + model_index * 10 + template_index,
            )
            pooled_by_template[template].extend(values)
            all_scores.extend(values)
        per_model[model] = per_template
    return {
        "analysis_version": "qa-v2-constraint-bootstrap-v1",
        "template_version": "constraint-v2",
        "bootstrap": {"unit": "item", "n_bootstrap": n_bootstrap, "seed": seed},
        "model_template": per_model,
        "pooled_by_template": {
            template: _bootstrap(
                pooled_by_template[template],
                n_bootstrap=n_bootstrap,
                seed=seed + 100 + index,
            )
            for index, template in enumerate(CONSTRAINT_TEMPLATES)
        },
        "project_score": _bootstrap(all_scores, n_bootstrap=n_bootstrap, seed=seed + 200),
    }


def write_constraint_v2_artifacts(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Persist canonical constraint-v2 statistics and a compact comparison figure."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "qa_v2_constraint_summary.json"
    summary_path.write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        models = sorted(summary.get("model_template", {}))
        if models:
            figure, axis = plt.subplots(figsize=(max(6.0, 0.85 * len(models)), 4.2))
            positions = np.arange(len(models), dtype=float)
            for offset, template in zip((-0.18, 0.18), CONSTRAINT_TEMPLATES):
                values = [summary["model_template"][model][template]["mean"] for model in models]
                errors = [
                    (summary["model_template"][model][template]["ci95"][1] - value)
                    if value is not None
                    else 0.0
                    for model, value in zip(models, values)
                ]
                axis.bar(positions + offset, values, width=0.34, yerr=errors, label=template, capsize=3)
            axis.set_xticks(positions, [model.split("/", 1)[-1] for model in models], rotation=35, ha="right")
            axis.set_ylim(0.0, 1.0)
            axis.set_ylabel("Constraint-v2 score")
            axis.legend(frameon=False)
            figure.tight_layout()
            figure.savefig(destination / "qa_v2_constraint_scores.png", dpi=220)
            plt.close(figure)
    except Exception:
        pass
    return summary_path


__all__ = [
    "CONSTRAINT_TEMPLATES",
    "load_constraint_v2_records",
    "summarize_constraint_v2",
    "write_constraint_v2_artifacts",
]
