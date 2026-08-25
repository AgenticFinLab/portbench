"""Freeze exact locked constraint-template test items used for provider calls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from portbench.agent_eval.canonical import canonical_json, sha256_hex


def _pair_id(pair: dict[str, Any]) -> str:
    """Return the stable QA identifier from either supported dataset layout."""
    return str(pair.get("qa_id", pair.get("id", "")))


def freeze_constraint_test(
    dataset_dir: str | Path,
    *,
    template_version: str,
    max_pairs_per_template: int = 50,
    output_path: str | Path | None = None,
) -> Path:
    """Write an immutable selection manifest without changing source test records."""
    if template_version not in {
        "constraint-v2",
        "constraint-decision-v2",
    }:
        raise ValueError(f"Unsupported constraint template version: {template_version}")
    root = Path(dataset_dir)
    test_path = root / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"constraint test split not found: {test_path}")
    records = [
        json.loads(line)
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: dict[str, list[dict[str, str]]] = {}
    generator_versions: set[str] = set()
    for template in ("T3", "T4"):
        candidates = [
            pair
            for pair in records
            if pair.get("template_id", pair.get("template")) == template
            and (pair.get("metadata") or {}).get("template_version") == template_version
        ]
        if len(candidates) < max_pairs_per_template:
            raise ValueError(
                f"{template} has only {len(candidates)} {template_version} test items; "
                f"need {max_pairs_per_template}"
            )
        chosen = candidates[:max_pairs_per_template]
        selected[template] = [
            {"qa_id": _pair_id(pair), "pair_hash": sha256_hex(canonical_json(pair))}
            for pair in chosen
        ]
        generator_versions.update(
            str((pair.get("metadata") or {}).get("generator_version", ""))
            for pair in chosen
        )
    payload = {
        "manifest_version": f"qa-{template_version}-test-v1",
        "template_version": template_version,
        "max_pairs_per_template": max_pairs_per_template,
        "generator_versions": sorted(generator_versions),
        "selected": selected,
    }
    filename = f"{template_version.replace('-', '_')}_test_manifest.json"
    destination = Path(output_path) if output_path else root / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def freeze_constraint_v2_test(
    dataset_dir: str | Path,
    *,
    max_pairs_per_template: int = 50,
    output_path: str | Path | None = None,
) -> Path:
    """Freeze the exact locked constraint-v2 test items used for provider calls."""
    return freeze_constraint_test(
        dataset_dir,
        template_version="constraint-v2",
        max_pairs_per_template=max_pairs_per_template,
        output_path=output_path,
    )


def freeze_constraint_decision_v2_test(
    dataset_dir: str | Path,
    *,
    max_pairs_per_template: int = 50,
    output_path: str | Path | None = None,
) -> Path:
    """Freeze the exact compact T3-D and T4-D test items used for provider calls."""
    return freeze_constraint_test(
        dataset_dir,
        template_version="constraint-decision-v2",
        max_pairs_per_template=max_pairs_per_template,
        output_path=output_path,
    )


def main(argv: list[str] | None = None) -> int:
    """Freeze a deterministic 50-by-2 provider-evaluation selection."""
    parser = argparse.ArgumentParser(description="Freeze constraint QA test items")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--template-version",
        choices=(
            "constraint-v2",
            "constraint-decision-v2",
        ),
        required=True,
    )
    parser.add_argument("--max-pairs-per-template", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    print(
        freeze_constraint_test(
            args.dataset_dir,
            template_version=args.template_version,
            max_pairs_per_template=args.max_pairs_per_template,
            output_path=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
