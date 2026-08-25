"""Audit archived S1-S3 outputs before they are reused in an SA experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .freeze import data_snapshot_hash
from .legacy_cache import iter_episode_json_files


PREFIX_STAGES = ("S1", "S2", "S3")


def _sha256_text(value: str) -> str:
    """Return the SHA-256 digest of one UTF-8 string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one audit report without changing source artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class ReuseAuditFinding:
    """Record the eligibility decision for one archived episode."""

    source_path: str
    decision_date: str
    provider: str
    model: str
    profile: str
    scenario: str
    eligible: bool
    reasons: list[str]


def _coordinates(path: Path, source_root: Path) -> dict[str, str]:
    """Extract legacy run coordinates from the documented experiments layout."""
    relative = path.relative_to(source_root)
    parts = relative.parts
    values = {
        "provider": "",
        "model": "",
        "profile": "",
        "scenario": "",
    }
    if len(parts) >= 6 and parts[0] in {"weekly", "monthly", "quarterly"}:
        values.update(
            {
                "provider": parts[1],
                "model": parts[2],
                "profile": parts[4],
                "scenario": parts[5],
            }
        )
    return values


def _stage_map(episode: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the one S1-S3 record for each archived stage when present."""
    result: dict[str, dict[str, Any]] = {}
    for stage in episode.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        stage_id = str(stage.get("stage_id", "")).upper()
        if stage_id in PREFIX_STAGES and stage_id not in result:
            result[stage_id] = stage
    return result


def _source_contract_reasons(
    episode: dict[str, Any],
    target: dict[str, Any],
    coordinates: dict[str, str],
) -> list[str]:
    """Return every missing or behavior-changing source-contract field."""
    reasons: list[str] = []
    contract = (episode.get("provenance") or {}).get("stage_reuse_contract")
    if not isinstance(contract, dict):
        return ["missing provenance.stage_reuse_contract"]

    for field, expected in target.items():
        actual = contract.get(field)
        if actual is None:
            reasons.append(f"missing source contract field: {field}")
        elif actual != expected:
            reasons.append(f"source contract mismatch: {field}")

    for field in ("provider", "model", "model_revision"):
        actual = contract.get(field)
        if not str(actual or "").strip():
            reasons.append(f"missing source contract field: {field}")
        elif field != "model_revision" and actual != coordinates[field]:
            reasons.append(f"source contract mismatch: {field}")

    stages = contract.get("stages")
    if not isinstance(stages, dict):
        reasons.append("missing source contract stage records")
        return reasons
    for stage_id in PREFIX_STAGES:
        record = stages.get(stage_id)
        if not isinstance(record, dict):
            reasons.append(f"missing source contract stage: {stage_id}")
            continue
        for field in (
            "call_key",
            "prompt_hash",
            "raw_response_hash",
            "visible_input_hash",
            "response_schema_hash",
        ):
            if not str(record.get(field, "")).strip():
                reasons.append(f"missing source contract {stage_id}.{field}")
    return reasons


def audit_stage_reuse(
    source_root: str | Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Return a strict, zero-provider audit for archived S1-S3 reuse."""
    root = Path(source_root)
    target = {
        "architecture_id": "SA",
        "memory_mode": "none",
        "tool_mode": "none",
        "pipeline_schema_version": config.pipeline_schema_version,
        "data_version": config.data_version,
        "data_snapshot_hash": data_snapshot_hash(config.data_dir, config.data_version),
        "generation": asdict(config.generation),
    }
    findings: list[ReuseAuditFinding] = []
    for path in iter_episode_json_files(root):
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        coordinates = _coordinates(path, root)
        reasons = _source_contract_reasons(episode, target, coordinates)
        stages = _stage_map(episode)
        for stage_id in PREFIX_STAGES:
            stage = stages.get(stage_id)
            if stage is None:
                reasons.append(f"missing archived stage: {stage_id}")
                continue
            if not str(stage.get("prompt", "")).strip():
                reasons.append(f"missing archived prompt: {stage_id}")
            elif contract := (episode.get("provenance") or {}).get("stage_reuse_contract"):
                record = (contract.get("stages") or {}).get(stage_id) or {}
                if record.get("prompt_hash") != _sha256_text(str(stage["prompt"])):
                    reasons.append(f"archived prompt hash mismatch: {stage_id}")
            if not str(stage.get("raw_response", "")).strip():
                reasons.append(f"missing archived raw response: {stage_id}")
            elif contract := (episode.get("provenance") or {}).get("stage_reuse_contract"):
                record = (contract.get("stages") or {}).get(stage_id) or {}
                if record.get("raw_response_hash") != _sha256_text(str(stage["raw_response"])):
                    reasons.append(f"archived raw response hash mismatch: {stage_id}")
            if not isinstance(stage.get("parsed_output"), dict):
                reasons.append(f"missing archived parse output: {stage_id}")
        findings.append(
            ReuseAuditFinding(
                source_path=str(path),
                decision_date=str(episode.get("decision_date", "")),
                eligible=not reasons,
                reasons=sorted(set(reasons)),
                **coordinates,
            )
        )

    eligible = [finding for finding in findings if finding.eligible]
    reason_counts: dict[str, int] = {}
    for finding in findings:
        for reason in finding.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "audit_version": "stage-reuse-audit-v1",
        "provider_calls": 0,
        "source_root": str(root),
        "target_contract": target,
        "episodes_scanned": len(findings),
        "eligible_episodes": len(eligible),
        "passed": bool(findings) and len(eligible) == len(findings),
        "reason_counts": dict(sorted(reason_counts.items())),
        "findings": [asdict(finding) for finding in findings],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the strict archived-stage audit and persist its JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    report = audit_stage_reuse(args.source_root, config)
    _atomic_write(Path(args.output), report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "episodes_scanned": report["episodes_scanned"],
                "eligible_episodes": report["eligible_episodes"],
                "provider_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
