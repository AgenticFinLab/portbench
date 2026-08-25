"""Offline release-gate checks for the SA-only upgrade experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..qa_eval import paths as qa_paths
from . import paths
from .config import ExperimentConfig
from .providers import spec_model_name, spec_provider_name


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist a gate verdict for a later experiment to consume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _artifact_failures(root: Path) -> list[str]:
    """Return missing or invalid call-parse findings under one call root."""
    failures: list[str] = []
    calls_root = root / "calls"
    if not calls_root.exists():
        return ["no completed call artifacts"]
    for call_path in calls_root.glob("*/*.json"):
        try:
            call = json.loads(call_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(f"unreadable call artifact: {call_path}")
            continue
        if not str(call.get("raw_response", "")).strip():
            failures.append(f"empty completed response: {call_path}")
        key = str(call.get("call_key", ""))
        namespace = call_path.parent.name
        if not list((root / "parses" / namespace).glob(f"{key}.*.json")):
            failures.append(f"missing parse artifact: {call_path}")
    return failures


def _check_episode_logs(
    scenario_dir: Path,
    expected_stages: set[str],
    expected_prefix_stages: set[str],
    label: str,
) -> list[str]:
    """Check one scenario has valid v4 factual and online repair episode logs."""
    findings: list[str] = []
    episode_paths = sorted(scenario_dir.glob("pipeline_logs/*/episodes/*.json"))
    result_path = scenario_dir / "backtest_result.json"
    if not result_path.exists():
        findings.append(f"missing completed backtest result for {label}")
    else:
        try:
            expected_episodes = int(
                json.loads(result_path.read_text(encoding="utf-8")).get("n_rebalances", 0)
            )
        except (OSError, ValueError, json.JSONDecodeError):
            findings.append(f"unreadable backtest result for {label}")
        else:
            if len(episode_paths) != expected_episodes:
                findings.append(
                    f"episode count mismatch for {label}: "
                    f"expected {expected_episodes}, found {len(episode_paths)}"
                )
    if not episode_paths:
        return findings + [f"missing pipeline logs for {label}"]
    for episode_path in episode_paths:
        try:
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            findings.append(f"unreadable episode: {episode_path}")
            continue
        if (
            episode.get("architecture_id") != "SA"
            or episode.get("schema_version") != "pipeline-v4-sa-causal"
        ):
            findings.append(f"non-SA-v4 episode: {episode_path}")
        reused_sources = (episode.get("provenance") or {}).get("reused_stage_sources") or {}
        if set(reused_sources) != expected_prefix_stages:
            findings.append(f"unexpected reused-stage provenance: {episode_path}")
        elif any(str(source) != "pit-repair-v2" for source in reused_sources.values()):
            findings.append(f"invalid reused-stage provenance: {episode_path}")
        if any(str(stage.get("error", "")).strip() for stage in episode.get("stages") or []):
            findings.append(f"stage error: {episode_path}")
        branches = episode.get("interventions") or []
        seen = {
            str(branch.get("stage_id"))
            for branch in branches
            if branch.get("operator") == "repair"
            and branch.get("mode") == "online"
            and not branch.get("error")
        }
        if seen != expected_stages:
            findings.append(f"incomplete online repair branches: {episode_path}")
        for stage in episode.get("stages") or []:
            prompt = str(stage.get("prompt", "")).lower()
            parsed_output = stage.get("parsed_output") or {}
            if parsed_output.get("refused"):
                findings.append(f"invalid model response fallback: {episode_path}")
            if (parsed_output.get("plan") or {}).get("metadata", {}).get("parse_error"):
                findings.append(f"invalid S4 plan fallback: {episode_path}")
            if (parsed_output.get("decision") or {}).get("metadata", {}).get("parse_error"):
                findings.append(f"invalid S5 decision fallback: {episode_path}")
            if stage.get("stage_id") == "S4" and "expected_directions" in prompt:
                findings.append(f"S4 answer leakage: {episode_path}")
            if stage.get("stage_id") == "S5" and "if var is below" in prompt:
                findings.append(f"S5 answer-policy leakage: {episode_path}")
    return findings


def evaluate_sa_pilot(cfg: ExperimentConfig) -> dict[str, Any]:
    """Validate the online factual/repair pilot before the full SA matrix."""
    findings: list[str] = []
    episode_count = 0
    expected_stages = set(cfg.interventions.stages)
    expected_prefix_stages = set(cfg.factual_pit_prefix_stages)
    for spec in cfg.models:
        provider = spec_provider_name(spec)
        model = spec_model_name(spec).replace("/", "_").replace(":", "_") + "__SA"
        timestamp = paths.find_best_run(
            cfg.output_root,
            cfg.rebalance,
            provider,
            model,
            cfg.profiles,
        )
        if timestamp is None:
            findings.append(f"missing run for {provider}/{model}")
            continue
        run_dir = paths.run_dir(cfg.output_root, cfg.rebalance, provider, model, timestamp)
        for profile in cfg.profiles:
            for scenario in cfg.resolved_stress_scenarios():
                scenario_dir = run_dir / profile / f"stress_{scenario}"
                findings.extend(
                    _check_episode_logs(
                        scenario_dir,
                        expected_stages,
                        expected_prefix_stages,
                        scenario,
                    )
                )
                episode_count += len(list(scenario_dir.glob("pipeline_logs/*/episodes/*.json")))
            if cfg.run_normal:
                for normal in cfg.normal_periods:
                    label = f"normal_{normal.label}" if normal.label else "normal"
                    scenario_dir = run_dir / profile / label
                    findings.extend(
                        _check_episode_logs(
                            scenario_dir,
                            expected_stages,
                            expected_prefix_stages,
                            label,
                        )
                    )
                    episode_count += len(list(scenario_dir.glob("pipeline_logs/*/episodes/*.json")))
    if cfg.call_artifact_root:
        findings.extend(_artifact_failures(Path(cfg.call_artifact_root)))
    return {
        "gate": "sa_upgrade_pilot",
        "passed": not findings,
        "episode_count": episode_count,
        "findings": findings,
    }


def evaluate_qa_validation(cfg: ExperimentConfig) -> dict[str, Any]:
    """Validate the preregistered constraint-v2 development-score gate."""
    findings: list[str] = []
    checked: dict[str, dict[str, Any]] = {}
    for spec in cfg.models:
        provider = spec_provider_name(spec)
        model = spec_model_name(spec)
        for template in cfg.qa.templates:
            results_path = (
                qa_paths.qa_root(cfg.output_root)
                / provider
                / model
                / template
                / "results.jsonl"
            )
            scores: dict[str, float] = {}
            if results_path.exists():
                for line in results_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("template_version") != "constraint-v2" or record.get("error"):
                        continue
                    scores[str(record.get("qa_id", ""))] = float(record.get("score", 0.0))
            values = list(scores.values())
            key = f"{provider}/{model}/{template}"
            checked[key] = {
                "n_valid": len(values),
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values)) if values else None,
            }
            if len(values) != cfg.qa.max_pairs_per_template:
                findings.append(f"missing valid responses for {key}")
            elif float(np.mean(values)) >= 0.95 or float(np.std(values)) <= 0.05:
                findings.append(f"discriminative-score gate failed for {key}")
    return {
        "gate": "qa_v2_validation",
        "passed": not findings,
        "metrics": checked,
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    """Write one SA pilot or QA validation verdict from frozen artifacts."""
    parser = argparse.ArgumentParser(description="Validate SA-upgrade experiment release gates")
    parser.add_argument("--config", required=True, help="Pilot or QA validation configuration")
    parser.add_argument("--out", required=True, help="Gate verdict JSON path")
    parser.add_argument("--kind", choices=("pilot", "qa-validation"), required=True)
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    verdict = evaluate_sa_pilot(config) if args.kind == "pilot" else evaluate_qa_validation(config)
    _atomic_write(Path(args.out), verdict)
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
