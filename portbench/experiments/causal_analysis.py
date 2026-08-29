"""Paired simulator intervention analysis for the SA-only CEPS upgrade."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from portbench.agent_eval.intervention import REPAIR_DEFINITION


STAGES = ("S1", "S2", "S3", "S4", "S5")


def _stage_index(stage_id: str) -> int:
    """Return the canonical stage position for an intervention record."""
    if stage_id not in STAGES:
        raise ValueError(f"unsupported stage id: {stage_id}")
    return STAGES.index(stage_id)


def load_online_repair_records(
    root: str | Path,
    *,
    profile: str = "balanced",
) -> list[dict[str, Any]]:
    """Load eligible v4 online-repair episode records from pipeline logs."""
    root_path = Path(root)
    records: list[dict[str, Any]] = []
    for episode_path in sorted(root_path.glob("*/*/*/*/*/pipeline_logs/*/episodes/*.json")):
        parts = episode_path.relative_to(root_path).parts
        if len(parts) < 9 or parts[3] != profile:
            continue
        provider, model, _run_id, _profile, window = parts[:5]
        try:
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if episode.get("architecture_id") != "SA":
            continue
        if episode.get("schema_version") != "pipeline-v4-sa-causal":
            continue
        for branch in episode.get("interventions") or []:
            if (
                branch.get("operator") != "repair"
                or branch.get("repair_definition") != REPAIR_DEFINITION
                or branch.get("mode") != "online"
                or branch.get("error")
            ):
                continue
            stage_id = str(branch.get("stage_id", ""))
            if stage_id not in STAGES:
                continue
            score_delta = {
                str(key): float(value)
                for key, value in dict(branch.get("score_delta") or {}).items()
                if str(key) in STAGES
            }
            if not score_delta or "ceps_delta" not in branch:
                continue
            records.append(
                {
                    "model": f"{provider}/{model}",
                    "window": window,
                    "decision_date": str(episode.get("decision_date", "")),
                    "profile": profile,
                    "stage_id": stage_id,
                    "score_delta": score_delta,
                    "ceps_delta": float(branch["ceps_delta"]),
                    "source_path": str(episode_path),
                }
            )
    return records


def _moving_block_sample(values: list[float], block_size: int, rng: np.random.Generator) -> list[float]:
    """Resample an ordered series using contiguous blocks with replacement."""
    if not values:
        return []
    n_values = len(values)
    size = max(1, min(int(block_size), n_values))
    out: list[float] = []
    while len(out) < n_values:
        start = int(rng.integers(0, n_values))
        out.extend(values[(start + offset) % n_values] for offset in range(size))
    return out[:n_values]


def _model_window_values(
    records: Iterable[Mapping[str, Any]],
    value_key: str,
) -> dict[str, dict[str, list[float]]]:
    """Group chronological paired effects by model and window."""
    grouped: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if value_key == "ceps_delta":
            value = record.get("ceps_delta")
        else:
            value = dict(record.get("score_delta") or {}).get(value_key)
        if value is None:
            continue
        grouped[str(record["model"])][str(record["window"])].append(
            (str(record["decision_date"]), float(value))
        )
    return {
        model: {
            window: [value for _date, value in sorted(values)]
            for window, values in windows.items()
        }
        for model, windows in grouped.items()
    }


def _point_estimate(grouped: Mapping[str, Mapping[str, list[float]]]) -> tuple[float, int, int]:
    """Average dates within each window, windows within model, and models equally."""
    per_model = []
    n_raw = 0
    n_windows = 0
    for windows in grouped.values():
        window_means = []
        for values in windows.values():
            if values:
                n_raw += len(values)
                n_windows += 1
                window_means.append(float(np.mean(values)))
        if window_means:
            per_model.append(float(np.mean(window_means)))
    if not per_model:
        return float("nan"), 0, 0
    return float(np.mean(per_model)), n_raw, n_windows


def _bootstrap_summary(
    records: Iterable[Mapping[str, Any]],
    value_key: str,
    *,
    n_bootstrap: int,
    seed: int,
    block_size: int,
) -> dict[str, Any]:
    """Compute clustered moving-block bootstrap statistics for one paired effect."""
    grouped = _model_window_values(records, value_key)
    effect, n_raw, n_windows = _point_estimate(grouped)
    models = sorted(grouped)
    if not models:
        return {"effect": None, "ci95": [None, None], "n_raw": 0, "n_windows": 0, "n_models": 0, "p_value": None}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_models = rng.choice(models, size=len(models), replace=True)
        model_means = []
        for model in sampled_models:
            window_means = []
            for values in grouped[str(model)].values():
                sample = _moving_block_sample(values, block_size, rng)
                window_means.append(float(np.mean(sample)))
            model_means.append(float(np.mean(window_means)))
        draws[index] = float(np.mean(model_means))
    p_value = min(1.0, 2.0 * min(float(np.mean(draws <= 0.0)), float(np.mean(draws >= 0.0))))
    return {
        "effect": effect,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "n_raw": n_raw,
        "n_windows": n_windows,
        "n_models": len(models),
        "p_value": p_value,
    }


def _bh_fdr(items: list[dict[str, Any]]) -> None:
    """Attach Benjamini-Hochberg adjusted q-values to in-place result records."""
    valid = sorted(
        (item for item in items if item.get("p_value") is not None),
        key=lambda item: float(item["p_value"]),
    )
    total = len(valid)
    running = 1.0
    for rank, item in reversed(list(enumerate(valid, start=1))):
        candidate = float(item["p_value"]) * total / rank
        running = min(running, candidate)
        item["fdr_q_value"] = running


def evaluate_causal_mechanism(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Check that oracle repair is score-compatible and changes a downstream suffix."""
    source_records = list(records)
    findings: list[str] = []
    direct_effects: dict[str, list[float]] = {stage: [] for stage in STAGES}
    downstream_effects: list[float] = []
    for record in source_records:
        stage = str(record.get("stage_id", ""))
        scores = dict(record.get("score_delta") or {})
        if stage in direct_effects and stage in scores:
            direct_effects[stage].append(float(scores[stage]))
        if stage in STAGES:
            stage_index = STAGES.index(stage)
            downstream_effects.extend(
                float(scores[destination])
                for destination in STAGES[stage_index + 1 :]
                if destination in scores
            )
    for stage, values in direct_effects.items():
        if not values:
            findings.append(f"missing oracle repair records for {stage}")
        elif min(values) < -1e-9:
            findings.append(f"oracle repair reduced the directly repaired {stage} score")
    positive_direct = sum(
        value > 1e-6 for values in direct_effects.values() for value in values
    )
    if positive_direct == 0:
        findings.append("oracle repair produced no non-trivial direct improvement")
    nonzero_downstream = sum(abs(value) > 1e-6 for value in downstream_effects)
    if nonzero_downstream == 0:
        findings.append("oracle repair produced no measurable downstream response")
    return {
        "passed": not findings,
        "repair_definition": REPAIR_DEFINITION,
        "n_records": len(source_records),
        "positive_direct_effects": positive_direct,
        "nonzero_downstream_effects": nonzero_downstream,
        "direct_minima": {
            stage: min(values) if values else None
            for stage, values in direct_effects.items()
        },
        "findings": findings,
    }


def summarize_causal_attribution(
    records: Iterable[Mapping[str, Any]],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    block_size: int = 3,
) -> dict[str, Any]:
    """Estimate the upper-triangular 5x5 paired simulator influence matrix."""
    source_records = list(records)
    matrix: dict[str, dict[str, dict[str, Any]]] = {stage: {} for stage in STAGES}
    all_matrix_items: list[dict[str, Any]] = []
    ceps_effects: dict[str, dict[str, Any]] = {}
    decomposition: dict[str, dict[str, Any]] = {}
    for source_index, source in enumerate(STAGES):
        branch_records = [item for item in source_records if item.get("stage_id") == source]
        ceps_effects[source] = _bootstrap_summary(
            branch_records,
            "ceps_delta",
            n_bootstrap=n_bootstrap,
            seed=seed + source_index,
            block_size=block_size,
        )
        downstream_effects = []
        for destination_index, destination in enumerate(STAGES):
            if destination_index < source_index:
                continue
            summary = _bootstrap_summary(
                branch_records,
                destination,
                n_bootstrap=n_bootstrap,
                seed=seed + 100 + source_index * len(STAGES) + destination_index,
                block_size=block_size,
            )
            matrix[source][destination] = summary
            all_matrix_items.append(summary)
            if destination_index > source_index and summary["effect"] is not None:
                downstream_effects.append(float(summary["effect"]))
        direct = matrix[source][source]["effect"]
        decomposition[source] = {
            "direct_stage_improvement": direct,
            "downstream_propagation_effect": (
                float(np.mean(downstream_effects)) if downstream_effects else None
            ),
            "delta_ceps": ceps_effects[source]["effect"],
        }
    _bh_fdr(all_matrix_items)
    return {
        "analysis_protocol": "paired simulator intervention / within-simulator stage attribution",
        "operator": "repair",
        "repair_definition": REPAIR_DEFINITION,
        "mode": "online",
        "closed_loop": False,
        "bootstrap": {
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "block_size": block_size,
            "within_window": "moving-block",
            "pooled_models": "clustered macro-average",
            "window_aggregation": "equal-weight",
            "matrix_fdr": "BH-FDR 0.05",
        },
        "n_intervention_records": len(source_records),
        "mechanism_gate": evaluate_causal_mechanism(source_records),
        "influence_matrix": matrix,
        "delta_ceps": ceps_effects,
        "decomposition": decomposition,
    }


def write_causal_artifacts(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write a canonical JSON summary and a labeled influence-matrix heatmap."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "causal_attribution_summary.json"
    summary_path.write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        data = np.full((len(STAGES), len(STAGES)), np.nan)
        for row, source in enumerate(STAGES):
            for column, destination_stage in enumerate(STAGES):
                value = summary["influence_matrix"].get(source, {}).get(destination_stage, {}).get("effect")
                if value is not None:
                    data[row, column] = float(value)
        figure, axis = plt.subplots(figsize=(6.2, 5.0))
        image = axis.imshow(data, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axis.set_xticks(range(len(STAGES)), STAGES)
        axis.set_yticks(range(len(STAGES)), STAGES)
        axis.set_xlabel("Scored stage after repair")
        axis.set_ylabel("Repaired stage")
        for row in range(len(STAGES)):
            for column in range(len(STAGES)):
                if not np.isnan(data[row, column]):
                    axis.text(column, row, f"{data[row, column]:.3f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, label="Paired score delta")
        figure.tight_layout()
        figure.savefig(destination / "stage_influence_matrix.png", dpi=220)
        plt.close(figure)
    except Exception:
        pass
    return summary_path


def main(argv: list[str] | None = None) -> int:
    """Run offline causal aggregation from completed pipeline log artifacts."""
    parser = argparse.ArgumentParser(description="Analyze paired simulator intervention artifacts")
    parser.add_argument("--input", required=True, help="Rebalance directory containing provider/model runs")
    parser.add_argument("--output", required=True, help="Output directory for causal analysis artifacts")
    parser.add_argument("--bootstrap", type=int, default=10_000, help="Number of bootstrap draws")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed")
    parser.add_argument("--block-size", type=int, default=3, help="Moving-block length in rebalance dates")
    parser.add_argument("--gate-output", help="Optional mechanism-gate JSON path")
    args = parser.parse_args(argv)
    records = load_online_repair_records(args.input)
    summary = summarize_causal_attribution(
        records,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        block_size=args.block_size,
    )
    output = write_causal_artifacts(summary, args.output)
    if args.gate_output:
        gate_path = Path(args.gate_output)
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate = {
            **summary["mechanism_gate"],
            "source_root": str(args.input),
            "analysis_summary": str(output),
        }
        gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(output)
    return 0 if summary["mechanism_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
