"""
QAEvaluator: run LLM models against QA dataset pairs and collect scored results.

Mirrors BatchRunner structure: checkpoint-based resume, per-question error isolation,
thread-parallel question evaluation, and structured artifact persistence.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm.auto import tqdm

from ..agent_eval.call_artifacts import CallArtifactStore, CallRequest
from ..agent_eval.canonical import canonical_json, sha256_hex
from ..experiments.config import ExperimentConfig, ModelSpec
from ..experiments.providers import build_adapter, spec_provider_name, spec_model_name
from . import paths as qpaths
from .scorer import score_response


def _spec_display_label(spec: ModelSpec) -> str:
    """Human-readable label for tqdm / summary metadata (not used for paths)."""
    prov = spec_provider_name(spec)
    model = spec_model_name(spec)
    return f"{prov}/{model}"


def _rebuild_summary_from_results(t_dir: Path, template_id: str) -> dict | None:
    """
    Reconstruct a template summary dict from results.jsonl.
    Returns None if the file is missing or empty.
    Deduplicates by qa_id (keeps last occurrence).
    """
    results_file = t_dir / "results.jsonl"
    if not results_file.exists():
        return None
    try:
        seen: dict[str, dict] = {}
        for line in results_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qa_id = r.get("qa_id", r.get("id", ""))
            seen[qa_id] = r  # last write wins on duplicates
        records = list(seen.values())
        if not records:
            return None
        scores = [float(r.get("score", 0.0)) for r in records]
        by_regime: dict[str, list[float]] = {}
        for r in records:
            regime = r.get("regime") or r.get("market_regime", "unknown")
            by_regime.setdefault(regime, []).append(float(r.get("score", 0.0)))
        return {
            "template_id": template_id,
            "accuracy": round(float(np.mean(scores)), 4),
            "n_total": len(scores),
            "n_correct": sum(1 for s in scores if s >= 0.99),
            "by_regime": {
                reg: round(float(np.mean(ss)), 4)
                for reg, ss in sorted(by_regime.items())
            },
            "scores": [round(s, 4) for s in scores],
        }
    except Exception:
        return None


def _load_qa_pairs(dataset_path: str, split: str) -> list[dict]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"QA dataset not found: {path}\n"
            "Run: python examples/qa_builder/build_qa_dataset.py"
        )

    # Support both directory layout (train.jsonl/val.jsonl/test.jsonl)
    # and single-file layout (qa_dataset.jsonl with "split" field per line)
    if path.is_dir():
        if split == "all":
            target = path / "all_pairs.jsonl"
            if not target.exists():
                # Merge all split files
                targets = [path / f"{s}.jsonl" for s in ("train", "val", "test")]
                targets = [t for t in targets if t.exists()]
            else:
                targets = [target]
        else:
            target = path / f"{split}.jsonl"
            if not target.exists():
                raise FileNotFoundError(f"Split file not found: {target}")
            targets = [target]
        pairs = []
        for t in targets:
            with open(t, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pairs.append(json.loads(line))
        return pairs

    # Single file — filter by split field
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if split == "all" or rec.get("split") == split:
                    pairs.append(rec)
    return pairs


def _apply_freeze_manifest(
    pairs: list[dict],
    manifest_path: str,
    expected_template_version: str = "",
) -> list[dict]:
    """Select and verify the locked constraint test items before any call."""
    if not manifest_path:
        return pairs
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest_version = str(manifest.get("template_version", ""))
    if manifest_version not in {
        "constraint-v2",
        "constraint-decision-v2",
    }:
        raise ValueError("QA freeze manifest has an unsupported template version")
    if expected_template_version and manifest_version != expected_template_version:
        raise ValueError("QA freeze manifest does not match the requested template version")
    indexed = {
        str(pair.get("qa_id", pair.get("id", ""))): pair
        for pair in pairs
    }
    selected: list[dict] = []
    for template in ("T3", "T4"):
        for entry in dict(manifest.get("selected") or {}).get(template, []):
            qa_id = str(entry.get("qa_id", ""))
            pair = indexed.get(qa_id)
            if pair is None:
                raise ValueError(f"frozen QA item is missing from the loaded split: {qa_id}")
            if sha256_hex(canonical_json(pair)) != entry.get("pair_hash"):
                raise ValueError(f"frozen QA item changed after manifest creation: {qa_id}")
            selected.append(pair)
    if not selected:
        raise ValueError("QA freeze manifest selects no items")
    return selected


def _strip_covariance_from_pair(pair: dict, base_tid: str) -> dict:
    """Return a shallow-copy of pair with covariance/correlation info removed from question.

    T4: removes the "Covariance(A,B) = X, Correlation = Y" line.
    T5: removes the "Covariance matrix (annualized):" header line and the matrix line that follows.
    """
    question = pair.get("question", "")
    if base_tid == "T4":
        lines = question.split("\n")
        question = "\n".join(l for l in lines if not l.startswith("Covariance("))
    elif base_tid == "T5":
        lines = question.split("\n")
        result, skip_next = [], False
        for line in lines:
            if skip_next:
                skip_next = False
                continue
            if line.startswith("Covariance matrix"):
                skip_next = True
                continue
            result.append(line)
        question = "\n".join(result)
    return {**pair, "question": question}


def _build_eval_prompt(
    pair: dict,
    t3t4_redesign: bool = False,
    template_version: str = "legacy",
) -> str:
    context = pair.get("context_summary", "")
    question = pair.get("question", "")
    tid = pair.get("template_id", pair.get("template", ""))
    meta = pair.get("metadata") or {}
    use_constraint_template = template_version in {
        "constraint-v2",
        "constraint-decision-v2",
    } and tid in (
        "T3",
        "T4",
    )
    use_json = bool(t3t4_redesign or meta.get("t3t4_redesign")) and tid in (
        "T3",
        "T4",
        "T4_restricted",
    )
    if use_constraint_template:
        instructions = (
            "Instructions:\n"
            "- Reply with a single JSON object only.\n"
            "- Use exactly the fields and numeric conventions required in the question.\n"
            "- Do not include markdown fences or any text outside the JSON object.\n\n"
            "Answer:"
        )
    elif use_json:
        instructions = (
            "Instructions:\n"
            "- Reply with a single JSON object only\n"
            '- Required keys: "answer" and "explanation"\n'
            "- Follow the answer format specified in the question\n"
            "- Keep the explanation to 1-3 short sentences\n\n"
            "Answer:"
        )
    else:
        instructions = (
            "Instructions:\n"
            "- Answer directly and concisely\n"
            "- Follow exactly the answer format specified in the question above\n"
            "- For direction prediction: reply with one word — positive, negative, or flat\n"
            "- For numeric answers: provide a single decimal number (e.g., -0.02 or 0.75)\n"
            "- For portfolio weight answers: provide decimals summing to 1.0 (e.g., 0.60 not 60%)\n\n"
            "Answer:"
        )
    return (
        "[PORTFOLIO MANAGEMENT QA]\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"{instructions}"
    )


def _parse_qa_response(raw: str, template_version: str, template_id: str) -> dict:
    """Validate one QA response before it is eligible for durable reuse."""
    response = str(raw or "").strip()
    if not response:
        raise ValueError("empty QA response")
    if template_version not in {
        "constraint-v2",
        "constraint-decision-v2",
    }:
        return {"raw_response": response}
    payload = json.loads(response)
    if not isinstance(payload, dict):
        raise ValueError("constraint response must be a JSON object")
    base_tid = template_id.replace("_restricted", "")
    if template_version == "constraint-decision-v2":
        return _validate_constraint_decision_payload(payload, base_tid)
    if base_tid == "T3":
        required = {"position_size", "binding_constraint", "constraint_margins"}
    elif base_tid == "T4":
        required = {"candidate_id", "weights", "calculated_metrics", "binding_constraints"}
    else:
        required = set()
    missing = required - set(payload)
    if missing:
        raise ValueError(f"constraint-v2 response missing keys: {sorted(missing)}")
    if base_tid == "T3":
        position_size = payload["position_size"]
        if not isinstance(position_size, (int, float)) or not 0.0 <= float(position_size) <= 1.0:
            raise ValueError("T3 position_size must be a number in [0, 1]")
        if payload["binding_constraint"] not in {
            "var",
            "es",
            "drawdown",
            "liquidity",
            "full_allocation",
        }:
            raise ValueError("T3 binding_constraint is invalid")
        margins = payload["constraint_margins"]
        if not isinstance(margins, dict) or set(margins) != {
            "var",
            "es",
            "drawdown",
            "liquidity",
            "full_allocation",
        }:
            raise ValueError("T3 constraint_margins must contain every named constraint")
        if any(not isinstance(value, (int, float)) for value in margins.values()):
            raise ValueError("T3 constraint_margins values must be numeric")
    if base_tid == "T4":
        weights = payload["weights"]
        metrics = payload["calculated_metrics"]
        if not isinstance(payload["candidate_id"], str) or not payload["candidate_id"]:
            raise ValueError("T4 candidate_id must be non-empty")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("T4 weights must be a non-empty object")
        if any(not isinstance(value, (int, float)) or float(value) < 0.0 for value in weights.values()):
            raise ValueError("T4 weights must be non-negative numbers")
        if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-4:
            raise ValueError("T4 weights must sum to one")
        if not isinstance(metrics, dict) or set(metrics) != {
            "expected_return",
            "variance",
            "turnover",
        }:
            raise ValueError("T4 calculated_metrics must contain all required metrics")
        if any(not isinstance(value, (int, float)) for value in metrics.values()):
            raise ValueError("T4 calculated_metrics values must be numeric")
        if not isinstance(payload["binding_constraints"], list):
            raise ValueError("T4 binding_constraints must be a list")
    return payload


def _validate_constraint_decision_payload(payload: dict, template_id: str) -> dict:
    """Validate the compact T3-D or T4-D decision response schema."""
    prefix = "plan" if template_id == "T3" else "candidate"
    required = {
        f"base_{prefix}_id",
        f"stress_{prefix}_id",
        "base_feasible_ids",
        "stress_feasible_ids",
        "base_binding_constraint",
        "stress_binding_constraint",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"constraint-decision response missing keys: {sorted(missing)}")
    for key in (f"base_{prefix}_id", f"stress_{prefix}_id"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ValueError(f"constraint-decision {key} must be a non-empty string")
    for key in ("base_feasible_ids", "stress_feasible_ids"):
        values = payload[key]
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"constraint-decision {key} must be a list of strings")
        if len(values) != len(set(values)):
            raise ValueError(f"constraint-decision {key} must not repeat IDs")
    for key in ("base_binding_constraint", "stress_binding_constraint"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ValueError(f"constraint-decision {key} must be a non-empty string")
    return payload


def _checkpoint_key(request: CallRequest) -> str:
    """Use the complete effective QA request as the resume checkpoint key."""
    return request.call_key


class QAEvaluator:
    def __init__(self, cfg: ExperimentConfig, raw_yaml: Optional[str] = None):
        self.cfg = cfg
        self._raw_yaml = raw_yaml

        all_pairs = _load_qa_pairs(cfg.qa.dataset_path, cfg.qa.split)
        all_pairs = _apply_freeze_manifest(
            all_pairs,
            cfg.qa.freeze_manifest,
            cfg.qa.template_version,
        )

        # Filter by templates (support both "template_id" and "template" field names)
        templates = set(cfg.qa.templates)
        all_pairs = [
            p for p in all_pairs
            if p.get("template_id", p.get("template", "")) in templates
        ]

        # Group by template and cap
        self._pairs_by_template: dict[str, list[dict]] = {}
        for p in all_pairs:
            tid = p.get("template_id", p.get("template", ""))
            if tid not in self._pairs_by_template:
                self._pairs_by_template[tid] = []
            if len(self._pairs_by_template[tid]) < cfg.qa.max_pairs_per_template:
                self._pairs_by_template[tid].append(p)

        # Restricted-info variants: duplicate T4/T5 under T4_restricted/T5_restricted
        info_level = cfg.qa.info_level
        if info_level in ("restricted", "both"):
            for base_tid in ("T4", "T5"):
                if base_tid in self._pairs_by_template:
                    self._pairs_by_template[f"{base_tid}_restricted"] = list(
                        self._pairs_by_template[base_tid]
                    )
        if info_level == "restricted":
            # Drop full-info T4/T5 so only restricted variants run
            for base_tid in ("T4", "T5"):
                self._pairs_by_template.pop(base_tid, None)

        # Filter models to LLM-only (skip baseline / mock)
        self._llm_specs = [s for s in cfg.models if s.kind() == "llm"]

    def dry_run(self) -> list[dict]:
        out = []
        for spec in self._llm_specs:
            label = _spec_display_label(spec)
            for tid, pairs in sorted(self._pairs_by_template.items()):
                out.append({
                    "model": label,
                    "template": tid,
                    "n_questions": len(pairs),
                })
        return out

    def run(self) -> dict:
        cfg = self.cfg
        root = qpaths.qa_root(cfg.output_root)
        root.mkdir(parents=True, exist_ok=True)

        model_summaries: dict[str, dict] = {}
        t0 = time.time()

        n_total = sum(
            len(pairs)
            for pairs in self._pairs_by_template.values()
        ) * len(self._llm_specs)

        n_models = len(self._llm_specs)
        templates_order = sorted(self._pairs_by_template.keys())

        pbar = tqdm(total=n_total, desc="qa_eval", unit="q", dynamic_ncols=True)

        for i, spec in enumerate(self._llm_specs):
            provider = spec_provider_name(spec)
            model_name = spec_model_name(spec)
            label = _spec_display_label(spec)

            tqdm.write(f"\n[QA] Model {i+1}/{n_models}: {label}")
            tqdm.write(f"      templates: {', '.join(templates_order)}")

            try:
                adapter = build_adapter(
                    spec.provider, spec.model,
                    temperature=spec.temperature if spec.temperature is not None else self.cfg.generation.temperature,
                    max_tokens=spec.max_tokens if spec.max_tokens is not None else self.cfg.generation.max_tokens,
                    timeout=self.cfg.timeout,
                )
            except Exception as exc:
                tqdm.write(f"[QA] adapter build failed for {label}: {exc}")
                pbar.update(sum(len(p) for p in self._pairs_by_template.values()))
                continue

            model_summary = self._run_model_qa(
                spec, adapter, provider, model_name, label, pbar,
            )
            model_summaries[label] = model_summary

            mean_acc = model_summary.get("mean_accuracy", 0.0)
            n_done = model_summary.get("n_total", 0)
            tqdm.write(f"[QA] completed {label}  mean_accuracy={mean_acc:.3f}  n={n_done}")

        pbar.close()

        summary = {
            "n_models": len(model_summaries),
            "elapsed_seconds": round(time.time() - t0, 2),
            "models": model_summaries,
        }
        (root / "qa_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        return summary

    def _run_model_qa(
        self,
        spec: ModelSpec,
        adapter,
        provider: str,
        model_name: str,
        label: str,
        pbar,
    ) -> dict:
        cfg = self.cfg
        ckpt_path = qpaths.qa_checkpoint_file(cfg.output_root, provider, model_name)
        completed_keys = qpaths.load_checkpoint(ckpt_path) if cfg.reuse_latest else set()
        ckpt_lock = threading.Lock()

        template_summaries: dict[str, dict] = {}

        for tid in sorted(self._pairs_by_template.keys()):
            pairs = self._pairs_by_template[tid]
            model_parallel = (
                spec.parallel_questions
                if spec.parallel_questions is not None
                else cfg.qa.parallel_questions
            )
            pbar.set_description(f"{label.split('/')[-1]} {tid}")
            tqdm.write(f"  -> {tid}  ({len(pairs)} questions, parallel={model_parallel})")
            t_summary = self._run_template(
                adapter, provider, model_name, label, tid, pairs,
                completed_keys, ckpt_lock, ckpt_path, pbar,
                parallel_questions=model_parallel,
            )
            template_summaries[tid] = t_summary
            acc = t_summary.get("accuracy", 0.0)
            n = t_summary.get("n_total", 0)
            tqdm.write(f"     {tid} done  accuracy={acc:.3f}  n={n}")

        # Write model-level summary
        # Aggregate scores — exclude _restricted variants from the mean_accuracy
        # so the primary summary reflects full-info performance only.
        all_scores = []
        for tid, ts in template_summaries.items():
            if not tid.endswith("_restricted"):
                all_scores.extend(ts.get("scores", []))

        model_summary = {
            "provider": provider,
            "model": model_name,
            "mean_accuracy": round(float(np.mean(all_scores)), 4) if all_scores else 0.0,
            "n_total": len(all_scores),
            "n_correct": sum(1 for s in all_scores if s >= 0.99),
            "templates": {
                tid: {k: v for k, v in ts.items() if k != "scores"}
                for tid, ts in template_summaries.items()
            },
        }

        m_dir = qpaths.qa_model_dir(cfg.output_root, provider, model_name)
        m_dir.mkdir(parents=True, exist_ok=True)
        # Use a separate file for restricted runs to avoid overwriting full-info summary
        info_level = cfg.qa.info_level
        summary_fname = (
            "qa_model_summary_restricted.json"
            if info_level == "restricted"
            else "qa_model_summary.json"
        )
        (m_dir / summary_fname).write_text(
            json.dumps(model_summary, indent=2, default=str), encoding="utf-8"
        )

        # Render figures
        if cfg.logging.save_figures:
            try:
                self._render_model_figures(provider, model_name, label, template_summaries)
            except Exception:
                pass

        return model_summary

    def _run_template(
        self,
        adapter,
        provider: str,
        model_name: str,
        label: str,
        template_id: str,
        pairs: list[dict],
        completed_keys: set[str],
        ckpt_lock: threading.Lock,
        ckpt_path: Path,
        pbar,
        parallel_questions: int = 4,
    ) -> dict:
        cfg = self.cfg
        t_dir = qpaths.qa_template_dir(
            cfg.output_root, provider, model_name, template_id
        )
        t_dir.mkdir(parents=True, exist_ok=True)
        artifact_root = (
            Path(cfg.call_artifact_root) / "qa" / provider / model_name / template_id
            if cfg.call_artifact_root
            else t_dir / "call_artifacts"
        )
        artifact_store = CallArtifactStore(artifact_root)

        is_restricted = template_id.endswith("_restricted")
        base_tid = template_id.replace("_restricted", "") if is_restricted else template_id

        scores: list[float] = []
        by_regime: dict[str, list[float]] = {}
        results_lock = threading.Lock()

        def _eval_one(pair: dict) -> None:
            qa_id = pair.get("qa_id", pair.get("id", ""))
            t0 = time.time()
            try:
                eval_pair = (
                    _strip_covariance_from_pair(pair, base_tid) if is_restricted else pair
                )
                prompt = _build_eval_prompt(
                    eval_pair,
                    t3t4_redesign=cfg.qa.t3t4_redesign,
                    template_version=cfg.qa.template_version,
                )
                request = CallRequest(
                    provider=provider,
                    model=model_name,
                    model_revision=str(getattr(adapter, "_model_revision", "")),
                    stage_id=f"QA:{template_id}",
                    system_prompt=str(getattr(adapter, "_system_prompt", "")),
                    user_prompt=prompt,
                    response_schema={"template_version": cfg.qa.template_version},
                    generation_config={
                        "temperature": getattr(adapter, "_temperature", None),
                        "max_tokens": getattr(adapter, "_max_tokens", None),
                        "timeout": getattr(adapter, "_timeout", None),
                    },
                    visible_input={
                        "qa_id": qa_id,
                        "template_id": template_id,
                        "context_summary": eval_pair.get("context_summary", ""),
                        "question": eval_pair.get("question", ""),
                    },
                    data_version=str((pair.get("metadata") or {}).get("generator_version", "")),
                )
                ck = _checkpoint_key(request)
                if ck in completed_keys:
                    pbar.update(1)
                    return
                _, call_artifact, _ = artifact_store.complete_or_call(
                    request,
                    parser_version=f"qa-{cfg.qa.template_version}-parser-v1",
                    parse=lambda raw: _parse_qa_response(
                        raw,
                        cfg.qa.template_version,
                        template_id,
                    ),
                    call_fn=lambda: adapter.complete(prompt),
                    provenance={"qa_id": qa_id, "template_id": template_id},
                    max_attempts=cfg.qa.call_max_attempts,
                    retry_failed=cfg.qa.retry_failed_calls,
                )
                response = call_artifact.raw_response
                latency = time.time() - t0

                meta = pair.get("metadata") or {}
                score_artifact = artifact_store.score(
                    request,
                    f"qa-{cfg.qa.template_version}-parser-v1",
                    cfg.qa.scorer_version,
                    {
                        "answer": pair.get("answer", ""),
                        "answer_numeric": pair.get("answer_numeric"),
                        "assets": pair.get("assets"),
                        "metadata": meta,
                    },
                    lambda _: score_response(
                        template_id=base_tid,
                        gt_answer=pair.get("answer", ""),
                        llm_response=response,
                        answer_numeric=pair.get("answer_numeric"),
                        assets=pair.get("assets"),
                        redesign=bool(
                            cfg.qa.t3t4_redesign or meta.get("t3t4_redesign")
                        ),
                        explanation_keypoints=meta.get("explanation_keypoints"),
                        numeric_weight=cfg.qa.t3t4_numeric_weight,
                        explanation_weight=cfg.qa.t3t4_explanation_weight,
                        template_version=cfg.qa.template_version,
                        metadata=meta,
                    ),
                )
                sc = float(score_artifact.score_payload)

                record = {
                    "qa_id": qa_id,
                    "template_id": template_id,
                    "score": round(sc, 4),
                    "response": response if cfg.qa.save_responses else "",
                    "call_key": request.call_key,
                    "latency": round(latency, 2),
                    "regime": pair.get("market_regime", ""),
                    "complexity": pair.get("complexity", ""),
                    "split": pair.get("split", ""),
                    "template_version": cfg.qa.template_version,
                    "scorer_version": cfg.qa.scorer_version,
                }
            except Exception as exc:
                latency = time.time() - t0
                sc = 0.0
                record = {
                    "qa_id": qa_id,
                    "template_id": template_id,
                    "score": 0.0,
                    "error": str(exc)[:200],
                    "latency": round(latency, 2),
                    "regime": pair.get("market_regime", ""),
                    "complexity": pair.get("complexity", ""),
                    "split": pair.get("split", ""),
                }
                # Do NOT checkpoint errors/timeouts — allow them to be retried on resume.
                with results_lock:
                    qpaths.append_result(t_dir, record)
                    scores.append(sc)
                    regime = pair.get("market_regime", "unknown")
                    by_regime.setdefault(regime, []).append(sc)
                pbar.update(1)
                return

            with results_lock:
                qpaths.append_result(t_dir, record)
                scores.append(sc)
                regime = pair.get("market_regime", "unknown")
                by_regime.setdefault(regime, []).append(sc)

            with ckpt_lock:
                completed_keys.add(ck)
                qpaths.write_checkpoint(ckpt_path, completed_keys)

            pbar.update(1)

        max_workers = max(1, parallel_questions)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_eval_one, p): p for p in pairs}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception:
                    pass

        # Always rebuild summary from the full results.jsonl so that previously
        # checkpointed results (skipped this run) are included alongside new ones.
        rebuilt = _rebuild_summary_from_results(t_dir, template_id)
        if rebuilt is not None:
            summary = rebuilt
        else:
            # Fallback: only new scores (e.g. empty results.jsonl)
            summary = {
                "template_id": template_id,
                "accuracy": round(float(np.mean(scores)), 4) if scores else 0.0,
                "n_total": len(scores),
                "n_correct": sum(1 for s in scores if s >= 0.99),
                "by_regime": {
                    r: round(float(np.mean(ss)), 4)
                    for r, ss in sorted(by_regime.items())
                },
                "scores": [round(s, 4) for s in scores],
            }
        (t_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    def _render_model_figures(
        self,
        provider: str,
        model_name: str,
        label: str,
        template_summaries: dict[str, dict],
    ) -> None:
        from ..visualization.qa_accuracy_plots import (
            plot_qa_accuracy_heatmap,
            plot_qa_accuracy_by_regime,
            plot_qa_score_distribution,
        )
        from ..visualization.style import save_figure

        cfg = self.cfg
        fig_dir = qpaths.qa_figures_dir(cfg.output_root, provider, model_name)
        fig_dir.mkdir(parents=True, exist_ok=True)

        acc_data = {
            label: {
                tid: ts["accuracy"]
                for tid, ts in template_summaries.items()
            }
        }
        fig = plot_qa_accuracy_heatmap(acc_data, title=f"QA Accuracy — {label}")
        save_figure(fig, str(fig_dir / "accuracy_by_template.png"), formats=("png",))

        regime_data = {
            label: {
                tid: ts.get("by_regime", {})
                for tid, ts in template_summaries.items()
            }
        }
        fig = plot_qa_accuracy_by_regime(regime_data, title=f"QA Accuracy by Regime — {label}")
        save_figure(fig, str(fig_dir / "accuracy_by_regime.png"), formats=("png",))

        dist_data = {
            label: {
                tid: ts.get("scores", [])
                for tid, ts in template_summaries.items()
            }
        }
        fig = plot_qa_score_distribution(dist_data, title=f"QA Score Distribution — {label}")
        save_figure(fig, str(fig_dir / "score_distribution.png"), formats=("png",))
