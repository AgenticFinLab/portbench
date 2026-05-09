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

from ..experiments.config import ExperimentConfig, ModelSpec
from ..experiments.providers import build_adapter, model_label as _model_label
from . import paths as qpaths
from .scorer import score_response


def _spec_label(spec: ModelSpec) -> str:
    kind = spec.kind()
    if kind == "baseline":
        return f"baseline-{spec.baseline}"
    if kind == "mock":
        return "mock"
    return _model_label(spec.provider, spec.model)


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


def _build_eval_prompt(pair: dict) -> str:
    context = pair.get("context_summary", "")
    question = pair.get("question", "")
    return (
        "[PORTFOLIO MANAGEMENT QA]\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Answer directly and concisely\n"
        "- For numeric answers, provide a single number\n"
        "- For direction answers, reply with exactly one word: positive, negative, or flat\n"
        "- For allocation answers, provide weights as percentages summing to 100%\n"
        "- For rebalancing decisions, reply with: rebalance or hold\n\n"
        "Answer:"
    )


class QAEvaluator:
    def __init__(self, cfg: ExperimentConfig, raw_yaml: Optional[str] = None):
        self.cfg = cfg
        self._raw_yaml = raw_yaml

        all_pairs = _load_qa_pairs(cfg.qa.dataset_path, cfg.qa.split)

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

        # Filter models to LLM-only (skip baseline / mock)
        self._llm_specs = [s for s in cfg.models if s.kind() == "llm"]

    def dry_run(self) -> list[dict]:
        out = []
        for spec in self._llm_specs:
            label = _spec_label(spec)
            for tid, pairs in sorted(self._pairs_by_template.items()):
                out.append({
                    "model": label,
                    "template": tid,
                    "n_questions": len(pairs),
                })
        return out

    def run(self) -> dict:
        cfg = self.cfg
        root = qpaths.qa_root(cfg.output_root, cfg.batch_id)
        root.mkdir(parents=True, exist_ok=True)

        model_summaries: dict[str, dict] = {}
        t0 = time.time()

        n_total = sum(
            len(pairs)
            for pairs in self._pairs_by_template.values()
        ) * len(self._llm_specs)

        pbar = tqdm(total=n_total, desc="qa_eval", unit="q", dynamic_ncols=True)

        for spec in self._llm_specs:
            label = _spec_label(spec)
            try:
                adapter = build_adapter(spec.provider, spec.model)
            except Exception as exc:
                print(f"[QA] adapter build failed for {label}: {exc}")
                pbar.update(sum(len(p) for p in self._pairs_by_template.values()))
                continue

            model_summary = self._run_model_qa(
                spec, adapter, label, pbar,
            )
            model_summaries[label] = model_summary

        pbar.close()

        summary = {
            "batch_id": cfg.batch_id,
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
        label: str,
        pbar,
    ) -> dict:
        cfg = self.cfg
        ckpt_path = qpaths.qa_checkpoint_file(cfg.output_root, cfg.batch_id, label)
        completed_keys = qpaths.load_checkpoint(ckpt_path) if cfg.resume else set()
        ckpt_lock = threading.Lock()

        template_summaries: dict[str, dict] = {}

        for tid in sorted(self._pairs_by_template.keys()):
            pairs = self._pairs_by_template[tid]
            t_summary = self._run_template(
                adapter, label, tid, pairs,
                completed_keys, ckpt_lock, ckpt_path, pbar,
            )
            template_summaries[tid] = t_summary

        # Write model-level summary
        all_scores = []
        for ts in template_summaries.values():
            all_scores.extend(ts.get("scores", []))

        model_summary = {
            "model": label,
            "mean_accuracy": round(float(np.mean(all_scores)), 4) if all_scores else 0.0,
            "n_total": len(all_scores),
            "n_correct": sum(1 for s in all_scores if s >= 0.99),
            "templates": {
                tid: {k: v for k, v in ts.items() if k != "scores"}
                for tid, ts in template_summaries.items()
            },
        }

        m_dir = qpaths.qa_model_dir(cfg.output_root, cfg.batch_id, label)
        m_dir.mkdir(parents=True, exist_ok=True)
        (m_dir / "qa_model_summary.json").write_text(
            json.dumps(model_summary, indent=2, default=str), encoding="utf-8"
        )

        # Render figures
        if cfg.logging.save_figures:
            try:
                self._render_model_figures(label, template_summaries)
            except Exception:
                pass

        return model_summary

    def _run_template(
        self,
        adapter,
        label: str,
        template_id: str,
        pairs: list[dict],
        completed_keys: set[str],
        ckpt_lock: threading.Lock,
        ckpt_path: Path,
        pbar,
    ) -> dict:
        cfg = self.cfg
        t_dir = qpaths.qa_template_dir(
            cfg.output_root, cfg.batch_id, label, template_id
        )
        t_dir.mkdir(parents=True, exist_ok=True)

        scores: list[float] = []
        by_regime: dict[str, list[float]] = {}
        results_lock = threading.Lock()

        def _eval_one(pair: dict) -> None:
            qa_id = pair.get("qa_id", pair.get("id", ""))
            t_id = pair.get("template_id", pair.get("template", template_id))
            ck = f"{template_id}:{qa_id}"

            if ck in completed_keys:
                pbar.update(1)
                return

            t0 = time.time()
            try:
                prompt = _build_eval_prompt(pair)
                response = adapter.complete(prompt)
                latency = time.time() - t0

                sc = score_response(
                    template_id=template_id,
                    gt_answer=pair.get("answer", ""),
                    llm_response=response or "",
                    answer_numeric=pair.get("answer_numeric"),
                    assets=pair.get("assets"),
                )

                record = {
                    "qa_id": qa_id,
                    "template_id": template_id,
                    "score": round(sc, 4),
                    "response": (response or "")[:500] if cfg.qa.save_responses else "",
                    "latency": round(latency, 2),
                    "regime": pair.get("market_regime", ""),
                    "complexity": pair.get("complexity", ""),
                    "split": pair.get("split", ""),
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

            with results_lock:
                qpaths.append_result(t_dir, record)
                scores.append(sc)
                regime = pair.get("market_regime", "unknown")
                by_regime.setdefault(regime, []).append(sc)

            with ckpt_lock:
                completed_keys.add(ck)
                qpaths.write_checkpoint(ckpt_path, completed_keys, self.cfg.batch_id)

            pbar.update(1)

        max_workers = max(1, cfg.qa.parallel_questions)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_eval_one, p): p for p in pairs}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception:
                    pass

        # Write template summary
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
        self, label: str, template_summaries: dict[str, dict]
    ) -> None:
        from ..visualization.qa_accuracy_plots import (
            plot_qa_accuracy_heatmap,
            plot_qa_accuracy_by_regime,
            plot_qa_score_distribution,
        )
        from ..visualization.style import save_figure

        cfg = self.cfg
        fig_dir = qpaths.qa_figures_dir(cfg.output_root, cfg.batch_id, label)
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Accuracy heatmap (single model — use model as the only row)
        acc_data = {
            label: {
                tid: ts["accuracy"]
                for tid, ts in template_summaries.items()
            }
        }
        fig = plot_qa_accuracy_heatmap(acc_data, title=f"QA Accuracy — {label}")
        save_figure(fig, str(fig_dir / "accuracy_by_template.png"), formats=("png",))

        # By regime
        regime_data = {
            label: {
                tid: ts.get("by_regime", {})
                for tid, ts in template_summaries.items()
            }
        }
        fig = plot_qa_accuracy_by_regime(regime_data, title=f"QA Accuracy by Regime — {label}")
        save_figure(fig, str(fig_dir / "accuracy_by_regime.png"), formats=("png",))

        # Score distribution
        dist_data = {
            label: {
                tid: ts.get("scores", [])
                for tid, ts in template_summaries.items()
            }
        }
        fig = plot_qa_score_distribution(dist_data, title=f"QA Score Distribution — {label}")
        save_figure(fig, str(fig_dir / "score_distribution.png"), formats=("png",))
