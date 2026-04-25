"""
Tier 1 QA evaluation: test LLM agent against the PortBench QA dataset.

Loads datasets/qa_dataset/test.jsonl, queries the model for each question,
and computes per-template accuracy using exact-match comparison.

Usage:
    python examples/agent_eval/run_qa_eval.py                          # mock agent
    python examples/agent_eval/run_qa_eval.py --model qwen:qwen-plus
    python examples/agent_eval/run_qa_eval.py --templates T1 T2 T5
    python examples/agent_eval/run_qa_eval.py --max-samples 50

Output:
    outputs/qa/{model_name}/{timestamp}/
        qa_results.json    — per-sample results + per-template accuracy
        summary.txt        — human-readable accuracy table
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.agent_eval.mock_agent import MockAgentAdapter
from portbench.agent_eval.llm_adapters import AnthropicAdapter, OpenAIAdapter, LiteLLMAdapter


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="PortBench QA dataset evaluation")
    parser.add_argument("--model", type=str, default=None,
                        help="'<provider>:<model_id>' e.g. 'qwen:qwen-plus'")
    parser.add_argument("--qa-path", type=str, default="datasets/qa_dataset/test.jsonl",
                        help="Path to QA dataset test split")
    parser.add_argument("--templates", nargs="+", default=None,
                        help="Filter to specific templates, e.g. T1 T2 T5")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples evaluated (for quick testing)")
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel threads for LLM calls (default: 8)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Adapter factory (mirrors run_evaluation.py)
# ---------------------------------------------------------------------------

_OPENAI_COMPAT = {
    "qwen":     ("DASHSCOPE_API_KEY",  "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "kimi":     ("MOONSHOT_API_KEY",   "https://api.moonshot.cn/v1"),
    "deepseek": ("DEEPSEEK_API_KEY",   "https://api.deepseek.com/v1"),
}


def _build_adapter(args):
    if args.model is None:
        return MockAgentAdapter(noise_level=args.noise, seed=args.seed)
    if ":" not in args.model:
        raise ValueError(f"--model must be '<provider>:<model_id>', got: {args.model!r}")
    provider, model = args.model.split(":", 1)
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicAdapter(model=model)
    elif provider == "openai":
        return OpenAIAdapter(model=model)
    elif provider in _OPENAI_COMPAT:
        key_env, base_url = _OPENAI_COMPAT[provider]
        if not os.environ.get(key_env):
            print(f"  WARNING: {key_env} is not set.")
        return OpenAIAdapter(model=model, base_url=base_url, api_key_env=key_env)
    else:
        return LiteLLMAdapter(model=f"{provider}/{model}")


# ---------------------------------------------------------------------------
# Exact-match comparison
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _extract_answer(model_answer: str) -> str:
    """Extract the answer string from a model response.

    Models return either plain text ("flat") or a JSON object with a key
    matching one of: prediction, answer, signal, decision, action, regime,
    rebalance_needed. Falls back to the full normalized string.
    """
    text = model_answer.strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            for key in ("prediction", "answer", "signal", "decision", "action", "regime", "rebalance_needed"):
                if key in obj:
                    return str(obj[key])
        except json.JSONDecodeError:
            # Try to extract the first quoted value after a known key
            m = re.search(r'"(?:prediction|answer|signal|decision|action|regime|rebalance_needed)"\s*:\s*"([^"]+)"', text)
            if m:
                return m.group(1)
    return text


def _is_correct(model_answer: str, ground_truth: str) -> bool:
    return _normalize(_extract_answer(model_answer)) == _normalize(ground_truth)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_qa_evaluation(args):
    qa_path = Path(args.qa_path)
    if not qa_path.exists():
        print(f"QA dataset not found: {qa_path}")
        print("  Run examples/qa_builder/build_qa_dataset.py first to generate it.")
        return

    # Load samples
    samples = []
    with open(qa_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.templates:
        samples = [s for s in samples if (s.get("template_id") or s.get("template", "")) in args.templates]
    if args.max_samples:
        samples = samples[:args.max_samples]

    if not samples:
        print("No samples found (check --templates filter).")
        return

    adapter = _build_adapter(args)
    model_name = adapter.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/qa/{model_name}/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"PortBench QA Evaluation — Model: {model_name}")
    print(f"Samples: {len(samples)}  Workers: {args.workers}")
    print("=" * 60)

    completed_count = [0]
    lock = threading.Lock()

    def _evaluate_sample(i_sample):
        i, sample = i_sample
        question = sample.get("question", "")
        ground_truth = str(sample.get("answer", sample.get("ground_truth", "")))
        template_id = sample.get("template_id") or sample.get("template", "unknown")
        try:
            model_answer = adapter.complete(question)
        except Exception as e:
            model_answer = f"ERROR: {e}"
        correct = _is_correct(model_answer, ground_truth)
        with lock:
            completed_count[0] += 1
            n = completed_count[0]
            if n % 10 == 0 or n == len(samples):
                print(f"  [{n}/{len(samples)}] template={template_id} correct={correct}")
        return {
            "sample_id": sample.get("id", i),
            "template_id": template_id,
            "question": question,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "correct": correct,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_evaluate_sample, (i, s)): i for i, s in enumerate(samples)}
        raw_results = [None] * len(samples)
        for future in as_completed(futures):
            idx = futures[future]
            raw_results[idx] = future.result()

    results = raw_results

    # Per-template accuracy
    from collections import defaultdict
    template_correct: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        template_correct[r["template_id"]].append(r["correct"])

    per_template = {
        tid: {
            "correct": sum(vs),
            "total": len(vs),
            "accuracy": round(sum(vs) / len(vs), 4),
        }
        for tid, vs in sorted(template_correct.items())
    }

    all_correct = sum(r["correct"] for r in results)
    overall_accuracy = round(all_correct / len(results), 4) if results else 0.0

    output = {
        "model_name": model_name,
        "timestamp": timestamp,
        "n_samples": len(results),
        "overall_accuracy": overall_accuracy,
        "per_template": per_template,
        "results": results,
    }

    with open(output_dir / "qa_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    lines = [
        f"QA Evaluation Summary",
        f"Model:    {model_name}",
        f"Samples:  {len(results)}",
        f"Overall:  {overall_accuracy:.1%}",
        "",
        "Per-template accuracy:",
    ]
    for tid, stats in per_template.items():
        lines.append(f"  {tid}: {stats['accuracy']:.1%}  ({stats['correct']}/{stats['total']})")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary + "\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_qa_evaluation(args)
