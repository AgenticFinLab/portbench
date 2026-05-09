"""
Smoke-test every configured LLM provider by sending a minimal chat request.

Run:
    .venv/Scripts/python.exe examples/experiments/test_llm_call.py
    .venv/Scripts/python.exe examples/experiments/test_llm_call.py --providers tencent ark

Exit code: 0 if all tested providers responded, 1 if any failed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# Add repo root to path when run directly
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from portbench.experiments.providers import PROVIDER_REGISTRY, build_adapter

PROMPT = "Reply with exactly one sentence: confirm you are ready for portfolio management tasks."

@dataclass
class Result:
    provider: str
    model: str
    ok: bool
    reply: str
    latency: float
    error: str = ""


def test_provider(provider_key: str) -> Result:
    spec = PROVIDER_REGISTRY[provider_key]
    api_key = os.environ.get(f"{spec.env_prefix}_API_KEY", "")
    model = os.environ.get(f"{spec.env_prefix}_MODEL", "")

    if not api_key:
        return Result(provider_key, model or "?", False, "", 0.0,
                      f"{spec.env_prefix}_API_KEY not set")
    if not model:
        return Result(provider_key, "?", False, "", 0.0,
                      f"{spec.env_prefix}_MODEL not set")

    t0 = time.time()
    try:
        adapter = build_adapter(provider_key)
        reply = adapter.complete(PROMPT)
        latency = time.time() - t0
        ok = bool(reply and reply.strip())
        return Result(provider_key, model, ok, (reply or "").strip()[:120], latency,
                      "" if ok else "empty response")
    except Exception as exc:
        return Result(provider_key, model, False, "", time.time() - t0, str(exc)[:200])


def main() -> int:
    p = argparse.ArgumentParser(description="Test LLM provider connectivity")
    p.add_argument(
        "--providers", nargs="+", default=None,
        help="Provider keys to test (default: all with API key set)"
    )
    args = p.parse_args()

    # Determine which providers to test
    if args.providers:
        keys = [k for k in args.providers if k in PROVIDER_REGISTRY]
        unknown = [k for k in args.providers if k not in PROVIDER_REGISTRY]
        if unknown:
            print(f"[WARN] Unknown providers skipped: {unknown}")
    else:
        keys = [
            k for k, spec in PROVIDER_REGISTRY.items()
            if os.environ.get(f"{spec.env_prefix}_API_KEY")
            and os.environ.get(f"{spec.env_prefix}_MODEL", "").lower() not in ("", "xxx")
        ]

    if not keys:
        print("No providers configured. Set API keys in .env and retry.")
        return 1

    print(f"Testing {len(keys)} provider(s): {', '.join(keys)}\n")
    print(f"Prompt: \"{PROMPT}\"\n")
    print("-" * 72)

    results: list[Result] = []
    for key in keys:
        print(f"  {key:<12s} ", end="", flush=True)
        r = test_provider(key)
        results.append(r)
        if r.ok:
            print(f"OK  [{r.latency:.1f}s]  model={r.model}")
            print(f"               > {r.reply}")
        else:
            print(f"FAIL  model={r.model}")
            print(f"               x {r.error}")
        print()

    print("-" * 72)
    n_ok = sum(1 for r in results if r.ok)
    n_fail = len(results) - n_ok
    print(f"Result: {n_ok}/{len(results)} passed", end="")
    if n_fail:
        failed = [r.provider for r in results if not r.ok]
        print(f"  |  FAILED: {', '.join(failed)}", end="")
    print()

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
