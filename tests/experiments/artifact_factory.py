"""Build temporary legacy experiment artifacts for tests."""

from __future__ import annotations

import json
from pathlib import Path


def write_legacy_episode(root: Path) -> Path:
    """Write one complete episode and return its stage-file directory."""
    episode_dir = root / "pipeline_logs" / "ep_toy"
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Use the smallest payload that satisfies every legacy stage contract.
    stage_payloads = {
        "S1": {"asset_views": {"A": 0.5}},
        "S2": {"signals": {"A": "buy"}},
        "S3": {"weights": {"A": 0.6, "B": 0.4}},
        "S4": {"executed_weights": {"A": 0.6, "B": 0.4}},
        "S5": {"portfolio_var": -0.01, "portfolio_drawdown": -0.02},
    }
    stages = []
    for stage_id, parsed_output in stage_payloads.items():
        stages.append(
            {
                "stage_id": stage_id,
                "prompt": f"{stage_id} prompt",
                "raw_response": json.dumps(parsed_output),
                "parsed_output": parsed_output,
                "score": 1.0,
            }
        )

    # Keep a full episode log for the legacy cache scanner.
    episode_path = episode_dir / "ep_toy.json"
    episode_path.write_text(json.dumps({"stages": stages}), encoding="utf-8")

    # Keep standalone S3 and current weights for the step-replay entry point.
    (episode_dir / "S3.json").write_text(
        json.dumps({"target_weights": {"A": 0.6, "B": 0.4}}),
        encoding="utf-8",
    )
    (episode_dir / "current_weights.json").write_text(
        json.dumps({"A": 0.5, "B": 0.5}),
        encoding="utf-8",
    )
    return episode_dir
