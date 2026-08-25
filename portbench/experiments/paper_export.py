"""Export manuscript-ready values only from canonical frozen analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _read_json(path: str | Path) -> dict[str, Any]:
    """Read one canonical analysis JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export_paper_artifacts(
    *,
    causal_summary_path: str | Path,
    qa_summary_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Write LaTeX table fragments and a numeric manifest from frozen results."""
    causal = _read_json(causal_summary_path)
    qa = _read_json(qa_summary_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"causal": causal, "qa_constraint_v2": qa}
    manifest = destination / "paper_numbers.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latex_break = chr(92) * 2

    causal_rows = []
    for stage, values in causal.get("delta_ceps", {}).items():
        effect = values.get("effect")
        low, high = values.get("ci95", [None, None])
        if effect is not None:
            causal_rows.append(
                f"{stage} & {effect:.4f} & [{low:.4f}, {high:.4f}] & "
                f"{values.get('n_raw', 0)} {latex_break}"
            )
    (destination / "causal_delta_ceps_rows.tex").write_text(
        "\n".join(causal_rows) + ("\n" if causal_rows else ""), encoding="utf-8"
    )

    qa_rows = []
    for model, templates in qa.get("model_template", {}).items():
        for template in ("T3", "T4"):
            values = templates.get(template, {})
            mean = values.get("mean")
            low, high = values.get("ci95", [None, None])
            if mean is not None:
                qa_rows.append(
                    f"{model} & {template} & {mean:.4f} & [{low:.4f}, {high:.4f}] & "
                    f"{values.get('n', 0)} {latex_break}"
                )
    (destination / "qa_constraint_v2_rows.tex").write_text(
        "\n".join(qa_rows) + ("\n" if qa_rows else ""), encoding="utf-8"
    )
    return manifest


__all__ = ["export_paper_artifacts"]
