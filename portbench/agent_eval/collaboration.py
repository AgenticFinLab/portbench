"""Explicit multi-agent collaboration for portfolio weight optimization."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from portbench.agent_eval.base import MarketSnapshot, S2Output, S3Output
from portbench.agent_eval.canonical import canonical_json
from portbench.agent_eval.stages import S3WeightOptimization, _call_with_json_retry


def _require_fields(payload: Dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    """Reject incomplete collaboration responses instead of silently degrading."""
    # A missing critique field changes protocol semantics, so it is not repaired implicitly.
    missing = [field for field in fields if field not in payload]
    if missing:
        raise RuntimeError(f"{label} response is missing required fields: {missing}")


def _normalized_s3(payload: Dict[str, Any], assets: list[str], raw: str) -> S3Output:
    """Validate and normalize the optimizer revision into the S3 contract."""
    raw_weights = payload.get("weights")
    if not isinstance(raw_weights, dict):
        raise RuntimeError("optimizer revision must contain a weights object")
    weights: Dict[str, float] = {}
    # Restrict the revision to the declared S2 asset universe.
    for asset in assets:
        try:
            weights[asset] = max(0.0, float(raw_weights.get(asset, 0.0)))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid revised weight for {asset}") from exc
    total = sum(weights.values())
    if total <= 0.0:
        raise RuntimeError("optimizer revision returned no positive weights")
    # Normalize once after validation to preserve a long-only fully invested contract.
    weights = {asset: round(weight / total, 6) for asset, weight in weights.items()}

    def number(name: str) -> float:
        try:
            return float(payload.get(name, 0.0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"optimizer revision has invalid {name}") from exc

    return S3Output(
        weights=weights,
        expected_return=number("expected_return"),
        expected_vol=number("expected_vol"),
        sharpe_estimate=number("sharpe_estimate"),
        raw_llm_output=raw,
    )


class CollaborativeS3WeightOptimization(S3WeightOptimization):
    """Run proposal, two independent critiques, and an optimizer revision."""

    def __init__(self, runtime: Any, *, oracle_mode: str = "ex_post") -> None:
        super().__init__(
            runtime.stage_adapter("S3"),
            use_tools=runtime.spec.tools_enabled,
            oracle_mode=oracle_mode,
        )
        self.runtime = runtime
        # Bind each specialist call to a distinct node and stable logical call identifier.
        self.risk_adapter = runtime.collaboration_adapter(
            call_id="S3-risk-critique",
            agent_id="risk",
            round_id="critique",
        )
        self.execution_adapter = runtime.collaboration_adapter(
            call_id="S3-execution-critique",
            agent_id="executor",
            round_id="critique",
        )
        self.revision_adapter = runtime.collaboration_adapter(
            call_id="S3-optimizer-revision",
            agent_id="optimizer",
            round_id="revision",
        )

    def _critique(
        self,
        *,
        adapter: Any,
        snapshot: MarketSnapshot,
        prompt: str,
        stage_name: str,
        required: tuple[str, ...],
    ) -> tuple[Dict[str, Any], str]:
        """Call one specialist and enforce its declared response contract."""
        parsed, raw = _call_with_json_retry(
            adapter,
            prompt,
            self.runtime.spec.tools_enabled,
            stage_name,
            snapshot,
        )
        _require_fields(parsed, required, stage_name)
        return parsed, raw

    def run(self, snapshot: MarketSnapshot, prior_output: S2Output = None) -> S3Output:
        """Execute the four-call S3 collaboration protocol."""
        if prior_output is None:
            raise ValueError("collaborative S3 requires S2 signals")

        # The ordinary S3 optimizer produces the candidate before any specialist sees it.
        proposal = super().run(snapshot, prior_output)
        if proposal.refused:
            raise RuntimeError("optimizer proposal was refused")
        proposal_payload = asdict(proposal)
        # Risk and execution agents receive separate copies through explicit message edges.
        self.runtime.publish(
            sender="optimizer",
            recipient="risk",
            stage_id="S3",
            round_id="proposal",
            message_type="candidate_weights",
            payload=proposal_payload,
        )
        self.runtime.publish(
            sender="optimizer",
            recipient="executor",
            stage_id="S3",
            round_id="proposal",
            message_type="candidate_weights",
            payload=proposal_payload,
        )

        # The risk critic focuses on portfolio constraints and lookback risk evidence.
        risk_prompt = (
            "[COLLABORATION TASK: RISK CRITIQUE]\n"
            "Audit the candidate portfolio using only the supplied Point-in-Time evidence and tools. "
            "Return JSON with risk_assessment (string), constraint_violations (array), and "
            "recommended_changes (object mapping assets to suggested weights or deltas).\n"
            f"Investor profile: {self.runtime.profile}\n"
            f"Candidate: {canonical_json(proposal_payload)}"
        )
        risk, risk_raw = self._critique(
            adapter=self.risk_adapter,
            snapshot=snapshot,
            prompt=risk_prompt,
            stage_name="S3-risk-critique",
            required=("risk_assessment", "constraint_violations", "recommended_changes"),
        )
        self.runtime.publish(
            sender="risk",
            recipient="optimizer",
            stage_id="S3",
            round_id="critique",
            message_type="risk_critique",
            payload=risk,
        )

        # The execution critic evaluates turnover and implementability independently.
        execution_prompt = (
            "[COLLABORATION TASK: EXECUTION CRITIQUE]\n"
            "Audit the candidate portfolio for turnover, transaction cost, and implementability using "
            "only current holdings and Point-in-Time tools. Return JSON with execution_assessment "
            "(string), turnover_concerns (array), and recommended_changes (object).\n"
            f"Candidate: {canonical_json(proposal_payload)}"
        )
        execution, execution_raw = self._critique(
            adapter=self.execution_adapter,
            snapshot=snapshot,
            prompt=execution_prompt,
            stage_name="S3-execution-critique",
            required=("execution_assessment", "turnover_concerns", "recommended_changes"),
        )
        self.runtime.publish(
            sender="executor",
            recipient="optimizer",
            stage_id="S3",
            round_id="critique",
            message_type="execution_critique",
            payload=execution,
        )

        assets = list(prior_output.signals)
        # The revision prompt contains both critiques so neither specialist is advisory-only.
        revision_prompt = (
            "[COLLABORATION TASK: OPTIMIZER REVISION]\n"
            "Revise the candidate after considering both specialist critiques. Return JSON with weights "
            "for every asset, expected_return, expected_vol, sharpe_estimate, and revision_rationale. "
            "Weights must be non-negative and sum to one.\n"
            f"Signals: {canonical_json(asdict(prior_output))}\n"
            f"Candidate: {canonical_json(proposal_payload)}\n"
            f"Risk critique: {canonical_json(risk)}\n"
            f"Execution critique: {canonical_json(execution)}"
        )
        revised_payload, revision_raw = _call_with_json_retry(
            self.revision_adapter,
            revision_prompt,
            self.runtime.spec.tools_enabled,
            "S3-optimizer-revision",
            snapshot,
        )
        _require_fields(
            revised_payload,
            ("weights", "expected_return", "expected_vol", "sharpe_estimate"),
            "S3-optimizer-revision",
        )
        revised = _normalized_s3(revised_payload, assets, revision_raw)
        # Persist parsed and raw collaboration evidence with the final S3 decision.
        revised.collaboration = {
            "proposal": proposal_payload,
            "risk_critique": risk,
            "execution_critique": execution,
            "revision": revised_payload,
            "raw_responses": {
                "risk_critique": risk_raw,
                "execution_critique": execution_raw,
            },
        }
        self._last_prompt = revision_prompt
        return revised


__all__ = ["CollaborativeS3WeightOptimization"]
