"""LLM tool orchestration (DraftSpec-only + gated core calls)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from crca_core.core.api import (
    EstimatorConfig,
    FeasibilityConstraints,
    PCMCIConfig,
    TabularDiscoveryConfig,
    TargetQuery,
    discover_tabular,
    discover_timeseries_pcmci,
    design_intervention,
    identify_effect,
    estimate_effect_dowhy,
    simulate_counterfactual,
)
from crca_core.models.refusal import RefusalChecklistItem, RefusalReasonCode, RefusalResult
from crca_core.models.result import AnyResult, IdentificationResult
from crca_core.models.spec import (
    AssumptionItem,
    AssumptionSpec,
    AssumptionStatus,
    CausalGraphSpec,
    DataColumnSpec,
    DataSpec,
    DraftSpec,
    EdgeSpec,
    LockedSpec,
    NodeSpec,
    RoleSpec,
)
from crca_llm.client import MissingApiKeyError, OpenAIClient
from crca_llm.coauthor import DraftBundle
from crca_llm.types import LLMRunResult


def _fallback_bundle(user_text: str, observed_columns: Optional[Sequence[str]]) -> DraftBundle:
    cols = list(observed_columns or [])
    data_spec = DataSpec(
        columns=[DataColumnSpec(name=c, dtype="unknown") for c in cols],
        measurement_error_notes="unknown",
    )
    graph = CausalGraphSpec(nodes=[NodeSpec(name=c) for c in cols], edges=[])
    assumptions = AssumptionSpec(
        items=[
            AssumptionItem(
                name="DRAFT_ONLY",
                status=AssumptionStatus.unknown,
                description="Fallback draft used due to LLM parse failure.",
            )
        ],
        falsification_plan=["Collect domain constraints/time ordering and re-run drafting."],
    )
    drafts = [
        DraftSpec(
            data=data_spec,
            graph=graph,
            roles=RoleSpec(),
            assumptions=assumptions,
            draft_notes=f"Fallback draft from observed columns. User text: {user_text[:200]}",
        )
    ]
    return DraftBundle(drafts=drafts, review_checklist=[])


def _parse_llm_drafts(payload: dict) -> DraftBundle:
    drafts_payload = payload.get("drafts", [])
    drafts: List[DraftSpec] = []
    for item in drafts_payload:
        nodes = [NodeSpec(name=n) for n in item.get("nodes", [])]
        edges = [EdgeSpec(source=a, target=b) for a, b in item.get("edges", [])]
        graph = CausalGraphSpec(nodes=nodes, edges=edges)
        roles = RoleSpec(
            treatments=item.get("treatments", []),
            outcomes=item.get("outcomes", []),
            mediators=item.get("mediators", []),
            instruments=item.get("instruments", []),
            adjustment_candidates=item.get("adjustment_candidates", []),
            prohibited_controls=item.get("prohibited_controls", []),
        )
        data_spec = DataSpec(
            columns=[DataColumnSpec(name=c, dtype="unknown") for c in item.get("columns", [])],
            measurement_error_notes="unknown",
        )
        assumptions = AssumptionSpec(items=[], falsification_plan=[])
        drafts.append(DraftSpec(data=data_spec, graph=graph, roles=roles, assumptions=assumptions))
    checklist = payload.get("review_checklist", [])
    return DraftBundle(drafts=drafts, review_checklist=checklist)


@dataclass
class LLMOrchestrator:
    client: Optional[OpenAIClient] = None
    default_model: str = "gpt-4o-mini"
    enable_audit_log: bool = True

    def _draft_with_llm(self, user_text: str, observed_columns: Optional[Sequence[str]]) -> DraftBundle:
        client = self.client or OpenAIClient.from_env()
        client.enable_audit_log = self.enable_audit_log
        prompt = {
            "role": "user",
            "content": (
                "You are drafting causal specs. Return JSON with keys: "
                "`drafts` (list), `review_checklist` (list). Each draft has: "
                "`nodes` (list of strings), `edges` (list of [source,target]), "
                "`treatments`, `outcomes`, `mediators`, `instruments`, "
                "`adjustment_candidates`, `prohibited_controls`, `columns`.\n"
                f"User text: {user_text}\n"
                f"Observed columns: {list(observed_columns or [])}\n"
                "Return JSON only."
            ),
        }
        try:
            content = client.chat_completion(
                messages=[prompt],
                model=self.default_model,
                response_format={"type": "json_object"},
            )
        except MissingApiKeyError as exc:
            raise

        try:
            payload = json.loads(content)
            return _parse_llm_drafts(payload)
        except Exception:
            return _fallback_bundle(user_text, observed_columns)

    def run(
        self,
        *,
        user_text: str,
        observed_columns: Optional[Sequence[str]] = None,
        locked_spec: Optional[LockedSpec] = None,
        data: Optional[Any] = None,
        actions: Optional[Sequence[str]] = None,
        target_query: Optional[TargetQuery] = None,
        constraints: Optional[FeasibilityConstraints] = None,
        factual_observation: Optional[Dict[str, float]] = None,
        intervention: Optional[Dict[str, float]] = None,
    ) -> LLMRunResult:
        actions = list(actions or [])
        refusals: List[RefusalResult] = []
        core_results: List[AnyResult] = []

        # Drafting is mandatory and requires API key.
        try:
            draft_bundle = self._draft_with_llm(user_text, observed_columns)
        except MissingApiKeyError as exc:
            refusal = RefusalResult(
                message=str(exc),
                reason_codes=[RefusalReasonCode.INPUT_INVALID],
                checklist=[
                    RefusalChecklistItem(
                        item="Set OPENAI_API_KEY",
                        rationale="LLM orchestration requires an API key.",
                    )
                ],
                suggested_next_steps=["Export OPENAI_API_KEY and retry."],
            )
            return LLMRunResult(draft_bundle=DraftBundle(drafts=[], review_checklist=[]), refusals=[refusal])

        if actions and locked_spec is None:
            refusal = RefusalResult(
                message="LockedSpec is required to run core tools.",
                reason_codes=[RefusalReasonCode.SPEC_NOT_LOCKED],
                checklist=[RefusalChecklistItem(item="Provide LockedSpec", rationale="Core tools are gated.")],
                suggested_next_steps=["Lock a spec first, then re-run with locked_spec."],
            )
            refusals.append(refusal)
            return LLMRunResult(draft_bundle=draft_bundle, refusals=refusals)

        identification_result: IdentificationResult | None = None

        for action in actions:
            if action == "discover_tabular":
                if data is None:
                    refusals.append(
                        RefusalResult(
                            message="Tabular discovery requires data.",
                            reason_codes=[RefusalReasonCode.INPUT_INVALID],
                            checklist=[RefusalChecklistItem(item="Provide data", rationale="Missing DataFrame.")],
                        )
                    )
                    continue
                result = discover_tabular(data, TabularDiscoveryConfig())
                core_results.append(result)
                if isinstance(result, RefusalResult):
                    refusals.append(result)
            elif action == "discover_timeseries":
                if data is None:
                    refusals.append(
                        RefusalResult(
                            message="Time-series discovery requires data.",
                            reason_codes=[RefusalReasonCode.INPUT_INVALID],
                            checklist=[RefusalChecklistItem(item="Provide data", rationale="Missing DataFrame.")],
                        )
                    )
                    continue
                result = discover_timeseries_pcmci(data, PCMCIConfig())
                core_results.append(result)
                if isinstance(result, RefusalResult):
                    refusals.append(result)
            elif action == "design_intervention":
                if target_query is None:
                    refusals.append(
                        RefusalResult(
                            message="Intervention design requires TargetQuery.",
                            reason_codes=[RefusalReasonCode.INPUT_INVALID],
                            checklist=[RefusalChecklistItem(item="Provide TargetQuery", rationale="Missing target query.")],
                        )
                    )
                    continue
                result = design_intervention(
                    locked_spec=locked_spec,
                    target_query=target_query,
                    constraints=constraints or FeasibilityConstraints(),
                )
                core_results.append(result)
            elif action == "identify":
                result = identify_effect(
                    locked_spec=locked_spec,
                    treatment=locked_spec.roles.treatments[0] if locked_spec.roles.treatments else "",
                    outcome=locked_spec.roles.outcomes[0] if locked_spec.roles.outcomes else "",
                )
                core_results.append(result)
                if isinstance(result, RefusalResult):
                    refusals.append(result)
                else:
                    identification_result = result
            elif action == "estimate":
                if data is None:
                    refusals.append(
                        RefusalResult(
                            message="Estimation requires data.",
                            reason_codes=[RefusalReasonCode.INPUT_INVALID],
                            checklist=[RefusalChecklistItem(item="Provide data", rationale="Missing DataFrame.")],
                        )
                    )
                    continue
                result = estimate_effect_dowhy(
                    data=data,
                    locked_spec=locked_spec,
                    treatment=locked_spec.roles.treatments[0] if locked_spec.roles.treatments else "",
                    outcome=locked_spec.roles.outcomes[0] if locked_spec.roles.outcomes else "",
                    identification_result=identification_result,
                    config=EstimatorConfig(),
                )
                core_results.append(result)
                if isinstance(result, RefusalResult):
                    refusals.append(result)
            elif action == "counterfactual":
                if factual_observation is None or intervention is None:
                    refusals.append(
                        RefusalResult(
                            message="Counterfactual requires factual_observation and intervention.",
                            reason_codes=[RefusalReasonCode.INPUT_INVALID],
                            checklist=[
                                RefusalChecklistItem(item="Provide factual_observation", rationale="Missing factual data."),
                                RefusalChecklistItem(item="Provide intervention", rationale="Missing intervention."),
                            ],
                        )
                    )
                    continue
                result = simulate_counterfactual(
                    locked_spec=locked_spec,
                    factual_observation=factual_observation,
                    intervention=intervention,
                )
                core_results.append(result)
                if isinstance(result, RefusalResult):
                    refusals.append(result)

        return LLMRunResult(
            draft_bundle=draft_bundle,
            core_results=core_results,
            refusals=refusals,
        )

