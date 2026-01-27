"""ReAct GodClass: controller + critique + memory + tool router in one."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from crca_core.core.api import (
    EstimatorConfig,
    FeasibilityConstraints,
    PCMCIConfig,
    TabularDiscoveryConfig,
    TargetQuery,
    design_intervention,
    discover_tabular,
    discover_timeseries_pcmci,
    identify_effect,
    estimate_effect_dowhy,
    simulate_counterfactual,
)
from crca_core.models.refusal import RefusalChecklistItem, RefusalReasonCode, RefusalResult
from crca_core.models.result import AnyResult, IdentificationResult
from crca_reasoning.types import (
    LRMPlanResult,
    RationaleTrace,
    ReActAction,
    ReActCycleTrace,
    ReActObservation,
)


ReasonerFn = Callable[["ReActGodClass"], tuple[str, List[ReActAction], str]]


@dataclass
class ReActGodClass:
    """All-in-one ReAct controller with strict gating and critique."""

    max_cycles: int = 3
    memory_store: List[Dict[str, Any]] = field(default_factory=list)
    staged_store: List[Dict[str, Any]] = field(default_factory=list)
    last_identification: Optional[IdentificationResult] = None

    # -----------------
    # Memory operations
    # -----------------
    def read_memory(self) -> List[Dict[str, Any]]:
        return list(self.memory_store)

    def stage_write(self, item: Dict[str, Any]) -> None:
        self.staged_store.append(item)

    def finalize_memory(self) -> None:
        self.memory_store.extend(self.staged_store)
        self.staged_store = []

    # -----------------
    # Critique mechanics
    # -----------------
    def critique_cycle(self, trace: ReActCycleTrace) -> Dict[str, Any]:
        issues: List[Dict[str, Any]] = []
        score = 0.0
        threshold = 1.0
        if trace.actions and not trace.observations:
            issues.append(
                {
                    "code": "NO_OBSERVATIONS",
                    "message": "Actions were proposed but no observations recorded.",
                    "severity": "error",
                }
            )
            score += 1.0
        if any(obs.refusal is not None for obs in trace.observations):
            issues.append(
                {
                    "code": "REFUSAL_OCCURRED",
                    "message": "At least one tool call resulted in refusal.",
                    "severity": "warning",
                }
            )
            score += 0.5
        return {
            "issues": issues,
            "score": score,
            "threshold": threshold,
            "should_revise": score >= threshold,
        }

    # -----------------
    # Tool routing
    # -----------------
    def call_tool(self, *, tool_name: str, payload: Dict[str, Any]) -> AnyResult | RefusalResult:
        if tool_name in {"identify", "estimate", "counterfactual", "design_intervention"}:
            if payload.get("locked_spec") is None:
                return RefusalResult(
                    message="LockedSpec required for this tool.",
                    reason_codes=[RefusalReasonCode.SPEC_NOT_LOCKED],
                    checklist=[
                        RefusalChecklistItem(item="Provide LockedSpec", rationale="Tool is gated.")
                    ],
                    suggested_next_steps=["Lock a spec and retry."],
                )

        if tool_name == "identify":
            res = identify_effect(
                locked_spec=payload["locked_spec"],
                treatment=payload.get("treatment", ""),
                outcome=payload.get("outcome", ""),
            )
            if isinstance(res, IdentificationResult):
                self.last_identification = res
            return res

        if tool_name == "estimate":
            ident = payload.get("identification_result") or self.last_identification
            return estimate_effect_dowhy(
                data=payload.get("data"),
                locked_spec=payload["locked_spec"],
                treatment=payload.get("treatment", ""),
                outcome=payload.get("outcome", ""),
                identification_result=ident,
                config=payload.get("config", EstimatorConfig()),
            )

        if tool_name == "counterfactual":
            return simulate_counterfactual(
                locked_spec=payload["locked_spec"],
                factual_observation=payload.get("factual_observation", {}),
                intervention=payload.get("intervention", {}),
                allow_partial_observation=payload.get("allow_partial_observation", False),
            )

        if tool_name == "design_intervention":
            return design_intervention(
                locked_spec=payload["locked_spec"],
                target_query=payload.get("target_query", TargetQuery()),
                constraints=payload.get("constraints", FeasibilityConstraints()),
            )

        if tool_name == "discover_tabular":
            return discover_tabular(
                payload.get("data"),
                payload.get("config", TabularDiscoveryConfig()),
                payload.get("assumptions"),
            )

        if tool_name == "discover_timeseries":
            return discover_timeseries_pcmci(
                payload.get("data"),
                payload.get("config", PCMCIConfig()),
                payload.get("assumptions"),
            )

        return RefusalResult(
            message=f"Unknown tool: {tool_name}",
            reason_codes=[RefusalReasonCode.UNSUPPORTED_OPERATION],
            checklist=[RefusalChecklistItem(item="Use a supported tool", rationale="Unknown tool name.")],
        )

    # -----------------
    # ReAct execution
    # -----------------
    def run(self, reasoner: ReasonerFn) -> LRMPlanResult:
        cycle_traces: List[ReActCycleTrace] = []
        core_results: List[AnyResult] = []
        refusals: List[RefusalResult] = []
        rationale = RationaleTrace()

        for _ in range(self.max_cycles):
            reasoning, actions, rationale_step = reasoner(self)
            cycle = ReActCycleTrace(reasoning=reasoning, actions=actions, observations=[])
            if rationale_step:
                rationale.steps.append(rationale_step)

            for action in actions:
                result = self.call_tool(tool_name=action.tool_name, payload=action.payload)
                if isinstance(result, RefusalResult):
                    obs = ReActObservation(tool_name=action.tool_name, refusal=result)
                    refusals.append(result)
                else:
                    obs = ReActObservation(tool_name=action.tool_name, result=result)
                    core_results.append(result)
                cycle.observations.append(obs)

            critique = self.critique_cycle(cycle)
            cycle.critique = critique
            cycle_traces.append(cycle)

            if critique.get("should_revise"):
                continue
            break

        for res in core_results:
            try:
                self.stage_write(res.model_dump())
            except Exception:
                continue
        self.finalize_memory()

        return LRMPlanResult(
            cycle_traces=cycle_traces,
            rationale_trace=rationale,
            core_results=core_results,
            refusals=refusals,
            finalized=True,
        )
