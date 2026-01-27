"""Unified tool router with strict gating."""

from __future__ import annotations

from typing import Any, Dict, Optional

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
from crca_core.models.spec import LockedSpec


class ToolRouter:
    """Routes tool calls with pre-Act gating."""

    def __init__(self) -> None:
        self.last_identification: Optional[IdentificationResult] = None

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
