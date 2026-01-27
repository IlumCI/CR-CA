"""Types for ReAct reasoning and outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from crca_core.models.refusal import RefusalResult
from crca_core.models.result import AnyResult


class ReActAction(BaseModel):
    tool_name: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class ReActObservation(BaseModel):
    tool_name: str
    result: Optional[AnyResult] = None
    refusal: Optional[RefusalResult] = None


class ReActCycleTrace(BaseModel):
    reasoning: str
    actions: List[ReActAction] = Field(default_factory=list)
    observations: List[ReActObservation] = Field(default_factory=list)
    critique: Optional[Dict[str, Any]] = None


class RationaleTrace(BaseModel):
    steps: List[str] = Field(default_factory=list)


class LRMPlanResult(BaseModel):
    cycle_traces: List[ReActCycleTrace] = Field(default_factory=list)
    rationale_trace: Optional[RationaleTrace] = None
    core_results: List[AnyResult] = Field(default_factory=list)
    refusals: List[RefusalResult] = Field(default_factory=list)
    finalized: bool = False
