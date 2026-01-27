"""End-of-cycle critique schema and rules."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from crca_reasoning.types import ReActCycleTrace


class CritiqueIssue(BaseModel):
    code: str
    message: str
    severity: str = "warning"  # info|warning|error


class CritiqueResult(BaseModel):
    issues: List[CritiqueIssue] = Field(default_factory=list)
    score: float = 0.0
    threshold: float = 1.0
    should_revise: bool = False
    notes: Optional[str] = None


def critique_cycle(trace: ReActCycleTrace) -> CritiqueResult:
    issues: List[CritiqueIssue] = []
    score = 0.0
    threshold = 1.0
    if trace.actions and not trace.observations:
        issues.append(
            CritiqueIssue(
                code="NO_OBSERVATIONS",
                message="Actions were proposed but no observations recorded.",
                severity="error",
            )
        )
        score += 1.0
    if any(obs.refusal is not None for obs in trace.observations):
        issues.append(
            CritiqueIssue(
                code="REFUSAL_OCCURRED",
                message="At least one tool call resulted in refusal.",
                severity="warning",
            )
        )
        score += 0.5
    should_revise = score >= threshold
    return CritiqueResult(
        issues=issues,
        score=score,
        threshold=threshold,
        should_revise=should_revise,
    )
