"""Hard-refusal types for H1 enforcement."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RefusalReasonCode(str, Enum):
    """Stable reason codes for refusal-first behavior."""

    SPEC_NOT_LOCKED = "SPEC_NOT_LOCKED"
    NO_SCM_FOR_COUNTERFACTUAL = "NO_SCM_FOR_COUNTERFACTUAL"
    NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
    TIME_INDEX_INVALID = "TIME_INDEX_INVALID"
    ASSUMPTIONS_UNDECLARED = "ASSUMPTIONS_UNDECLARED"
    INPUT_INVALID = "INPUT_INVALID"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"


class RefusalChecklistItem(BaseModel):
    """A single required input/action needed to proceed."""

    item: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)


class RefusalResult(BaseModel):
    """Structured refusal (no numeric causal output)."""

    result_type: str = Field(default="Refusal", frozen=True)
    reason_codes: List[RefusalReasonCode] = Field(default_factory=list)
    message: str = Field(..., min_length=1)
    checklist: List[RefusalChecklistItem] = Field(default_factory=list)
    suggested_next_steps: List[str] = Field(default_factory=list)
    details: Optional[str] = None

