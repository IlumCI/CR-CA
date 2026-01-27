"""Typed outputs for crca_llm orchestrator."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from crca_core.models.refusal import RefusalResult
from crca_core.models.result import AnyResult
from crca_llm.coauthor import DraftBundle


class LLMRunResult(BaseModel):
    """Container for LLM drafts + core results + refusals."""

    draft_bundle: DraftBundle
    core_results: List[AnyResult] = Field(default_factory=list)
    refusals: List[RefusalResult] = Field(default_factory=list)
    llm_metadata: Optional[dict] = None

