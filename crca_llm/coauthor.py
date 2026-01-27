"""Non-authoritative LLM coauthor layer.

This module is deliberately constrained:
- It may draft candidate `DraftSpec` objects and review checklists.
- It must never produce numeric causal outputs.
- It must never create `LockedSpec` objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from crca_core.models.spec import (
    AssumptionItem,
    AssumptionSpec,
    AssumptionStatus,
    CausalGraphSpec,
    DataColumnSpec,
    DataSpec,
    DraftSpec,
    NodeSpec,
    RoleSpec,
)


class CoauthorConfig(BaseModel):
    model: str = Field(default="gpt-4o-mini")
    n_candidates: int = Field(default=3, ge=1, le=5)


class DraftBundle(BaseModel):
    """LLM coauthor output: draft specs + checklists (no numbers)."""

    drafts: List[DraftSpec]
    review_checklist: List[str] = Field(default_factory=list)
    disclaimers: List[str] = Field(
        default_factory=lambda: [
            "DRAFT-NON-AUTHORITATIVE: This spec is a hypothesis and must be reviewed/locked by a human.",
            "No numeric causal claims are produced by the coauthor layer.",
        ]
    )


def _default_checklist() -> List[str]:
    return [
        "Define treatment and outcome variables explicitly.",
        "Confirm time ordering (what can cause what).",
        "List plausible confounders and whether they are measured.",
        "Check for collider variables that must not be adjusted for.",
        "Define intervention semantics (set vs shift vs mechanism-change) if counterfactuals are needed.",
        "State discovery assumptions if running causal discovery (faithfulness, causal sufficiency/latent confounding).",
    ]


def _offline_draft(user_text: str, observed_columns: Optional[Sequence[str]], n: int) -> DraftBundle:
    cols = list(observed_columns or [])
    data_spec = DataSpec(
        columns=[DataColumnSpec(name=c, dtype="unknown") for c in cols],
        measurement_error_notes="unknown",
    )

    # Minimal placeholder graph: nodes from observed columns (no edges).
    graph = CausalGraphSpec(nodes=[NodeSpec(name=c) for c in cols], edges=[])

    assumptions = AssumptionSpec(
        items=[
            AssumptionItem(
                name="DRAFT_ONLY",
                status=AssumptionStatus.unknown,
                description="This spec was generated offline without LLM; treat as a template only.",
            )
        ],
        falsification_plan=[
            "Collect domain constraints/time ordering and re-run drafting with an LLM (optional).",
        ],
    )

    drafts = [
        DraftSpec(
            data=data_spec,
            graph=graph,
            roles=RoleSpec(),
            assumptions=assumptions,
            draft_notes=f"Offline draft template generated from observed columns. User text: {user_text[:200]}",
        )
        for _ in range(n)
    ]
    return DraftBundle(drafts=drafts, review_checklist=_default_checklist())


@dataclass
class LLMCoauthor:
    """Coauthor that can draft candidate specs, but is never authoritative."""

    config: CoauthorConfig = field(default_factory=CoauthorConfig)

    def draft_specs(
        self,
        *,
        user_text: str,
        observed_columns: Optional[Sequence[str]] = None,
    ) -> DraftBundle:
        """Draft multiple candidate DraftSpec objects.

        If no OpenAI API key is configured, returns an offline template bundle.
        """

        if os.environ.get("OPENAI_API_KEY") is None:
            return _offline_draft(user_text, observed_columns, self.config.n_candidates)

        # Online LLM drafting is intentionally not implemented yet in this repo’s default test environment.
        # We return offline drafts to preserve deterministic behavior and avoid accidental numeric leakage.
        return _offline_draft(user_text, observed_columns, self.config.n_candidates)

