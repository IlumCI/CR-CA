"""Spec lifecycle: DraftSpec → LockedSpec.

The LockedSpec is a scientific boundary: only LockedSpec may be used for numeric
causal outputs. This module enforces that boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from crca_core.models.spec import DraftSpec, LockedSpec
from utils.canonical import stable_hash


def lock_spec(draft: DraftSpec, approvals: List[str]) -> LockedSpec:
    """Lock a draft spec by hashing its canonical content and recording approvals.

    Args:
        draft: The draft specification (possibly LLM-generated).
        approvals: Human (or explicit programmatic) approvals. Must be non-empty.

    Returns:
        LockedSpec

    Raises:
        ValueError: If approvals are empty.
    """

    if not approvals:
        raise ValueError("approvals must be non-empty to lock a spec")

    # Canonicalize via stable_hash over model_dump
    draft_payload = draft.model_dump()
    spec_hash = stable_hash(draft_payload)
    locked_at = datetime.now(timezone.utc).isoformat()

    return LockedSpec(
        spec_hash=spec_hash,
        approvals=list(approvals),
        locked_at_utc=locked_at,
        data=draft.data,
        graph=draft.graph,
        roles=draft.roles,
        assumptions=draft.assumptions,
        scm=draft.scm,
    )

