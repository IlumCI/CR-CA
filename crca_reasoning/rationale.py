"""Rationale trace output (non-scientific)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RationaleTrace(BaseModel):
    steps: list[str] = Field(default_factory=list)

