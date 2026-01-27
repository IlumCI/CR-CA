"""Provenance manifest for reproducible causal R&D runs.

The manifest must not contain raw data; only hashes and schema summaries.
"""

from __future__ import annotations

import platform
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProvenanceManifest(BaseModel):
    """Required provenance for every `crca_core` result."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: str = Field(default_factory=utc_now_iso)
    spec_hash: str
    data_hash: Optional[str] = None

    library_versions: Dict[str, str] = Field(default_factory=dict)
    random_seeds: Dict[str, Any] = Field(default_factory=dict)
    algorithm_config: Dict[str, Any] = Field(default_factory=dict)
    hardware_notes: Optional[Dict[str, Any]] = None

    @classmethod
    def minimal(
        cls,
        *,
        spec_hash: str,
        data_hash: Optional[str] = None,
        random_seeds: Optional[Dict[str, Any]] = None,
        algorithm_config: Optional[Dict[str, Any]] = None,
    ) -> "ProvenanceManifest":
        """Create a minimal manifest with environment versions populated."""

        versions = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        return cls(
            spec_hash=spec_hash,
            data_hash=data_hash,
            library_versions=versions,
            random_seeds=random_seeds or {},
            algorithm_config=algorithm_config or {},
        )

