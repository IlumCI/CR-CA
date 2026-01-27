"""Structured memory store for reasoning artifacts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class StructuredMemory:
    """Write-on-finalize memory store."""

    def __init__(self) -> None:
        self._store: List[Dict[str, Any]] = []
        self._staged: List[Dict[str, Any]] = []

    def read_all(self) -> List[Dict[str, Any]]:
        return list(self._store)

    def stage_write(self, item: Dict[str, Any]) -> None:
        self._staged.append(item)

    def finalize(self) -> None:
        self._store.extend(self._staged)
        self._staged = []

