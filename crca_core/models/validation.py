"""Spec validation (DraftSpec or LockedSpec).

Validation is intentionally conservative: it checks for obvious structural
issues and missing required fields for different operations.
"""

from __future__ import annotations

from typing import Iterable, Optional

from crca_core.models.result import ValidationIssue, ValidationReport
from crca_core.models.spec import DraftSpec, LockedSpec


def _node_names(nodes) -> set[str]:
    return {n.name for n in nodes}


def validate_spec(spec: DraftSpec | LockedSpec) -> ValidationReport:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    # Basic graph consistency
    node_names = _node_names(spec.graph.nodes)
    for e in spec.graph.edges:
        if e.source not in node_names:
            errors.append(
                ValidationIssue(
                    code="EDGE_SOURCE_UNKNOWN",
                    message=f"Edge source '{e.source}' not in nodes",
                    path="graph.edges",
                )
            )
        if e.target not in node_names:
            errors.append(
                ValidationIssue(
                    code="EDGE_TARGET_UNKNOWN",
                    message=f"Edge target '{e.target}' not in nodes",
                    path="graph.edges",
                )
            )

    # Time-series checks
    if spec.data.time_index is not None:
        time_col = spec.data.time_index.column
        if time_col and time_col not in {c.name for c in spec.data.columns}:
            errors.append(
                ValidationIssue(
                    code="TIME_INDEX_COLUMN_UNKNOWN",
                    message=f"time_index.column '{time_col}' not in data columns",
                    path="data.time_index.column",
                )
            )

    # Roles consistency
    for v in spec.roles.treatments + spec.roles.outcomes + spec.roles.mediators + spec.roles.instruments:
        if v not in node_names:
            warnings.append(
                ValidationIssue(
                    code="ROLE_NODE_UNKNOWN",
                    message=f"Role variable '{v}' not present in graph nodes",
                    path="roles",
                )
            )

    ok = len(errors) == 0
    return ValidationReport(ok=ok, errors=errors, warnings=warnings)

