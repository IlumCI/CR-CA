"""In-house identification (ID) algorithm scaffold.

This module implements a conservative identification strategy:
- If no bidirected edges (no latent confounding), return g-formula.
- If latent confounding is present, we currently return not identifiable.

This is intentionally strict and avoids over-claiming identifiability.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from crca_core.identify.graph import CausalGraph


def identify_g_formula(
    graph: CausalGraph, treatment: str, outcome: str
) -> Tuple[str, str]:
    """Return g-formula expression for DAGs without latent confounding."""
    vars_all = sorted(list(graph.nodes))
    summation_vars = [v for v in vars_all if v not in {treatment, outcome}]
    summation = f"sum_{{{','.join(summation_vars)}}}" if summation_vars else ""
    expr = (
        f"{summation} Π_v P(v | Pa(v)) with do({treatment})"
        if summation
        else f"Π_v P(v | Pa(v)) with do({treatment})"
    )
    return "id_g_formula", expr


def id_algorithm(
    graph: CausalGraph,
    treatment: str,
    outcome: str,
) -> Optional[Tuple[str, str]]:
    """Return an identification expression if possible, otherwise None."""
    # If no bidirected edges, g-formula identifies interventional distribution.
    if not graph.bidirected:
        return identify_g_formula(graph, treatment, outcome)

    # Conservative: refuse when latent confounding is present (no over-claiming).
    return None
