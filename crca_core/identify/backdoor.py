"""Backdoor identification helper."""

from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Set

from crca_core.identify.graph import CausalGraph


def find_backdoor_adjustment_set(
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    *,
    max_candidates: int = 12,
    max_set_size: int = 6,
) -> Optional[Set[str]]:
    """Find a valid backdoor adjustment set (if any).

    This uses a bounded search over observed, non-descendant candidates.
    """
    x = treatment
    y = outcome
    observed = set(graph.observed)
    descendants_x = graph.descendants([x])
    candidates = sorted(list(observed - {x, y} - descendants_x))

    if len(candidates) > max_candidates:
        return None

    g_bd = graph.remove_outgoing([x])
    for k in range(0, min(max_set_size, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, k):
            z = set(combo)
            if g_bd.d_separated([x], [y], list(z)):
                return z
    return None

