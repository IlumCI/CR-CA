"""Instrumental variable identification helper (conservative)."""

from __future__ import annotations

from typing import Optional, Sequence

import networkx as nx

from crca_core.identify.graph import CausalGraph


def find_instrument(
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    instruments: Sequence[str],
) -> Optional[str]:
    """Return a candidate instrument satisfying conservative IV checks."""
    for z in instruments:
        if z in (treatment, outcome):
            continue

        # Relevance: Z causes X
        try:
            if not nx.has_path(graph.directed, z, treatment):
                continue
        except nx.NetworkXError:
            continue

        # Exclusion: no directed path from Z to Y that avoids X
        try:
            paths = nx.all_simple_paths(graph.directed, z, outcome)
            bad = False
            for p in paths:
                if treatment not in p:
                    bad = True
                    break
            if bad:
                continue
        except nx.NetworkXNoPath:
            pass

        # Independence (conservative): Z and Y d-separated given X in graph removing X's outgoing edges
        if not graph.remove_outgoing([treatment]).d_separated([z], [outcome], [treatment]):
            continue

        return z
    return None
