"""Frontdoor identification helper."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import networkx as nx

from crca_core.identify.graph import CausalGraph


def _directed_paths_through(
    graph: CausalGraph, treatment: str, outcome: str, mediator: str
) -> bool:
    """Return True if all directed paths from treatment to outcome go through mediator."""
    try:
        paths = list(nx.all_simple_paths(graph.directed, treatment, outcome))
    except nx.NetworkXNoPath:
        return False
    if not paths:
        return False
    return all(mediator in p for p in paths)


def find_frontdoor_mediator(
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    mediators: Sequence[str],
) -> Optional[str]:
    """Return a mediator that satisfies a conservative frontdoor check."""
    for m in mediators:
        if m in (treatment, outcome):
            continue
        if not _directed_paths_through(graph, treatment, outcome, m):
            continue

        # No backdoor from X to M (empty set) in graph with outgoing edges removed.
        if not graph.remove_outgoing([treatment]).d_separated([treatment], [m], []):
            continue

        # Backdoor from M to Y is blocked by X (conservative check).
        if not graph.remove_outgoing([m]).d_separated([m], [outcome], [treatment]):
            continue

        return m
    return None

