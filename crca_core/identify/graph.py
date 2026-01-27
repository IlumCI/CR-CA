"""Graph utilities for identification.

Supports directed edges plus bidirected (latent confounding) edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from crca_core.models.spec import CausalGraphSpec


def _parse_latent_confounders(latent: Sequence[str]) -> Set[Tuple[str, str]]:
    """Parse latent confounder pairs from strings.

    Supported formats:
    - "A<->B"
    - "A,B"
    """
    pairs: Set[Tuple[str, str]] = set()
    for item in latent:
        if "<->" in item:
            a, b = [s.strip() for s in item.split("<->", 1)]
        elif "," in item:
            a, b = [s.strip() for s in item.split(",", 1)]
        else:
            # Unknown format, skip conservatively.
            continue
        if a and b and a != b:
            pairs.add(tuple(sorted((a, b))))
    return pairs


@dataclass(frozen=True)
class CausalGraph:
    nodes: Tuple[str, ...]
    observed: Set[str]
    directed: nx.DiGraph
    bidirected: Set[Tuple[str, str]]

    @classmethod
    def from_spec(cls, spec: CausalGraphSpec) -> "CausalGraph":
        nodes = tuple(n.name for n in spec.nodes)
        observed = {n.name for n in spec.nodes if n.observed}
        g = nx.DiGraph()
        for n in nodes:
            g.add_node(n)
        for e in spec.edges:
            g.add_edge(e.source, e.target)
        bidirected = _parse_latent_confounders(spec.latent_confounders)
        return cls(nodes=nodes, observed=observed, directed=g, bidirected=bidirected)

    def ancestors(self, nodes: Iterable[str]) -> Set[str]:
        anc: Set[str] = set()
        for n in nodes:
            anc |= nx.ancestors(self.directed, n)
        anc |= set(nodes)
        return anc

    def descendants(self, nodes: Iterable[str]) -> Set[str]:
        desc: Set[str] = set()
        for n in nodes:
            desc |= nx.descendants(self.directed, n)
        desc |= set(nodes)
        return desc

    def induced_subgraph(self, nodes: Set[str]) -> "CausalGraph":
        g = self.directed.subgraph(nodes).copy()
        bidirected = {p for p in self.bidirected if p[0] in nodes and p[1] in nodes}
        observed = {n for n in self.observed if n in nodes}
        return CausalGraph(nodes=tuple(nodes), observed=observed, directed=g, bidirected=bidirected)

    def remove_outgoing(self, x: Sequence[str]) -> "CausalGraph":
        g = self.directed.copy()
        for node in x:
            for _, child in list(g.out_edges(node)):
                g.remove_edge(node, child)
        return CausalGraph(nodes=self.nodes, observed=set(self.observed), directed=g, bidirected=set(self.bidirected))

    def remove_incoming(self, x: Sequence[str]) -> "CausalGraph":
        g = self.directed.copy()
        for node in x:
            for parent, _ in list(g.in_edges(node)):
                g.remove_edge(parent, node)
        return CausalGraph(nodes=self.nodes, observed=set(self.observed), directed=g, bidirected=set(self.bidirected))

    def c_components(self, nodes: Optional[Set[str]] = None) -> List[Set[str]]:
        """Compute c-components (bidirected connected components)."""
        nodes = nodes or set(self.nodes)
        # Build undirected graph of bidirected connections
        undirected = nx.Graph()
        undirected.add_nodes_from(nodes)
        for a, b in self.bidirected:
            if a in nodes and b in nodes:
                undirected.add_edge(a, b)
        return [set(c) for c in nx.connected_components(undirected)]

    def d_separated(self, x: Sequence[str], y: Sequence[str], z: Sequence[str]) -> bool:
        # Use networkx d-separation check
        from networkx.algorithms.d_separation import is_d_separator

        return bool(is_d_separator(self.directed, set(x), set(y), set(z)))

