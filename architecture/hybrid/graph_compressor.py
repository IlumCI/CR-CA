"""
Graph compression and abstraction module.

Provides functionality for creating composite nodes, latent factors,
and graph simplification through abstraction.
"""

from typing import Dict, List, Optional, Set, Any
import logging

from templates.graph_management import GraphManager

logger = logging.getLogger(__name__)


class GraphCompressor:
    """
    Compresses and abstracts causal graphs through composite nodes and latent factors.
    
    Features:
    - Composite node creation (multiple variables → single node)
    - Latent factor abstraction
    - Graph simplification (remove low-confidence edges, merge redundant paths)
    - Hierarchical structure creation
    """
    
    def __init__(self, graph_manager: GraphManager):
        """
        Initialize graph compressor.
        
        Args:
            graph_manager: GraphManager instance to compress
        """
        self.graph_manager = graph_manager
        self.composite_nodes: Dict[str, List[str]] = {}  # composite_name -> [original_nodes]
        self.latent_factors: Dict[str, List[str]] = {}  # latent_name -> [observed_nodes]
    
    def create_composite_node(
        self,
        variables: List[str],
        name: str,
        preserve_originals: bool = True
    ) -> str:
        """
        Create a composite node from multiple variables.
        
        Example: "product quality" + "service quality" → "quality_factor"
        
        Args:
            variables: List of variable names to combine
            name: Name for the composite node
            preserve_originals: Whether to keep original nodes in graph
            
        Returns:
            Name of the created composite node
        """
        # Validate variables exist
        for var in variables:
            if var not in self.graph_manager.get_nodes():
                logger.warning(f"Variable {var} not in graph, skipping composite creation")
                return name
        
        # Store composite mapping
        self.composite_nodes[name] = variables.copy()
        
        # Ensure composite node exists
        self.graph_manager.ensure_node_exists(name)
        
        # Aggregate edges from original variables
        # Incoming edges: union of all incoming edges from original variables
        incoming_sources: Set[str] = set()
        for var in variables:
            parents = self.graph_manager.get_parents(var)
            incoming_sources.update(parents)
        
        for source in incoming_sources:
            # Aggregate strength from all original variables
            total_strength = 0.0
            count = 0
            for var in variables:
                strength = self.graph_manager.edge_strength(source, var)
                if strength > 0:
                    total_strength += strength
                    count += 1
            
            if count > 0:
                avg_strength = total_strength / count
                self.graph_manager.add_relationship(
                    source, name,
                    strength=avg_strength,
                    relation_type="composite_incoming"
                )
        
        # Outgoing edges: union of all outgoing edges from original variables
        outgoing_targets: Set[str] = set()
        for var in variables:
            children = self.graph_manager.get_children(var)
            outgoing_targets.update(children)
        
        for target in outgoing_targets:
            # Aggregate strength from all original variables
            total_strength = 0.0
            count = 0
            for var in variables:
                strength = self.graph_manager.edge_strength(var, target)
                if strength > 0:
                    total_strength += strength
                    count += 1
            
            if count > 0:
                avg_strength = total_strength / count
                self.graph_manager.add_relationship(
                    name, target,
                    strength=avg_strength,
                    relation_type="composite_outgoing"
                )
        
        # Optionally remove original nodes
        if not preserve_originals:
            for var in variables:
                # Remove all edges involving this variable
                children = self.graph_manager.get_children(var)
                for child in children:
                    # Edge removal would need to be added to GraphManager
                    pass
                parents = self.graph_manager.get_parents(var)
                for parent in parents:
                    # Edge removal would need to be added to GraphManager
                    pass
        
        return name
    
    def abstract_to_latent(self, variables: List[str], latent_name: Optional[str] = None) -> str:
        """
        Abstract multiple observed variables into a latent factor.
        
        Example: "product quality", "service quality" → "quality_factor" (latent)
        
        Args:
            variables: List of observed variable names
            latent_name: Name for latent factor (auto-generated if None)
            
        Returns:
            Name of the created latent factor
        """
        if latent_name is None:
            latent_name = f"latent_{len(self.latent_factors)}"
        
        # Store latent mapping
        self.latent_factors[latent_name] = variables.copy()
        
        # Ensure latent node exists
        self.graph_manager.ensure_node_exists(latent_name)
        
        # Create edges: latent → observed (latent causes observed)
        for var in variables:
            self.graph_manager.add_relationship(
                latent_name, var,
                strength=1.0,
                relation_type="latent_to_observed"
            )
        
        # Inherit incoming edges from observed variables
        incoming_sources: Set[str] = set()
        for var in variables:
            parents = self.graph_manager.get_parents(var)
            incoming_sources.update(parents)
        
        for source in incoming_sources:
            if source not in variables:  # Don't create self-loops
                total_strength = 0.0
                count = 0
                for var in variables:
                    strength = self.graph_manager.edge_strength(source, var)
                    if strength > 0:
                        total_strength += strength
                        count += 1
                
                if count > 0:
                    avg_strength = total_strength / count
                    self.graph_manager.add_relationship(
                        source, latent_name,
                        strength=avg_strength,
                        relation_type="latent_incoming"
                    )
        
        return latent_name
    
    def compress_subgraph(self, nodes: List[str]) -> Dict[str, Any]:
        """
        Compress a subgraph by removing low-confidence edges and merging redundant paths.
        
        Args:
            nodes: List of nodes in the subgraph to compress
            
        Returns:
            Dictionary with compression statistics
        """
        removed_edges = 0
        merged_paths = 0
        
        # Remove low-confidence edges (confidence < 0.3)
        edges_to_remove = []
        for source in nodes:
            if source not in self.graph_manager.graph:
                continue
            for target, meta in self.graph_manager.graph[source].items():
                if target not in nodes:
                    continue
                if isinstance(meta, dict):
                    confidence = meta.get("confidence", 1.0)
                    if confidence < 0.3:
                        edges_to_remove.append((source, target))
        
        # Note: Edge removal would need to be implemented in GraphManager
        removed_edges = len(edges_to_remove)
        
        # Merge redundant paths (simplified: detect and mark for merging)
        # In a full implementation, would identify paths with same source/target
        # and merge them into a single edge with aggregated strength
        
        return {
            "removed_edges": removed_edges,
            "merged_paths": merged_paths,
            "original_nodes": len(nodes),
            "compressed_nodes": len(nodes)  # Would be reduced in full implementation
        }
    
    def simplify_graph(
        self,
        min_confidence: float = 0.3,
        min_strength: float = 0.1
    ) -> Dict[str, Any]:
        """
        Simplify entire graph by removing low-confidence/strength edges.
        
        Args:
            min_confidence: Minimum confidence threshold
            min_strength: Minimum strength threshold
            
        Returns:
            Dictionary with simplification statistics
        """
        all_nodes = self.graph_manager.get_nodes()
        removed_edges = 0
        
        edges_to_remove = []
        for source in all_nodes:
            if source not in self.graph_manager.graph:
                continue
            for target, meta in self.graph_manager.graph[source].items():
                if isinstance(meta, dict):
                    confidence = meta.get("confidence", 1.0)
                    strength = meta.get("strength", 1.0)
                    if confidence < min_confidence or strength < min_strength:
                        edges_to_remove.append((source, target))
        
        removed_edges = len(edges_to_remove)
        
        # Note: Actual edge removal would need GraphManager.remove_edge method
        # For now, just return statistics
        
        return {
            "removed_edges": removed_edges,
            "remaining_nodes": len(all_nodes),
            "min_confidence": min_confidence,
            "min_strength": min_strength
        }
    
    def get_composite_mapping(self) -> Dict[str, List[str]]:
        """
        Get mapping of composite nodes to original variables.
        
        Returns:
            Dictionary mapping composite node names to lists of original variables
        """
        return self.composite_nodes.copy()
    
    def get_latent_mapping(self) -> Dict[str, List[str]]:
        """
        Get mapping of latent factors to observed variables.
        
        Returns:
            Dictionary mapping latent factor names to lists of observed variables
        """
        return self.latent_factors.copy()
