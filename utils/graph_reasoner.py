"""
Graph-first reasoning engine for CRCA.

A standalone module that provides pure graph-based reasoning capabilities.
All answers come from graph state only - never directly from text parsing.

This module is designed to be reusable across CRCA components and works
with any GraphManager instance.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
import logging
from collections import deque, defaultdict

from templates.graph_management import GraphManager

logger = logging.getLogger(__name__)


class GraphFirstReasoner:
    """
    Graph-first reasoning engine that ONLY uses graph state for answers.
    
    Never parses text directly for answers. All reasoning is derived from
    graph structure, edge strengths, temporal relationships, and graph traversal.
    
    This is a standalone utility that can be used by:
    - Hybrid agent
    - Other CRCA components
    - Any code needing graph-based reasoning
    """
    
    def __init__(self, graph_manager: Optional[GraphManager] = None):
        """
        Initialize the graph-first reasoner.
        
        Args:
            graph_manager: Optional GraphManager instance. If None, must be provided in method calls.
        """
        self.graph_manager = graph_manager
    
    def reason_from_graph(
        self,
        task: str,
        graph_state: Optional[Dict[str, Any]] = None,
        graph_manager: Optional[GraphManager] = None
    ) -> Dict[str, Any]:
        """
        Reason about a task using ONLY graph state.
        
        This method never parses the task text directly for answers.
        It queries the graph structure to derive answers.
        
        Args:
            task: Natural language task (used for intent detection only, not for answers)
            graph_state: Optional explicit graph state dict
            graph_manager: GraphManager instance (uses self.graph_manager if not provided)
            
        Returns:
            Dictionary with reasoning results derived from graph state
        """
        gm = graph_manager or self.graph_manager
        if gm is None:
            raise ValueError("GraphManager must be provided either in __init__ or as parameter")
        
        # Extract graph state from GraphManager if not provided
        if graph_state is None:
            graph_state = self._extract_graph_state(gm)
        
        # Detect query intent from task (but don't use task for answers)
        intent = self._detect_query_intent(task)
        
        # Reason from graph state only
        result = self._reason_from_graph_state(graph_state, intent, gm)
        
        return result
    
    def query_causal_path(
        self,
        source: str,
        target: str,
        graph_manager: Optional[GraphManager] = None
    ) -> List[str]:
        """
        Query for a causal path from source to target using graph structure.
        
        Args:
            source: Source node
            target: Target node
            graph_manager: GraphManager instance (uses self.graph_manager if not provided)
            
        Returns:
            List of nodes forming the causal path, or empty list if no path exists
        """
        gm = graph_manager or self.graph_manager
        if gm is None:
            raise ValueError("GraphManager must be provided")
        
        # Use GraphManager's path finding
        path = gm.identify_path(source, target)
        
        # If no direct path, try to find indirect paths through intermediate nodes
        if not path:
            path = self._find_indirect_path(source, target, gm)
        
        return path
    
    def query_effects(
        self,
        intervention: Dict[str, float],
        graph_manager: Optional[GraphManager] = None,
        max_depth: int = 5
    ) -> Dict[str, float]:
        """
        Query for effects of an intervention using graph traversal.
        
        Args:
            intervention: Dictionary mapping variable names to intervention values
            graph_manager: GraphManager instance (uses self.graph_manager if not provided)
            max_depth: Maximum depth to traverse from intervention nodes
            
        Returns:
            Dictionary mapping affected variables to their expected values
        """
        gm = graph_manager or self.graph_manager
        if gm is None:
            raise ValueError("GraphManager must be provided")
        
        effects: Dict[str, float] = {}
        visited: Set[str] = set()
        
        # Start from intervention nodes
        queue: deque = deque([(node, value, 0) for node, value in intervention.items()])
        
        while queue:
            current_node, current_value, depth = queue.popleft()
            
            if depth > max_depth or current_node in visited:
                continue
            
            visited.add(current_node)
            
            # Store effect
            if current_node not in intervention:  # Don't overwrite intervention values
                effects[current_node] = current_value
            
            # Traverse to children
            children = gm.get_children(current_node)
            for child in children:
                if child in visited:
                    continue
                
                # Get edge strength
                edge_strength = gm.edge_strength(current_node, child)
                
                # Calculate effect (simple linear propagation)
                child_value = current_value * edge_strength
                
                queue.append((child, child_value, depth + 1))
        
        return effects
    
    def query_temporal_sequence(
        self,
        variable: str,
        time_horizon: int,
        graph_manager: Optional[GraphManager] = None
    ) -> Dict[int, float]:
        """
        Query for temporal sequence of a variable over time.
        
        Args:
            variable: Variable to track
            time_horizon: Number of time steps to project
            graph_manager: GraphManager instance (uses self.graph_manager if not provided)
            
        Returns:
            Dictionary mapping time step to expected value
        """
        gm = graph_manager or self.graph_manager
        if gm is None:
            raise ValueError("GraphManager must be provided")
        
        sequence: Dict[int, float] = {}
        
        # Get initial value from graph (if available)
        initial_value = 1.0  # Default
        
        # Check for temporal edges affecting this variable
        parents = gm.get_parents(variable)
        
        for t in range(time_horizon):
            value = initial_value
            
            # Aggregate effects from parents
            for parent in parents:
                edge_strength = gm.edge_strength(parent, variable)
                parent_value = sequence.get(t - 1, initial_value) if t > 0 else initial_value
                value += parent_value * edge_strength
            
            sequence[t] = value
        
        return sequence
    
    def query_feedback_loops(
        self,
        variable: str,
        graph_manager: Optional[GraphManager] = None
    ) -> List[Dict[str, Any]]:
        """
        Query for feedback loops involving a variable.
        
        Args:
            variable: Variable to check for feedback loops
            graph_manager: GraphManager instance (uses self.graph_manager if not provided)
            
        Returns:
            List of feedback loop descriptions
        """
        gm = graph_manager or self.graph_manager
        if gm is None:
            raise ValueError("GraphManager must be provided")
        
        loops: List[Dict[str, Any]] = []
        
        # Find cycles involving this variable
        visited: Set[str] = set()
        path: List[str] = []
        
        def find_cycles(node: str, target: str) -> None:
            """Find cycles starting from node that return to target."""
            if node in visited:
                if node == target and len(path) > 1:
                    # Found a cycle
                    loop_path = path + [target]
                    loops.append({
                        "path": loop_path,
                        "length": len(loop_path) - 1,
                        "strength": self._calculate_loop_strength(loop_path, gm)
                    })
                return
            
            visited.add(node)
            path.append(node)
            
            children = gm.get_children(node)
            for child in children:
                find_cycles(child, target)
            
            path.pop()
            visited.remove(node)
        
        find_cycles(variable, variable)
        
        return loops
    
    def query_graph_state(
        self,
        question: str,
        graph_manager: Optional[GraphManager] = None
    ) -> Dict[str, Any]:
        """
        Query graph state to answer a question.
        
        Args:
            question: Natural language question
            graph_manager: GraphManager instance (uses self.graph_manager if not provided)
            
        Returns:
            Dictionary with answer derived from graph state
        """
        gm = graph_manager or self.graph_manager
        if gm is None:
            raise ValueError("GraphManager must be provided")
        
        # Extract graph state
        graph_state = self._extract_graph_state(gm)
        
        # Detect query intent
        intent = self._detect_query_intent(question)
        
        # Answer from graph state
        answer = self._reason_from_graph_state(graph_state, intent, gm)
        
        return answer
    
    def reason_from_graph_state(
        self,
        state: Dict[str, Any],
        query: str,
        graph_manager: Optional[GraphManager] = None
    ) -> Dict[str, Any]:
        """
        Pure graph reasoning from explicit graph state.
        
        Args:
            state: Graph state dictionary
            query: Query string (for intent detection)
            graph_manager: GraphManager instance (uses self.graph_manager if not provided)
            
        Returns:
            Dictionary with reasoning results
        """
        gm = graph_manager or self.graph_manager
        if gm is None:
            raise ValueError("GraphManager must be provided")
        
        intent = self._detect_query_intent(query)
        
        return self._reason_from_graph_state(state, intent, gm)
    
    # Private helper methods
    
    def _extract_graph_state(self, graph_manager: GraphManager) -> Dict[str, Any]:
        """
        Extract current graph state from GraphManager.
        
        Args:
            graph_manager: GraphManager instance
            
        Returns:
            Dictionary representing graph state
        """
        nodes = graph_manager.get_nodes()
        edges = graph_manager.get_edges()
        
        # Build edge structure with metadata
        edge_data = {}
        for source, target in edges:
            edge_meta = graph_manager.graph.get(source, {}).get(target, {})
            edge_data[(source, target)] = {
                "strength": edge_meta.get("strength", 1.0),
                "confidence": edge_meta.get("confidence", 1.0),
                "relation_type": edge_meta.get("relation_type", "causal"),
                **{k: v for k, v in edge_meta.items() if k not in ["strength", "confidence", "relation_type"]}
            }
        
        return {
            "nodes": nodes,
            "edges": edges,
            "edge_data": edge_data,
            "topological_order": graph_manager.topological_sort(),
            "is_dag": graph_manager.is_dag()
        }
    
    def _detect_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect intent from query (but don't use query for answers).
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with intent information
        """
        query_lower = query.lower()
        
        intent = {
            "type": "analysis",  # default
            "question_type": None,
            "target_variables": [],
            "intervention_variables": [],
            "temporal": False
        }
        
        # Question type detection
        if any(word in query_lower for word in ["what", "which", "who"]):
            intent["question_type"] = "what"
        elif any(word in query_lower for word in ["how", "why"]):
            intent["question_type"] = "how"
        elif any(word in query_lower for word in ["when", "where"]):
            intent["question_type"] = "when_where"
        
        # Intent type detection
        if any(word in query_lower for word in ["predict", "forecast", "estimate", "will", "would"]):
            intent["type"] = "prediction"
        elif any(word in query_lower for word in ["what if", "if", "suppose", "assume"]):
            intent["type"] = "counterfactual"
        elif any(word in query_lower for word in ["effect", "impact", "influence"]):
            intent["type"] = "effect_analysis"
        elif any(word in query_lower for word in ["path", "connection", "link"]):
            intent["type"] = "path_query"
        elif any(word in query_lower for word in ["feedback", "loop", "cycle"]):
            intent["type"] = "feedback_analysis"
        
        # Temporal detection
        if any(word in query_lower for word in ["before", "after", "delay", "time", "days", "hours"]):
            intent["temporal"] = True
        
        return intent
    
    def _reason_from_graph_state(
        self,
        graph_state: Dict[str, Any],
        intent: Dict[str, Any],
        graph_manager: GraphManager
    ) -> Dict[str, Any]:
        """
        Core reasoning logic using graph state only.
        
        Args:
            graph_state: Graph state dictionary
            intent: Intent dictionary
            graph_manager: GraphManager instance
            
        Returns:
            Dictionary with reasoning results
        """
        result = {
            "reasoning_type": intent["type"],
            "graph_nodes": graph_state["nodes"],
            "graph_edges": graph_state["edges"],
            "answer": None,
            "confidence": 1.0,
            "supporting_evidence": []
        }
        
        intent_type = intent["type"]
        
        if intent_type == "path_query":
            # Find paths between variables mentioned in query
            # This is a simplified version - in practice, extract variables from query
            if len(graph_state["nodes"]) >= 2:
                path = self.query_causal_path(
                    graph_state["nodes"][0],
                    graph_state["nodes"][-1],
                    graph_manager
                )
                result["answer"] = f"Path found: {' -> '.join(path)}" if path else "No path found"
                result["supporting_evidence"] = [{"type": "path", "path": path}]
        
        elif intent_type == "effect_analysis":
            # Analyze effects from graph structure
            if graph_state["nodes"]:
                # Use first node as example (in practice, extract from query)
                source = graph_state["nodes"][0]
                effects = self.query_effects({source: 1.0}, graph_manager)
                result["answer"] = f"Effects of {source}: {effects}"
                result["supporting_evidence"] = [{"type": "effects", "effects": effects}]
        
        elif intent_type == "feedback_analysis":
            # Find feedback loops
            if graph_state["nodes"]:
                loops = self.query_feedback_loops(graph_state["nodes"][0], graph_manager)
                result["answer"] = f"Found {len(loops)} feedback loops"
                result["supporting_evidence"] = [{"type": "feedback_loops", "loops": loops}]
        
        elif intent_type == "prediction":
            # Make prediction from graph structure
            if graph_state["nodes"]:
                sequence = self.query_temporal_sequence(graph_state["nodes"][0], 10, graph_manager)
                result["answer"] = f"Temporal sequence: {sequence}"
                result["supporting_evidence"] = [{"type": "temporal_sequence", "sequence": sequence}]
        
        else:
            # Default: general graph analysis
            result["answer"] = f"Graph contains {len(graph_state['nodes'])} nodes and {len(graph_state['edges'])} edges"
            result["supporting_evidence"] = [
                {"type": "graph_structure", "nodes": graph_state["nodes"], "edges": graph_state["edges"]}
            ]
        
        return result
    
    def _find_indirect_path(
        self,
        source: str,
        target: str,
        graph_manager: GraphManager,
        max_depth: int = 10
    ) -> List[str]:
        """
        Find indirect path through intermediate nodes.
        
        Args:
            source: Source node
            target: Target node
            graph_manager: GraphManager instance
            max_depth: Maximum search depth
            
        Returns:
            Path as list of nodes, or empty list
        """
        # Use BFS to find shortest path
        queue: deque = deque([(source, [source])])
        visited: Set[str] = {source}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            children = graph_manager.get_children(current)
            for child in children:
                if child == target:
                    return path + [child]
                
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
        
        return []
    
    def _calculate_loop_strength(self, loop_path: List[str], graph_manager: GraphManager) -> float:
        """
        Calculate overall strength of a feedback loop.
        
        Args:
            loop_path: List of nodes forming the loop
            graph_manager: GraphManager instance
            
        Returns:
            Overall loop strength
        """
        if len(loop_path) < 2:
            return 0.0
        
        strengths = []
        for i in range(len(loop_path) - 1):
            source = loop_path[i]
            target = loop_path[i + 1]
            strength = graph_manager.edge_strength(source, target)
            strengths.append(strength)
        
        # Multiply all edge strengths (assuming independence)
        total_strength = 1.0
        for s in strengths:
            total_strength *= s
        
        return total_strength
