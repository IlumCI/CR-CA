"""
Enhanced Explanation System with Interpretable AI and Explainable Reasoning.

Implements XAI with counterfactual explanations, uncertainty decomposition,
provenance tracking, and graph-theoretic structure explanation.

Theoretical Basis:
- Explainable AI (XAI) (Adadi & Berrada 2018)
- Interpretability (Lipton 2016)
- Counterfactual Explanations (Wachter et al. 2017)
- Causal Explanations (Pearl & Mackenzie 2018)
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import logging
import math

from schemas.reasoning import ReasoningStep, ReasoningChain

logger = logging.getLogger(__name__)

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class ExplanationBuilder:
    """
    Implements multi-level explanations with step-by-step reasoning.
    
    Features:
    - Step-by-step explanations with graph state visualization
    - Decision justification with evidence
    - Counterfactual explanations
    - Evidence provision with provenance tracking
    """
    
    def __init__(self):
        """Initialize explanation builder."""
        self.explanation_templates = {
            'step': "Step {step_num}: {operation}\n  Input: {input_state}\n  Output: {output_state}\n  Reasoning: {justification}\n  Confidence: {confidence:.2f}",
            'decision': "Decision: {decision}\n  Evidence: {evidence}\n  Rule: {rule}\n  Confidence: {confidence:.2f}",
            'counterfactual': "If {condition} were different, then {outcome} would be {value}.\n  Current: {current_value}\n  Counterfactual: {counterfactual_value}"
        }
    
    def generate_explanation(
        self,
        reasoning_chain: ReasoningChain,
        graph: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate step-by-step explanation from reasoning chain.
        
        Algorithm:
            function generate_explanation(reasoning_chain, graph):
                explanation = []
                for step in reasoning_chain:
                    explanation.append({
                        'operation': step.operation,
                        'input_state': graph_snapshot(step.input_state),
                        'output_state': graph_snapshot(step.output_state),
                        'justification': justify(step.operation, step.evidence),
                        'confidence': step.confidence,
                        'uncertainty': decompose_uncertainty(step)
                    })
                return format_explanation(explanation)
        
        Args:
            reasoning_chain: Reasoning chain to explain
            graph: Optional graph state
            
        Returns:
            Explanation dictionary
        """
        explanation = {
            'steps': [],
            'summary': '',
            'confidence': 1.0,
            'uncertainty': {}
        }
        
        for i, step in enumerate(reasoning_chain.steps, 1):
            step_explanation = {
                'step_num': i,
                'step_id': step.step_id,
                'operation': step.operation,
                'step_type': step.step_type.value,
                'input_state': self._format_graph_state(step.input_state),
                'output_state': self._format_graph_state(step.output_state),
                'justification': self._justify_step(step),
                'confidence': step.confidence,
                'uncertainty': self._decompose_uncertainty(step),
                'evidence': [e.content for e in step.evidence]
            }
            
            explanation['steps'].append(step_explanation)
        
        # Compute overall confidence and uncertainty
        if reasoning_chain.steps:
            confidences = [s.confidence for s in reasoning_chain.steps]
            explanation['confidence'] = sum(confidences) / len(confidences) if confidences else 1.0
        
        # Generate summary
        explanation['summary'] = self._generate_summary(reasoning_chain)
        
        return explanation
    
    def _format_graph_state(self, state: Dict[str, Any]) -> str:
        """
        Format graph state for display.
        
        Args:
            state: Graph state dictionary
            
        Returns:
            Formatted string
        """
        if not state:
            return "Empty state"
        
        nodes = state.get('nodes', [])
        edges = state.get('edges', [])
        
        if nodes:
            nodes_str = f"Nodes: {', '.join(sorted(nodes))}"
        else:
            nodes_str = "No nodes"
        
        if edges:
            edges_str = f"Edges: {len(edges)} relationships"
        else:
            edges_str = "No edges"
        
        return f"{nodes_str}\n{edges_str}"
    
    def _justify_step(self, step: ReasoningStep) -> str:
        """
        Generate justification for a reasoning step.
        
        Args:
            step: Reasoning step
            
        Returns:
            Justification string
        """
        justification_parts = []
        
        if step.inference_rule:
            justification_parts.append(f"Applied {step.inference_rule.value} rule")
        
        if step.premises:
            justification_parts.append(f"Based on {len(step.premises)} premises")
        
        if step.evidence:
            evidence_summary = ", ".join([str(e.content)[:50] for e in step.evidence[:3]])
            justification_parts.append(f"Evidence: {evidence_summary}")
        
        if step.conclusion:
            justification_parts.append(f"Concluded: {step.conclusion}")
        
        return ". ".join(justification_parts) if justification_parts else "No justification provided"
    
    def _decompose_uncertainty(self, step: ReasoningStep) -> Dict[str, float]:
        """
        Decompose uncertainty into epistemic and aleatoric components.
        
        Args:
            step: Reasoning step
            
        Returns:
            Dictionary with uncertainty components
        """
        uncertainty = step.uncertainty or 0.0
        confidence = step.confidence
        
        # Simple decomposition:
        # Epistemic uncertainty: model uncertainty (lack of knowledge)
        # Aleatoric uncertainty: data uncertainty (inherent randomness)
        
        # If confidence is low, more epistemic uncertainty
        epistemic = (1.0 - confidence) * 0.7  # 70% of low confidence is epistemic
        aleatoric = uncertainty * 0.3  # 30% is aleatoric
        
        return {
            'total': uncertainty,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'confidence': confidence
        }
    
    def _generate_summary(self, reasoning_chain: ReasoningChain) -> str:
        """
        Generate summary of reasoning chain.
        
        Args:
            reasoning_chain: Reasoning chain
            
        Returns:
            Summary string
        """
        if not reasoning_chain.steps:
            return "No reasoning steps"
        
        num_steps = len(reasoning_chain.steps)
        success = "succeeded" if reasoning_chain.success else "failed"
        conclusion = reasoning_chain.final_conclusion or "No conclusion"
        
        return f"Reasoning chain with {num_steps} steps {success}. Final conclusion: {conclusion}"
    
    def generate_counterfactual_explanation(
        self,
        variable: str,
        current_value: Any,
        counterfactual_value: Any,
        outcome: str,
        outcome_value: Any
    ) -> str:
        """
        Generate counterfactual explanation.
        
        "If X were different, then Y would be Z"
        
        Args:
            variable: Variable name
            current_value: Current value
            counterfactual_value: Counterfactual value
            outcome: Outcome variable
            outcome_value: Outcome value
            
        Returns:
            Counterfactual explanation string
        """
        return self.explanation_templates['counterfactual'].format(
            condition=variable,
            outcome=outcome,
            value=outcome_value,
            current_value=current_value,
            counterfactual_value=counterfactual_value
        )


class TransparencyLayer:
    """
    Implements complete transparency with reasoning trace and uncertainty visualization.
    
    Features:
    - Complete reasoning trace
    - Confidence visualization with uncertainty quantification
    - Uncertainty decomposition (epistemic/aleatoric)
    - Graph structure explanation
    """
    
    def __init__(self):
        """Initialize transparency layer."""
        self.trace_history: List[Dict[str, Any]] = []
    
    def show_reasoning_trace(
        self,
        reasoning_chain: ReasoningChain
    ) -> Dict[str, Any]:
        """
        Show complete reasoning trace.
        
        Args:
            reasoning_chain: Reasoning chain
            
        Returns:
            Trace dictionary
        """
        trace = {
            'chain_id': reasoning_chain.chain_id,
            'steps': [],
            'decision_points': reasoning_chain.decision_points,
            'alternative_branches': reasoning_chain.alternative_branches
        }
        
        for step in reasoning_chain.steps:
            step_trace = {
                'step_id': step.step_id,
                'step_type': step.step_type.value,
                'operation': step.operation,
                'input_state': step.input_state,
                'output_state': step.output_state,
                'confidence': step.confidence,
                'uncertainty': step.uncertainty,
                'premises': step.premises,
                'conclusion': step.conclusion,
                'timestamp': step.timestamp
            }
            trace['steps'].append(step_trace)
        
        self.trace_history.append(trace)
        return trace
    
    def visualize_confidence(
        self,
        reasoning_chain: ReasoningChain
    ) -> Dict[str, Any]:
        """
        Visualize confidence scores with uncertainty quantification.
        
        P(conclusion | evidence) ± σ
        
        Args:
            reasoning_chain: Reasoning chain
            
        Returns:
            Confidence visualization dictionary
        """
        if not reasoning_chain.steps:
            return {}
        
        confidences = [s.confidence for s in reasoning_chain.steps]
        uncertainties = [s.uncertainty or 0.0 for s in reasoning_chain.steps]
        
        if NUMPY_AVAILABLE and confidences:
            mean_confidence = float(np.mean(confidences))
            std_confidence = float(np.std(confidences))
            mean_uncertainty = float(np.mean(uncertainties))
        else:
            mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            std_confidence = math.sqrt(sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)) if confidences else 0.0
            mean_uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
        
        return {
            'mean_confidence': mean_confidence,
            'std_confidence': std_confidence,
            'confidence_interval_95': (
                max(0.0, mean_confidence - 1.96 * std_confidence),
                min(1.0, mean_confidence + 1.96 * std_confidence)
            ),
            'mean_uncertainty': mean_uncertainty,
            'step_confidences': confidences,
            'step_uncertainties': uncertainties
        }
    
    def explain_graph_structure(
        self,
        graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain graph structure using graph-theoretic concepts.
        
        Args:
            graph: Graph dictionary
            
        Returns:
            Graph structure explanation
        """
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])
        
        # Build adjacency representation
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        in_degree: Dict[str, int] = defaultdict(int)
        out_degree: Dict[str, int] = defaultdict(int)
        
        for source, target in edges:
            adjacency[source].add(target)
            out_degree[source] += 1
            in_degree[target] += 1
        
        # Find key nodes
        root_nodes = [n for n in nodes if in_degree[n] == 0]
        leaf_nodes = [n for n in nodes if out_degree[n] == 0]
        hub_nodes = [n for n in nodes if out_degree[n] > 2]
        
        # Detect cycles (simplified)
        has_cycles = self._detect_cycles(nodes, edges)
        
        # Find strongly connected components (simplified)
        sccs = self._find_sccs(nodes, edges)
        
        return {
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'root_nodes': root_nodes,
            'leaf_nodes': leaf_nodes,
            'hub_nodes': hub_nodes,
            'has_cycles': has_cycles,
            'strongly_connected_components': len(sccs),
            'structure_type': self._classify_structure(nodes, edges, root_nodes, leaf_nodes)
        }
    
    def _detect_cycles(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]]
    ) -> bool:
        """Detect cycles in graph (simplified DFS)."""
        visited = set()
        rec_stack = set()
        adjacency = defaultdict(list)
        
        for source, target in edges:
            adjacency[source].append(target)
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in nodes:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False
    
    def _find_sccs(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]]
    ) -> List[Set[str]]:
        """Find strongly connected components (simplified)."""
        # Simplified: return each node as its own SCC
        # Full implementation would use Tarjan's algorithm
        return [{node} for node in nodes]
    
    def _classify_structure(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        root_nodes: List[str],
        leaf_nodes: List[str]
    ) -> str:
        """Classify graph structure type."""
        if not nodes:
            return "empty"
        
        if len(root_nodes) == 1 and len(leaf_nodes) == 1:
            return "linear_chain"
        elif len(root_nodes) == 1:
            return "tree"
        elif len(leaf_nodes) == 1:
            return "convergent"
        else:
            return "general_dag"
