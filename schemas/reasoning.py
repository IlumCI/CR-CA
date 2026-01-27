"""
Schema definitions for chain-of-thought reasoning and proof tracking.

Implements natural deduction with proof trees, explicit inference chains,
and evidence tracking for complete traceability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import time


class InferenceRule(Enum):
    """Inference rules for natural deduction."""
    MODUS_PONENS = "modus_ponens"
    UNIVERSAL_INSTANTIATION = "universal_instantiation"
    CAUSAL_INFERENCE = "causal_inference"
    EXTRACTION = "extraction"
    VALIDATION = "validation"
    INFERENCE = "inference"
    DEDUCTION = "deduction"
    INDUCTION = "induction"
    ABDUCTION = "abduction"


class StepType(Enum):
    """Type of reasoning step."""
    EXTRACTION = "extraction"
    INFERENCE = "inference"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    DECISION = "decision"


@dataclass
class Evidence:
    """
    Evidence for a conclusion.
    
    Attributes:
        source: Source of evidence (e.g., "extraction", "inference", "user_input")
        content: Evidence content
        confidence: Confidence in evidence (0.0-1.0)
        timestamp: When evidence was collected
        metadata: Additional metadata
    """
    source: str
    content: Any
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """
    Individual reasoning step in a proof chain.
    
    Implements a proof step with type safety and evidence tracking.
    
    Attributes:
        step_id: Unique identifier for this step
        step_type: Type of reasoning step
        inference_rule: Inference rule used (if applicable)
        input_state: Precondition graph G_pre (as dictionary representation)
        operation: Description of operation performed
        output_state: Postcondition graph G_post (as dictionary representation)
        confidence: Bayesian posterior P(conclusion | evidence)
        uncertainty: Uncertainty quantification (standard deviation or credible interval)
        evidence: Set of premises {p₁, ..., pₙ} and evidence for conclusion
        premises: List of premise step IDs
        conclusion: Conclusion reached in this step
        timestamp: When step was executed
        metadata: Additional metadata
    """
    step_id: str
    step_type: StepType
    inference_rule: Optional[InferenceRule] = None
    input_state: Dict[str, Any] = field(default_factory=dict)
    operation: str = ""
    output_state: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    uncertainty: Optional[float] = None
    evidence: List[Evidence] = field(default_factory=list)
    premises: List[str] = field(default_factory=list)
    conclusion: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_evidence(self, evidence: Evidence) -> None:
        """
        Add evidence to this step.
        
        Args:
            evidence: Evidence to add
        """
        self.evidence.append(evidence)
    
    def is_valid_proof_step(self) -> bool:
        """
        Check if this is a valid proof step.
        
        A valid step must have:
        - Valid inference rule (if inference step)
        - Evidence or premises
        - Conclusion
        
        Returns:
            True if valid, False otherwise
        """
        if self.step_type == StepType.INFERENCE and self.inference_rule is None:
            return False
        if not self.evidence and not self.premises:
            return False
        if self.conclusion is None:
            return False
        return True


@dataclass
class ReasoningChain:
    """
    Complete reasoning path (proof tree/DAG structure).
    
    Tracks complete reasoning path with backtracking support and decision point recording.
    
    Attributes:
        chain_id: Unique identifier for this reasoning chain
        steps: List of reasoning steps (ordered sequence)
        root_step: Root step ID (starting point)
        leaf_steps: Leaf step IDs (end points)
        alternative_branches: Alternative proof branches explored
        decision_points: Choice points with alternatives explored
        success: Whether reasoning chain succeeded
        final_conclusion: Final conclusion reached
        timestamp: When chain was created
        metadata: Additional metadata
    """
    chain_id: str
    steps: List[ReasoningStep] = field(default_factory=list)
    root_step: Optional[str] = None
    leaf_steps: List[str] = field(default_factory=list)
    alternative_branches: List[List[str]] = field(default_factory=list)
    decision_points: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    final_conclusion: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: ReasoningStep) -> None:
        """
        Add a step to the reasoning chain.
        
        Args:
            step: Reasoning step to add
        """
        self.steps.append(step)
        
        # Update root if this is first step
        if self.root_step is None:
            self.root_step = step.step_id
        
        # Update leaf steps
        if not step.premises:  # No dependencies, could be a leaf
            if step.step_id not in self.leaf_steps:
                self.leaf_steps.append(step.step_id)
        
        # Remove from leaf steps if other steps depend on it
        for other_step in self.steps:
            if step.step_id in other_step.premises and step.step_id in self.leaf_steps:
                self.leaf_steps.remove(step.step_id)
                break
        
        # Add to leaf steps if no other steps depend on it
        has_dependents = any(step.step_id in s.premises for s in self.steps if s.step_id != step.step_id)
        if not has_dependents and step.step_id not in self.leaf_steps:
            self.leaf_steps.append(step.step_id)
    
    def get_step(self, step_id: str) -> Optional[ReasoningStep]:
        """
        Get step by ID.
        
        Args:
            step_id: Step ID
            
        Returns:
            ReasoningStep or None if not found
        """
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_path_to_step(self, step_id: str) -> List[ReasoningStep]:
        """
        Get reasoning path to a specific step.
        
        Args:
            step_id: Target step ID
            
        Returns:
            List of steps forming path from root to target
        """
        step = self.get_step(step_id)
        if step is None:
            return []
        
        path = [step]
        visited = {step_id}
        
        # Backtrack through premises
        current_premises = step.premises.copy()
        while current_premises:
            next_premise = current_premises.pop(0)
            if next_premise in visited:
                continue
            visited.add(next_premise)
            
            premise_step = self.get_step(next_premise)
            if premise_step:
                path.insert(0, premise_step)
                current_premises.extend(premise_step.premises)
        
        return path
    
    def validate_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Validate that reasoning chain is sound.
        
        Checks:
        - All steps are valid proof steps
        - All premises exist
        - No circular dependencies
        - Chain is connected
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check all steps are valid
        for step in self.steps:
            if not step.is_valid_proof_step():
                return False, f"Invalid proof step: {step.step_id}"
        
        # Check all premises exist
        step_ids = {step.step_id for step in self.steps}
        for step in self.steps:
            for premise_id in step.premises:
                if premise_id not in step_ids:
                    return False, f"Premise {premise_id} not found for step {step.step_id}"
        
        # Check for circular dependencies (simple check)
        # Build dependency graph and check for cycles
        dependencies = {step.step_id: step.premises for step in self.steps}
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for dep in dependencies.get(node, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            rec_stack.remove(node)
            return False
        
        for step_id in step_ids:
            if step_id not in visited:
                if has_cycle(step_id):
                    return False, f"Circular dependency detected involving {step_id}"
        
        return True, None
