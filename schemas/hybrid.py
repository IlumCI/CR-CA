"""
Schema definitions for the hybrid agent system.

Defines data structures for provenance tracking, temporal edges,
language compilation, and error correction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import time


class TemporalType(Enum):
    """Temporal relationship types for causal edges."""
    BEFORE = "before"
    AFTER = "after"
    DELAYED = "delayed"
    FEEDBACK_LOOP = "feedback_loop"
    IMMEDIATE = "immediate"  # Default, no temporal delay


@dataclass
class EdgeProvenance:
    """
    Tracks the provenance of a causal edge - where it came from and how confident we are.
    
    Attributes:
        source_sentence: Original text that created the edge
        extraction_pattern: Pattern that matched (pattern ID/name)
        pattern_confidence: Initial confidence from pattern (0.0-1.0)
        extraction_timestamp: When edge was extracted (Unix timestamp)
        confidence_decay_rate: How confidence decays over time (per day, 0.0-1.0)
        validation_history: List of validations/updates with timestamps
        contradictions: List of contradictory edges found
    """
    source_sentence: str
    extraction_pattern: str
    pattern_confidence: float = 1.0
    extraction_timestamp: float = field(default_factory=time.time)
    confidence_decay_rate: float = 0.01  # 1% per day
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_current_confidence(self) -> float:
        """
        Calculate current confidence accounting for decay.
        
        Returns:
            Current confidence value (0.0-1.0)
        """
        days_elapsed = (time.time() - self.extraction_timestamp) / 86400.0
        decay_factor = (1.0 - self.confidence_decay_rate) ** days_elapsed
        return max(0.0, min(1.0, self.pattern_confidence * decay_factor))
    
    def add_validation(self, validation_type: str, result: bool, notes: str = "") -> None:
        """
        Add a validation entry to the history.
        
        Args:
            validation_type: Type of validation (e.g., "data_check", "expert_review")
            result: Whether validation passed
            notes: Additional notes about the validation
        """
        self.validation_history.append({
            "type": validation_type,
            "result": result,
            "notes": notes,
            "timestamp": time.time()
        })
    
    def add_contradiction(self, contradictory_edge: str, reason: str) -> None:
        """
        Record a contradiction with another edge.
        
        Args:
            contradictory_edge: Identifier of the contradictory edge
            reason: Why this is a contradiction
        """
        self.contradictions.append({
            "edge": contradictory_edge,
            "reason": reason,
            "timestamp": time.time()
        })


@dataclass
class TemporalEdge:
    """
    Represents a temporal causal relationship.
    
    Attributes:
        temporal_type: Type of temporal relationship
        delay: Optional delay duration (in time units, e.g., days)
        decay_function: How effect decays over time (function name or parameters)
        feedback_strength: For feedback loops, the strength of the feedback
    """
    temporal_type: TemporalType = TemporalType.IMMEDIATE
    delay: Optional[float] = None
    decay_function: Optional[str] = None  # e.g., "exponential", "linear", "step"
    decay_params: Dict[str, float] = field(default_factory=dict)
    feedback_strength: float = 1.0
    
    def apply_temporal_decay(self, base_strength: float, time_elapsed: float) -> float:
        """
        Apply temporal decay to a base strength value.
        
        Args:
            base_strength: Base strength of the effect
            time_elapsed: Time elapsed since the cause (in same units as delay)
            
        Returns:
            Decayed strength value
        """
        if self.decay_function is None:
            return base_strength
        
        if self.decay_function == "exponential":
            decay_rate = self.decay_params.get("rate", 0.1)
            return base_strength * (1.0 - decay_rate) ** time_elapsed
        elif self.decay_function == "linear":
            max_time = self.decay_params.get("max_time", 100.0)
            if time_elapsed >= max_time:
                return 0.0
            return base_strength * (1.0 - time_elapsed / max_time)
        elif self.decay_function == "step":
            threshold = self.decay_params.get("threshold", 1.0)
            return base_strength if time_elapsed < threshold else 0.0
        else:
            return base_strength


@dataclass
class AnnotatedToken:
    """
    Represents a token with correction metadata (for error correction pipeline).
    
    Attributes:
        original_form: What user typed
        normalized_form: Corrected version
        confidence: Correction confidence (0.0-1.0)
        correction_type: Type of correction (spelling/abbreviation/inferred)
        provenance: Why correction was made
        metadata: Optional additional metadata (e.g., dictionary info)
    """
    original_form: str
    normalized_form: str
    confidence: float = 1.0
    correction_type: str = "none"  # spelling, abbreviation, inferred, none
    provenance: str = ""
    metadata: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class LexicalGraph:
    """
    Represents a compiled lexical knowledge graph.
    
    Attributes:
        synonym_sets: Dictionary mapping canonical terms to sets of synonyms
        hypernym_chains: Dictionary mapping terms to their hypernym chains
        vocabulary: Set of all known terms
    """
    synonym_sets: Dict[str, Set[str]] = field(default_factory=dict)
    hypernym_chains: Dict[str, List[str]] = field(default_factory=dict)
    vocabulary: set = field(default_factory=set)


@dataclass
class SynonymSet:
    """
    A set of synonymous terms.
    
    Attributes:
        canonical: Canonical form of the term
        synonyms: Set of synonymous terms
    """
    canonical: str
    synonyms: set = field(default_factory=set)


@dataclass
class DependencyTree:
    """
    Represents a dependency parse tree.
    
    Attributes:
        nodes: List of nodes (words/phrases)
        edges: List of (head, dependent, relation) tuples
        root: Root node identifier
    """
    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)
    root: Optional[str] = None


@dataclass
class CausalStructure:
    """
    Represents extracted causal structure from a sentence.
    
    Attributes:
        cause: Cause variable/phrase
        effect: Effect variable/phrase
        relation_type: Type of causal relation
        confidence: Confidence in the extraction
    """
    cause: str
    effect: str
    relation_type: str
    confidence: float = 1.0
