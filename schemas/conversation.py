"""
Schema definitions for conversation management and context tracking.

Implements data structures for multi-turn conversations with memory networks,
attention mechanisms, and coreference resolution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import time


class MessageRole(Enum):
    """Role of message sender."""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


@dataclass
class ConversationMessage:
    """
    Represents a single message in a conversation.
    
    Attributes:
        role: Role of the message sender (user/agent/system)
        content: Message text content
        timestamp: When message was sent (Unix timestamp)
        metadata: Additional metadata (variables extracted, graph state, etc.)
        message_id: Unique identifier for this message
    """
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate message ID if not provided."""
        if self.message_id is None:
            self.message_id = f"{self.role.value}_{int(self.timestamp * 1000)}"


@dataclass
class VariableMapping:
    """
    Maps variables across conversation turns.
    
    Tracks variable evolution: φ: V_t → V_{t+1}
    
    Attributes:
        source_variable: Variable name at turn t
        target_variable: Variable name at turn t+1
        confidence: Confidence in the mapping (0.0-1.0)
        evidence: Evidence for the mapping (e.g., "same context", "explicit mention")
        turn_from: Source turn number
        turn_to: Target turn number
    """
    source_variable: str
    target_variable: str
    confidence: float = 1.0
    evidence: str = ""
    turn_from: int = 0
    turn_to: int = 0


@dataclass
class GraphSnapshot:
    """
    Immutable snapshot of graph state at a specific turn.
    
    Attributes:
        turn_number: Turn number this snapshot represents
        nodes: Set of node names
        edges: List of (source, target) tuples
        node_attributes: Dictionary of node attributes
        edge_attributes: Dictionary of edge attributes {(source, target): attrs}
        timestamp: When snapshot was created
    """
    turn_number: int
    nodes: Set[str] = field(default_factory=set)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    node_attributes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edge_attributes: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationContext:
    """
    Full conversation state with memory network and attention mechanisms.
    
    Implements episodic memory with exponential decay attention weights.
    
    Attributes:
        conversation_id: Unique identifier for this conversation
        messages: List of conversation messages (temporal ordering)
        graph_snapshots: Graph state snapshots per turn
        variable_mappings: Variable evolution mappings across turns
        current_turn: Current turn number
        attention_weights: Attention weights for each message (computed on demand)
        decay_lambda: Exponential decay parameter for attention
        topic_history: Conversation topic transitions
        reference_resolution: Coreference resolution mappings
    """
    conversation_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    graph_snapshots: Dict[int, GraphSnapshot] = field(default_factory=dict)
    variable_mappings: List[VariableMapping] = field(default_factory=list)
    current_turn: int = 0
    attention_weights: Optional[Dict[int, float]] = None
    decay_lambda: float = 0.1  # Exponential decay parameter
    topic_history: List[Dict[str, Any]] = field(default_factory=list)
    reference_resolution: Dict[str, str] = field(default_factory=dict)
    
    def add_message(self, message: ConversationMessage) -> None:
        """
        Add a message to the conversation.
        
        Args:
            message: Message to add
        """
        self.messages.append(message)
        self.current_turn = len(self.messages)
    
    def get_recent_messages(self, k: int = 10) -> List[ConversationMessage]:
        """
        Get k most recent messages.
        
        Args:
            k: Number of recent messages to retrieve
            
        Returns:
            List of k most recent messages
        """
        return self.messages[-k:] if len(self.messages) > k else self.messages
    
    def compute_attention_weights(self, query: Optional[str] = None) -> Dict[int, float]:
        """
        Compute attention weights for messages using exponential decay.
        
        Attention weights: w_i = exp(-λ·(t-i)) where t is current turn, i is message turn.
        
        Args:
            query: Optional query for attention computation (for future query-based attention)
            
        Returns:
            Dictionary mapping message index to attention weight
        """
        if query is None:
            # Simple exponential decay: w_i = exp(-λ·(t-i))
            weights = {}
            t = len(self.messages)
            import math
            for i, msg in enumerate(self.messages):
                if NUMPY_AVAILABLE:
                    weights[i] = float(np.exp(-self.decay_lambda * (t - i)))
                else:
                    weights[i] = float(math.exp(-self.decay_lambda * (t - i)))
            self.attention_weights = weights
            return weights
        else:
            # Future: Implement query-based attention
            # For now, fall back to exponential decay
            return self.compute_attention_weights(None)
    
    def get_context_window(self, window_size: int = 5) -> List[ConversationMessage]:
        """
        Get context window using attention-weighted selection.
        
        Args:
            window_size: Maximum number of messages to include
            
        Returns:
            List of messages in context window
        """
        weights = self.compute_attention_weights()
        # Sort by attention weight and take top k
        sorted_indices = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in sorted_indices[:window_size]]
        return [self.messages[i] for i in sorted(top_indices)]


# Import numpy for attention computation
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    # Fallback if numpy not available
    import math
    NUMPY_AVAILABLE = False
    np = None
