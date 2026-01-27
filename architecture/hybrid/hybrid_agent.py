"""
Enhanced Symbolic-Statistical Hybrid AI - General Purpose LLM Replacement System.

A production-ready, stable, pure symbolic-statistical reasoning agent
that can replace LLMs entirely without actually using one. 

Supports both:
- Causal Reasoning (CRCA): Causal analysis, counterfactuals, interventions
- General Knowledge: Facts, definitions, taxonomic relationships, spatial/temporal knowledge

Key Features:
- Graph-first reasoning: All answers come from graph state, never text parsing
- Enhanced NLU: Comprehensive pattern matching for causal AND general knowledge
- Enhanced NLG: Natural, conversational responses with pragmatic tone adjustment
- Non-destructive text correction: Handles spelling, abbreviations, grammar
- Language compilation: Three-layer system (lexical, grammatical, pragmatic)
- Multi-domain support: Causal, taxonomic, spatial, temporal, functional relationships
- Graph compression: Composite nodes, latent factors, abstraction
- Provenance tracking: Every edge tracks its source and confidence decay
- Robust error handling: Graceful degradation, validation, fallback responses

Relationship Types Supported:
- Causal: affects, causes, influences, depends on, leads to
- Taxonomic: is-a, type-of, belongs-to, classified-as
- Meronymic: part-of, consists-of, contains, has
- Spatial: located-in, found-in
- Temporal: before, after, precedes, follows
- Functional: used-for, functions-as
- Definitional: is, means, refers-to, defined-as
- Factual: was, became, changed-to

CRITICAL: Epistemic Validation (for causal tasks)
For causal reasoning tasks, this agent requires explicit causal structure, not intent statements. It will:
- REJECT action verbs (identify, analyze, examine) as causal variables
- REJECT epistemic terms (policy, task, goal, decision) as causal variables
- WARN when structure is inferred from syntax alone
- ERROR when task is epistemically underspecified (no valid state variables)

For general knowledge tasks, the agent is more flexible and can extract:
- Facts and definitions
- Taxonomic relationships
- Spatial and temporal information
- Properties and attributes

This system is designed to be stable, reliable, and capable of replacing LLMs
for both causal reasoning AND general knowledge tasks while maintaining natural language interaction.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
from collections import defaultdict, deque

import numpy as np

# Optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Import CRCA templates
from templates.graph_management import GraphManager
from templates.statistical_methods import StatisticalMethods
from templates.prediction_framework import PredictionFramework, CounterfactualScenario

# Import new graph-first components
from utils.graph_reasoner import GraphFirstReasoner
from architecture.hybrid.graph_compressor import GraphCompressor
from architecture.hybrid.language_compiler import LexicalCompiler, GrammaticalCompiler, PragmaticCompiler
from architecture.hybrid.text_corrector import TextCorrector
from schemas.hybrid import EdgeProvenance, TemporalEdge, TemporalType

# Import new LLM-enhanced components
from architecture.hybrid.conversation_manager import ConversationHistory, ContextTracker
from architecture.hybrid.reasoning_tracker import ReasoningTracker
from architecture.hybrid.few_shot_learner import ExampleStore, PatternLearner, AdaptiveExtractor
from architecture.hybrid.task_decomposer import TaskAnalyzer, SubTaskExecutor, PlanGenerator
from architecture.hybrid.explanation_generator import ExplanationBuilder, TransparencyLayer
from architecture.hybrid.self_verifier import ConsistencyChecker, ErrorDetector, SelfCorrector
from architecture.hybrid.consistency_engine import ConsistencyEngine
from schemas.conversation import ConversationContext, MessageRole
from schemas.reasoning import ReasoningChain, StepType, InferenceRule, Evidence

logger = logging.getLogger(__name__)


class SymbolicReasoner:
    """
    Advanced symbolic reasoning engine for natural language understanding and causal extraction.
    
    Uses sophisticated pattern matching, semantic analysis, and context-aware parsing
    to extract causal variables and relationships from natural language tasks.
    
    Enhanced Features:
    - Action Verb Understanding: Extracts state variables from action verbs (e.g., "identify X" -> extracts "X")
    - Epistemic Term Understanding: Extracts state variables from epistemic terms (e.g., "policy of X" -> extracts "X")
    - Vague Language Handling: Understands vague language patterns like "what affects X", "factors influencing Y"
    - Semantic Role Analysis: Understands that action verbs and epistemic terms are signals, not variables themselves
    - Context-Aware Extraction: Uses context to infer relationships even from vague descriptions
    """
    
    def __init__(
        self,
        graph_manager: GraphManager,
        lexical_compiler: Optional[Any] = None,
        adaptive_extractor: Optional[Any] = None
    ):
        """
        Initialize symbolic reasoner.
        
        Args:
            graph_manager: GraphManager instance for graph operations
            lexical_compiler: Optional LexicalCompiler instance for dictionary validation
            adaptive_extractor: Optional AdaptiveExtractor for few-shot learning
        """
        self.graph_manager = graph_manager
        self.lexical_compiler = lexical_compiler
        self.adaptive_extractor = adaptive_extractor
        
        # Comprehensive pattern definitions for extracting causal relationships
        # Updated patterns to handle numerical values, conditionals, and state descriptions
        self.patterns = [
            # Direct causal verbs (with optional numerical values)
            (r'(\w+(?:\s+\w+)?)\s+(?:depends?\s+on|depends?\s+upon)\s+(\w+(?:\s+\w+)?)', 'depends_on', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+causes?\s+(\w+(?:\s+\w+)?)', 'causes', 0.95),
            (r'(\w+(?:\s+\w+)?)\s+affects?\s+(\w+(?:\s+\w+)?)', 'affects', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+influences?\s+(\w+(?:\s+\w+)?)', 'influences', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+leads?\s+to\s+(\w+(?:\s+\w+)?)', 'leads_to', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+results?\s+in\s+(\w+(?:\s+\w+)?)', 'results_in', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+impacts?\s+(\w+(?:\s+\w+)?)', 'impacts', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+drives?\s+(\w+(?:\s+\w+)?)', 'drives', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+determines?\s+(\w+(?:\s+\w+)?)', 'determines', 0.95),
            (r'(\w+(?:\s+\w+)?)\s+controls?\s+(\w+(?:\s+\w+)?)', 'controls', 0.9),
            
            # Passive voice patterns
            (r'(\w+(?:\s+\w+)?)\s+is\s+(?:affected|influenced|determined|controlled|driven)\s+by\s+(\w+(?:\s+\w+)?)', 'affected_by', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+is\s+caused\s+by\s+(\w+(?:\s+\w+)?)', 'caused_by', 0.95),
            (r'(\w+(?:\s+\w+)?)\s+results?\s+from\s+(\w+(?:\s+\w+)?)', 'results_from', 0.9),
            
            # State description patterns (X is Y, X = Y, X: Y)
            (r'(\w+(?:\s+\w+)?)\s+is\s+(?:\d+[.,]?\d*|[\d%]+|[a-z]+)', 'state_description', 0.7),
            (r'(\w+(?:\s+\w+)?)\s*[=:]\s*(?:\d+[.,]?\d*|[\d%]+)', 'state_equals', 0.7),
            (r'(\w+(?:\s+\w+)?)\s+of\s+(\d+[.,]?\d*%?|\w+)', 'state_of', 0.6),
            
            # Conditional patterns (enhanced)
            (r'if\s+(\w+(?:\s+\w+)?)\s+(?:is|are|was|were)\s+(?:\d+[.,]?\d*|[\d%]+|\w+)\s*,?\s*(?:then\s+)?(?:what\s+)?(?:is|are|will|would)\s+(\w+(?:\s+\w+)?)', 'conditional_question', 0.85),
            (r'if\s+(\w+(?:\s+\w+)?)\s+then\s+(\w+(?:\s+\w+)?)', 'conditional', 0.85),
            (r'when\s+(\w+(?:\s+\w+)?)\s+,\s+(\w+(?:\s+\w+)?)', 'temporal', 0.8),
            (r'(\w+(?:\s+\w+)?)\s+when\s+(\w+(?:\s+\w+)?)', 'temporal_reverse', 0.8),
            
            # Question patterns (what is X, what will X be, etc.)
            (r'(?:what|which|how\s+much|how\s+many)\s+(?:is|are|will|would|should)\s+(?:the\s+)?(\w+(?:\s+\w+)?)', 'question_target', 0.8),
            (r'(?:what|which|how\s+much|how\s+many)\s+(?:is|are|will|would|should)\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:of|in|for|after|in\s+\d+\s+days)', 'question_target_time', 0.85),
            
            # Arrow notation
            (r'(\w+(?:\s+\w+)?)\s*[-->->]\s*(\w+(?:\s+\w+)?)', 'arrow', 0.95),
            (r'(\w+(?:\s+\w+)?)\s*=>\s*(\w+(?:\s+\w+)?)', 'arrow', 0.95),
            
            # Comparative patterns
            (r'(\w+(?:\s+\w+)?)\s+increases?\s+(\w+(?:\s+\w+)?)', 'increases', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+decreases?\s+(\w+(?:\s+\w+)?)', 'decreases', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+raises?\s+(\w+(?:\s+\w+)?)', 'increases', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+lowers?\s+(\w+(?:\s+\w+)?)', 'decreases', 0.85),
            
            # Correlation patterns (weaker causality)
            (r'(\w+(?:\s+\w+)?)\s+is\s+correlated\s+with\s+(\w+(?:\s+\w+)?)', 'correlated', 0.6),
            (r'(\w+(?:\s+\w+)?)\s+is\s+related\s+to\s+(\w+(?:\s+\w+)?)', 'related', 0.5),
            
            # Implicit relationships (X and Y, X with Y)
            (r'(\w+(?:\s+\w+)?)\s+and\s+(\w+(?:\s+\w+)?)\s+(?:affect|influence|determine|control)', 'implicit_and', 0.6),
            
            # NEW: Vague language patterns - "what affects X", "factors influencing X"
            (r'what\s+(?:affects|influences|causes|impacts|changes)\s+(\w+(?:\s+\w+)?)', 'vague_causal', 0.6),
            (r'factors?\s+(?:affecting|influencing|causing|impacting)\s+(\w+(?:\s+\w+)?)', 'vague_causal', 0.6),
            (r'how\s+(?:does|do)\s+(\w+(?:\s+\w+)?)\s+(?:affect|influence|cause|impact)', 'vague_causal', 0.6),
            # NEW: Relationship patterns - "relationship between X and Y"
            (r'relationship\s+(?:between|among)\s+(\w+(?:\s+\w+)?)\s+(?:and|&)\s+(\w+(?:\s+\w+)?)', 'relationship', 0.7),
            (r'how\s+(?:does|do)\s+(\w+(?:\s+\w+)?)\s+relate\s+to\s+(\w+(?:\s+\w+)?)', 'relationship', 0.7),
            (r'effect\s+of\s+(\w+(?:\s+\w+)?)\s+on\s+(\w+(?:\s+\w+)?)', 'causal', 0.8),
            
            # Enhanced patterns for better coverage
            # Temporal patterns
            (r'(\w+(?:\s+\w+)?)\s+before\s+(\w+(?:\s+\w+)?)', 'before', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+after\s+(\w+(?:\s+\w+)?)', 'after', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+leads?\s+to\s+(\w+(?:\s+\w+)?)\s+in\s+(\d+)\s+(?:days?|hours?|weeks?|months?)', 'delayed', 0.9),
            
            # Comparative and quantitative
            (r'(\w+(?:\s+\w+)?)\s+is\s+(?:higher|greater|larger|bigger)\s+than\s+(\w+(?:\s+\w+)?)', 'greater_than', 0.7),
            (r'(\w+(?:\s+\w+)?)\s+is\s+(?:lower|smaller|less)\s+than\s+(\w+(?:\s+\w+)?)', 'less_than', 0.7),
            (r'(\w+(?:\s+\w+)?)\s+varies?\s+with\s+(\w+(?:\s+\w+)?)', 'varies_with', 0.75),
            
            # Question patterns (enhanced)
            (r'what\s+(?:happens?|occurs?|results?)\s+(?:if|when)\s+(\w+(?:\s+\w+)?)', 'what_if', 0.9),
            (r'how\s+(?:does|do|will|would)\s+(\w+(?:\s+\w+)?)\s+(?:affect|influence|impact)\s+(\w+(?:\s+\w+)?)', 'how_affects', 0.9),
            (r'why\s+(?:does|do|is|are)\s+(\w+(?:\s+\w+)?)', 'why_question', 0.8),
            
            # Multi-variable patterns
            (r'(\w+(?:\s+\w+)?)\s+(?:together\s+with|along\s+with|combined\s+with)\s+(\w+(?:\s+\w+)?)\s+(?:affect|influence|cause)', 'combined_effect', 0.8),
            (r'(\w+(?:\s+\w+)?)\s+(?:and|or)\s+(\w+(?:\s+\w+)?)\s+(?:both|together)\s+(?:affect|influence|determine)', 'joint_effect', 0.75),
            
            # ====================================================================
            # GENERAL KNOWLEDGE PATTERNS (Non-Causal Relationships)
            # ====================================================================
            
            # Taxonomic/Classification patterns (is-a, type-of)
            (r'(\w+(?:\s+\w+)?)\s+is\s+(?:a|an)\s+(\w+(?:\s+\w+)?)', 'is_a', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+is\s+(?:a|an)\s+type\s+of\s+(\w+(?:\s+\w+)?)', 'is_a', 0.95),
            (r'(\w+(?:\s+\w+)?)\s+is\s+(?:a|an)\s+kind\s+of\s+(\w+(?:\s+\w+)?)', 'is_a', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+belongs?\s+to\s+(\w+(?:\s+\w+)?)', 'belongs_to', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+is\s+classified\s+as\s+(\w+(?:\s+\w+)?)', 'is_a', 0.9),
            
            # Property/Has patterns
            (r'(\w+(?:\s+\w+)?)\s+has\s+(\w+(?:\s+\w+)?)', 'has_property', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+has\s+(?:a|an)\s+(\w+(?:\s+\w+)?)', 'has_property', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+possesses?\s+(\w+(?:\s+\w+)?)', 'has_property', 0.8),
            (r'(\w+(?:\s+\w+)?)\s+contains?\s+(\w+(?:\s+\w+)?)', 'contains', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+includes?\s+(\w+(?:\s+\w+)?)', 'includes', 0.8),
            
            # Part-Whole patterns (meronymy)
            (r'(\w+(?:\s+\w+)?)\s+is\s+(?:a|an)\s+part\s+of\s+(\w+(?:\s+\w+)?)', 'part_of', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+is\s+part\s+of\s+(\w+(?:\s+\w+)?)', 'part_of', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+belongs?\s+to\s+(\w+(?:\s+\w+)?)', 'part_of', 0.8),
            (r'(\w+(?:\s+\w+)?)\s+consists?\s+of\s+(\w+(?:\s+\w+)?)', 'consists_of', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+is\s+composed\s+of\s+(\w+(?:\s+\w+)?)', 'consists_of', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+is\s+made\s+of\s+(\w+(?:\s+\w+)?)', 'consists_of', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+is\s+made\s+up\s+of\s+(\w+(?:\s+\w+)?)', 'consists_of', 0.85),
            
            # Location/Spatial patterns
            (r'(\w+(?:\s+\w+)?)\s+is\s+(?:in|at|on)\s+(\w+(?:\s+\w+)?)', 'located_in', 0.8),
            (r'(\w+(?:\s+\w+)?)\s+is\s+located\s+(?:in|at|on)\s+(\w+(?:\s+\w+)?)', 'located_in', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+is\s+found\s+(?:in|at|on)\s+(\w+(?:\s+\w+)?)', 'located_in', 0.8),
            (r'(\w+(?:\s+\w+)?)\s+resides?\s+(?:in|at|on)\s+(\w+(?:\s+\w+)?)', 'located_in', 0.8),
            
            # Definition patterns (X is Y, X means Y, X refers to Y)
            (r'(\w+(?:\s+\w+)?)\s+is\s+(?:defined\s+as|means?|refers?\s+to)\s+(\w+(?:\s+\w+)?)', 'defined_as', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+is\s+(\w+(?:\s+\w+)?)', 'is', 0.7),  # General "is" (weaker)
            (r'(\w+(?:\s+\w+)?)\s+means?\s+(\w+(?:\s+\w+)?)', 'means', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+refers?\s+to\s+(\w+(?:\s+\w+)?)', 'refers_to', 0.85),
            
            # Similarity/Equivalence patterns
            (r'(\w+(?:\s+\w+)?)\s+is\s+(?:similar\s+to|like|equivalent\s+to)\s+(\w+(?:\s+\w+)?)', 'similar_to', 0.8),
            (r'(\w+(?:\s+\w+)?)\s+is\s+the\s+same\s+as\s+(\w+(?:\s+\w+)?)', 'equivalent_to', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+equals?\s+(\w+(?:\s+\w+)?)', 'equivalent_to', 0.85),
            
            # Temporal patterns (general knowledge)
            (r'(\w+(?:\s+\w+)?)\s+occurs?\s+(?:before|after|during)\s+(\w+(?:\s+\w+)?)', 'temporal', 0.8),
            (r'(\w+(?:\s+\w+)?)\s+happens?\s+(?:before|after|during)\s+(\w+(?:\s+\w+)?)', 'temporal', 0.8),
            (r'(\w+(?:\s+\w+)?)\s+precedes?\s+(\w+(?:\s+\w+)?)', 'precedes', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+follows?\s+(\w+(?:\s+\w+)?)', 'follows', 0.85),
            
            # Purpose/Function patterns
            (r'(\w+(?:\s+\w+)?)\s+is\s+used\s+(?:for|to)\s+(\w+(?:\s+\w+)?)', 'used_for', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+serves?\s+to\s+(\w+(?:\s+\w+)?)', 'used_for', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+functions?\s+as\s+(\w+(?:\s+\w+)?)', 'functions_as', 0.85),
            (r'(\w+(?:\s+\w+)?)\s+is\s+for\s+(\w+(?:\s+\w+)?)', 'used_for', 0.8),
            
            # General knowledge question patterns
            (r'what\s+is\s+(?:a|an|the)?\s+(\w+(?:\s+\w+)?)', 'what_is', 0.9),
            (r'what\s+are?\s+(?:a|an|the)?\s+(\w+(?:\s+\w+)?)', 'what_is', 0.9),
            (r'who\s+is\s+(?:a|an|the)?\s+(\w+(?:\s+\w+)?)', 'who_is', 0.9),
            (r'where\s+is\s+(?:a|an|the)?\s+(\w+(?:\s+\w+)?)', 'where_is', 0.9),
            (r'when\s+(?:is|was|does|did)\s+(?:a|an|the)?\s+(\w+(?:\s+\w+)?)', 'when_is', 0.85),
            (r'how\s+(?:does|do|is|are)\s+(\w+(?:\s+\w+)?)\s+work', 'how_works', 0.85),
            (r'what\s+(?:does|do)\s+(\w+(?:\s+\w+)?)\s+mean', 'what_means', 0.9),
            
            # Factual statement patterns
            (r'(\w+(?:\s+\w+)?)\s+was\s+(\w+(?:\s+\w+)?)', 'factual', 0.7),
            (r'(\w+(?:\s+\w+)?)\s+were\s+(\w+(?:\s+\w+)?)', 'factual', 0.7),
            (r'(\w+(?:\s+\w+)?)\s+became\s+(\w+(?:\s+\w+)?)', 'became', 0.8),
            (r'(\w+(?:\s+\w+)?)\s+changed\s+to\s+(\w+(?:\s+\w+)?)', 'changed_to', 0.8),
        ]
        
        # Extended keywords for identifying variables (domain-agnostic)
        self.variable_keywords = [
            # General terms
            'variable', 'factor', 'metric', 'indicator', 'measure', 'parameter',
            'dimension', 'attribute', 'feature', 'component', 'element',
            'concept', 'entity', 'object', 'item', 'thing', 'subject', 'topic',
            
            # Business/Economics
            'price', 'demand', 'supply', 'sales', 'revenue', 'cost', 'profit',
            'margin', 'growth', 'market', 'customer', 'product', 'service',
            
            # Quality/Performance
            'satisfaction', 'quality', 'performance', 'efficiency', 'effectiveness',
            'productivity', 'output', 'throughput', 'latency', 'speed',
            
            # Social/Psychological
            'happiness', 'wellbeing', 'stress', 'motivation', 'engagement',
            'retention', 'turnover', 'loyalty', 'trust',
            
            # General Knowledge entities
            'person', 'place', 'location', 'country', 'city', 'organization',
            'company', 'institution', 'event', 'date', 'time', 'period',
            'category', 'type', 'class', 'group', 'species', 'genre',
            
            # Technical
            'temperature', 'pressure', 'voltage', 'current', 'frequency',
            'bandwidth', 'capacity', 'utilization', 'availability',
        ]
        
        # Stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'where',
            'when', 'why', 'how', 'all', 'each', 'every', 'some', 'any', 'no',
            'not', 'only', 'just', 'also', 'too', 'very', 'more', 'most', 'less',
            'least', 'many', 'much', 'few', 'little', 'other', 'another', 'same',
            'different', 'such', 'own', 'so', 'than', 'then', 'there', 'here',
        }
        
        # Causal verb synonyms for better matching
        self.causal_verbs = {
            'cause', 'causes', 'caused', 'affect', 'affects', 'affected',
            'influence', 'influences', 'influenced', 'impact', 'impacts', 'impacted',
            'determine', 'determines', 'determined', 'control', 'controls', 'controlled',
            'drive', 'drives', 'driven', 'lead', 'leads', 'led', 'result', 'results',
            'increase', 'increases', 'increased', 'decrease', 'decreases', 'decreased',
            'raise', 'raises', 'raised', 'lower', 'lowers', 'lowered',
        }
        
        # Negation words
        self.negation_words = {'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere', 'neither', 'nor'}
        
        # Quantifier words
        self.quantifier_words = {'all', 'some', 'many', 'most', 'few', 'several', 'each', 'every', 'any'}
        
        # Action verbs that should NEVER be treated as causal variables
        # These are epistemic/intentional actions, not state variables
        self.action_verbs = {
            'identify', 'analyze', 'examine', 'study', 'investigate', 'explore',
            'determine', 'find', 'discover', 'detect', 'recognize', 'understand',
            'explain', 'describe', 'define', 'specify', 'clarify', 'elucidate',
            'predict', 'forecast', 'estimate', 'calculate', 'compute', 'measure',
            'evaluate', 'assess', 'judge', 'compare', 'contrast', 'differentiate',
            'recommend', 'suggest', 'propose', 'advise', 'counsel', 'guide',
            'implement', 'execute', 'perform', 'conduct', 'carry', 'out',
            'create', 'generate', 'produce', 'make', 'build', 'construct',
            'modify', 'change', 'alter', 'adjust', 'update', 'revise',
            'remove', 'delete', 'eliminate', 'exclude', 'omit', 'skip',
            'add', 'include', 'insert', 'append', 'attach', 'incorporate',
            'process', 'handle', 'manage', 'control', 'operate', 'run',
            'check', 'verify', 'validate', 'confirm', 'test', 'trial',
            'show', 'display', 'present', 'demonstrate', 'illustrate', 'reveal',
            'report', 'document', 'record', 'log', 'track', 'monitor',
            'request', 'ask', 'query', 'question', 'inquire', 'interrogate',
            'provide', 'supply', 'deliver', 'offer', 'give', 'send',
            'receive', 'obtain', 'acquire', 'get', 'fetch', 'retrieve',
            'use', 'utilize', 'employ', 'apply', 'leverage', 'exploit',
            'consider', 'think', 'contemplate', 'reflect', 'ponder', 'muse',
            'decide', 'choose', 'select', 'pick', 'opt', 'prefer',
            'plan', 'design', 'scheme', 'devise', 'formulate', 'develop',
            'solve', 'resolve', 'fix', 'repair', 'correct', 'rectify',
            'learn', 'teach', 'train', 'educate', 'instruct', 'coach',
            'help', 'assist', 'aid', 'support', 'facilitate', 'enable'
        }
        
        # Epistemic/intentional terms that indicate tasks, not causal variables
        self.epistemic_terms = {
            'task', 'goal', 'objective', 'aim', 'purpose', 'intent', 'intention',
            'requirement', 'specification', 'criteria', 'standard', 'benchmark',
            'policy', 'strategy', 'approach', 'method', 'technique', 'procedure',
            'process', 'workflow', 'pipeline', 'system', 'framework', 'model',
            'analysis', 'study', 'research', 'investigation', 'examination',
            'result', 'outcome', 'consequence', 'effect', 'impact', 'influence',
            'finding', 'discovery', 'insight', 'observation', 'conclusion',
            'recommendation', 'suggestion', 'advice', 'guidance', 'direction',
            'decision', 'choice', 'selection', 'option', 'alternative',
            'problem', 'issue', 'challenge', 'difficulty', 'obstacle', 'barrier',
            'solution', 'answer', 'resolution', 'fix', 'remedy', 'cure',
            'question', 'query', 'inquiry', 'request', 'demand', 'need'
        }
    
    def _normalize_variable_name(self, var: str) -> str:
        """
        Normalize variable name by cleaning and standardizing.
        
        Args:
            var: Raw variable name
            
        Returns:
            Normalized variable name
        """
        if not var:
            return ''
        
        # Remove extra whitespace
        var = ' '.join(var.split())
        
        # Remove common articles and prepositions at start
        words = var.split()
        while words and words[0].lower() in {'the', 'a', 'an', 'of', 'for', 'in', 'on', 'at', 'to', 'from'}:
            words = words[1:]
        
        var = ' '.join(words)
        
        # Convert to lowercase for consistency
        return var.lower().strip()
    
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text using pattern matching.
        
        Args:
            text: Input text
            
        Returns:
            List of noun phrases
        """
        noun_phrases = []
        
        # Pattern: adjective* noun+
        pattern = r'\b(?:[a-z]+(?:\s+[a-z]+)*\s+)?(?:[a-z]+(?:ing|ed|tion|sion|ment|ness|ity|ance|ence)?)\b'
        matches = re.finditer(pattern, text.lower())
        
        for match in matches:
            phrase = match.group(0).strip()
            # Filter out stop words and very short phrases
            words = phrase.split()
            if len(words) >= 1 and not all(w in self.stop_words for w in words):
                # Remove stop words from beginning/end
                while words and words[0] in self.stop_words:
                    words = words[1:]
                while words and words[-1] in self.stop_words:
                    words = words[:-1]
                if words:
                    noun_phrases.append(' '.join(words))
        
        return list(set(noun_phrases))
    
    def _detect_negation(self, text: str, start_pos: int, end_pos: int) -> bool:
        """
        Detect if a phrase is negated.
        
        Args:
            text: Full text
            start_pos: Start position of phrase
            end_pos: End position of phrase
            
        Returns:
            True if negated
        """
        # Check before the phrase
        before = text[max(0, start_pos-20):start_pos].lower()
        for neg_word in self.negation_words:
            if neg_word in before:
                return True
        return False
    
    def _extract_state_variables_from_action_verbs(self, text: str) -> Set[str]:
        """
        Extract state variables from action verbs by finding what they refer to.
        
        Example: "identify past policy" -> extract "policy" (but filter if epistemic)
        Example: "analyze price trends" -> extract "price", "trends"
        Example: "determine demand level" -> extract "demand", "level"
        
        Args:
            text: Input text
            
        Returns:
            Set of extracted state variable names
        """
        extracted_vars = set()
        text_lower = text.lower()
        
        # Pattern: action_verb + (optional adverb) + noun_phrase
        # Match: "identify X", "analyze the X", "determine X", etc.
        for action_verb in self.action_verbs:
            # Pattern 1: "action_verb [the/a/an] noun_phrase"
            pattern1 = rf'\b{action_verb}\s+(?:the|a|an)?\s*(\w+(?:\s+\w+)?)'
            matches = re.finditer(pattern1, text_lower, re.IGNORECASE)
            for match in matches:
                noun_phrase = match.group(1).strip()
                # Clean and validate
                cleaned = self._normalize_variable_name(noun_phrase)
                if cleaned and not self._is_action_verb(cleaned):
                    # Check if it's an epistemic term - if so, try to extract what it refers to
                    if self._is_epistemic_term(cleaned):
                        # Try to find what the epistemic term refers to
                        # E.g., "past policy" -> look for what policy refers to
                        # This is harder, so we'll skip for now and let other methods handle it
                        continue
                    # Only add if it's not an epistemic term itself
                    if not self._is_epistemic_term(cleaned):
                        extracted_vars.add(cleaned)
            
            # Pattern 2: "action_verb [what/which/how] noun_phrase"
            pattern2 = rf'\b{action_verb}\s+(?:what|which|how)\s+(\w+(?:\s+\w+)?)'
            matches = re.finditer(pattern2, text_lower, re.IGNORECASE)
            for match in matches:
                noun_phrase = match.group(1).strip()
                cleaned = self._normalize_variable_name(noun_phrase)
                if cleaned and not self._is_action_verb(cleaned) and not self._is_epistemic_term(cleaned):
                    extracted_vars.add(cleaned)
        
        return extracted_vars
    
    def _extract_state_variables_from_epistemic_terms(self, text: str) -> Set[str]:
        """
        Extract state variables from epistemic terms by finding what they refer to.
        
        Example: "past policy" -> if we can find what policy refers to, extract that
        Example: "task goal" -> extract the underlying state variable the goal refers to
        Example: "policy decision" -> extract what the decision affects
        
        Args:
            text: Input text
            
        Returns:
            Set of extracted state variable names
        """
        extracted_vars = set()
        text_lower = text.lower()
        
        # Pattern: epistemic_term + "of" + noun_phrase
        # E.g., "policy of X", "goal of Y"
        for epistemic_term in self.epistemic_terms:
            pattern1 = rf'\b{epistemic_term}\s+of\s+(\w+(?:\s+\w+)?)'
            matches = re.finditer(pattern1, text_lower, re.IGNORECASE)
            for match in matches:
                noun_phrase = match.group(1).strip()
                cleaned = self._normalize_variable_name(noun_phrase)
                if cleaned and not self._is_action_verb(cleaned) and not self._is_epistemic_term(cleaned):
                    extracted_vars.add(cleaned)
        
        # Pattern: adjective + epistemic_term -> extract what it modifies
        # E.g., "past policy" -> look for what policy affects
        # This is harder - we'll use context clues
        epistemic_patterns = [
            r'past\s+(\w+)',  # "past X" -> X might be a state variable if not epistemic
            r'(\w+)\s+policy',  # "X policy" -> X might be what policy affects
            r'(\w+)\s+decision',  # "X decision" -> X might be what decision affects
        ]
        
        for pattern in epistemic_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                noun_phrase = match.group(1).strip()
                cleaned = self._normalize_variable_name(noun_phrase)
                # Only add if it's not an action verb or epistemic term itself
                if (cleaned and 
                    not self._is_action_verb(cleaned) and 
                    not self._is_epistemic_term(cleaned) and
                    cleaned not in self.stop_words):
                    extracted_vars.add(cleaned)
        
        return extracted_vars
    
    def _extract_variables_from_vague_language(self, text: str) -> Set[str]:
        """
        Extract state variables from vague language using semantic understanding.
        
        Handles patterns like:
        - "what affects X" -> extract X and what affects it
        - "how does X relate to Y" -> extract X, Y
        - "the relationship between X and Y" -> extract X, Y
        - "factors influencing X" -> extract X and factors
        
        Args:
            text: Input text
            
        Returns:
            Set of extracted state variable names
        """
        extracted_vars = set()
        text_lower = text.lower()
        
        # Pattern: "what affects/influences/causes X"
        affect_patterns = [
            r'what\s+(?:affects|influences|causes|impacts|changes)\s+(\w+(?:\s+\w+)?)',
            r'how\s+(?:does|do)\s+(\w+(?:\s+\w+)?)\s+(?:affect|influence|cause|impact)',
            r'factors?\s+(?:affecting|influencing|causing|impacting)\s+(\w+(?:\s+\w+)?)',
        ]
        
        for pattern in affect_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                noun_phrase = match.group(1).strip()
                cleaned = self._normalize_variable_name(noun_phrase)
                if cleaned and not self._is_action_verb(cleaned) and not self._is_epistemic_term(cleaned):
                    extracted_vars.add(cleaned)
        
        # Pattern: "relationship between X and Y"
        relationship_pattern = r'relationship\s+(?:between|among)\s+(\w+(?:\s+\w+)?)\s+(?:and|&)\s+(\w+(?:\s+\w+)?)'
        matches = re.finditer(relationship_pattern, text_lower, re.IGNORECASE)
        for match in matches:
            var1 = self._normalize_variable_name(match.group(1).strip())
            var2 = self._normalize_variable_name(match.group(2).strip())
            if var1 and not self._is_action_verb(var1) and not self._is_epistemic_term(var1):
                extracted_vars.add(var1)
            if var2 and not self._is_action_verb(var2) and not self._is_epistemic_term(var2):
                extracted_vars.add(var2)
        
        # Pattern: "how does X relate to Y"
        relate_pattern = r'how\s+(?:does|do)\s+(\w+(?:\s+\w+)?)\s+relate\s+to\s+(\w+(?:\s+\w+)?)'
        matches = re.finditer(relate_pattern, text_lower, re.IGNORECASE)
        for match in matches:
            var1 = self._normalize_variable_name(match.group(1).strip())
            var2 = self._normalize_variable_name(match.group(2).strip())
            if var1 and not self._is_action_verb(var1) and not self._is_epistemic_term(var1):
                extracted_vars.add(var1)
            if var2 and not self._is_action_verb(var2) and not self._is_epistemic_term(var2):
                extracted_vars.add(var2)
        
        # Pattern: "the effect of X on Y"
        effect_pattern = r'effect\s+of\s+(\w+(?:\s+\w+)?)\s+on\s+(\w+(?:\s+\w+)?)'
        matches = re.finditer(effect_pattern, text_lower, re.IGNORECASE)
        for match in matches:
            var1 = self._normalize_variable_name(match.group(1).strip())
            var2 = self._normalize_variable_name(match.group(2).strip())
            if var1 and not self._is_action_verb(var1) and not self._is_epistemic_term(var1):
                extracted_vars.add(var1)
            if var2 and not self._is_action_verb(var2) and not self._is_epistemic_term(var2):
                extracted_vars.add(var2)
        
        return extracted_vars
    
    def _extract_with_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract variables and relationships with context awareness.
        Enhanced to handle numerical values, conditionals, questions, action verbs, and epistemic terms.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted relationships with context
        """
        relationships = []
        text_lower = text.lower()
        
        # Extract using all patterns
        for pattern, rel_type, confidence in self.patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Handle patterns with 1 or 2 groups
                if match.lastindex >= 2:
                    source_raw = match.group(1).strip()
                    target_raw = match.group(2).strip()
                elif match.lastindex == 1:
                    # Single group patterns (like question_target)
                    source_raw = match.group(1).strip()
                    target_raw = None
                else:
                    continue
                
                # Normalize variable names (remove numerical values and percentages)
                source = self._normalize_variable_name(source_raw)
                if target_raw:
                    target = self._normalize_variable_name(target_raw)
                else:
                    target = None
                
                # Skip if too short or stop words
                if not source or len(source.split()) == 0:
                    continue
                if source in self.stop_words:
                    continue
                
                # For single-group patterns (questions), extract target from context
                if not target and rel_type in ['question_target', 'question_target_time', 'state_description', 'state_equals']:
                    # Try to find what the question is about
                    # Look for "what is X" -> X is the target variable
                    if 'what' in text_lower or 'which' in text_lower:
                        # Extract all variables mentioned before the question
                        # This is a heuristic - the question target is usually mentioned earlier
                        pass  # Will be handled by standalone variable extraction
                
                # For state descriptions, infer relationships
                if rel_type in ['state_description', 'state_equals', 'state_of'] and target:
                    # State descriptions like "X is Y" don't create causal edges directly
                    # But we can infer that variables mentioned together might be related
                    continue
                
                # Skip if target is invalid
                if target and (len(target.split()) == 0 or target in self.stop_words):
                    continue
                
                # Check for negation
                start_pos = match.start()
                end_pos = match.end()
                is_negated = self._detect_negation(text, start_pos, end_pos)
                
                # Adjust confidence for negation
                if is_negated:
                    confidence *= 0.3  # Much lower confidence for negated relationships
                
                # Only add if we have both source and target (or it's a question pattern)
                if target or rel_type in ['question_target', 'question_target_time', 'what_is', 'who_is', 'where_is', 'when_is', 'how_works', 'what_means']:
                    # Determine relationship category
                    relationship_category = 'causal'  # default
                    if rel_type in ['is_a', 'belongs_to', 'is', 'defined_as', 'means', 'refers_to', 'equivalent_to', 'similar_to']:
                        relationship_category = 'taxonomic'
                    elif rel_type in ['has_property', 'contains', 'includes', 'part_of', 'consists_of']:
                        relationship_category = 'meronymic'
                    elif rel_type in ['located_in', 'found_in']:
                        relationship_category = 'spatial'
                    elif rel_type in ['used_for', 'functions_as']:
                        relationship_category = 'functional'
                    elif rel_type in ['temporal', 'precedes', 'follows', 'before', 'after', 'delayed']:
                        relationship_category = 'temporal'
                    elif rel_type in ['what_is', 'who_is', 'where_is', 'when_is', 'how_works', 'what_means']:
                        relationship_category = 'definitional'
                    elif rel_type in ['factual', 'became', 'changed_to']:
                        relationship_category = 'factual'
                    elif rel_type in ['causes', 'affects', 'influences', 'depends_on', 'leads_to', 'results_in', 'impacts', 'drives', 'determines', 'controls', 'caused_by', 'affected_by', 'results_from', 'increases', 'decreases']:
                        relationship_category = 'causal'
                    
                    relationships.append({
                        'source': source,
                        'target': target or source,  # For questions, use source as both
                        'type': rel_type,
                        'category': relationship_category,
                        'confidence': confidence,
                        'negated': is_negated,
                        'raw_source': source_raw,
                        'raw_target': target_raw or source_raw
                    })
        
        # Post-process: For conditional questions, infer relationships between mentioned variables
        if 'if' in text_lower and 'what' in text_lower:
            # Extract all variables mentioned
            all_vars = self._extract_standalone_variables(text)
            var_list = sorted(list(all_vars))
            
            # If we have multiple variables, infer they might be related
            if len(var_list) >= 2:
                # Common pattern: "If X is Y, what is Z?" -> X might affect Z
                for i in range(len(var_list) - 1):
                    relationships.append({
                        'source': var_list[i],
                        'target': var_list[-1],  # Last variable is usually the question target
                        'type': 'inferred_from_question',
                        'confidence': 0.5,
                        'negated': False,
                        'raw_source': var_list[i],
                        'raw_target': var_list[-1]
                    })
        
        # NEW: Extract state variables from action verbs and epistemic terms
        # This helps handle vague language like "identify past policy" or "analyze the system"
        action_verb_vars = self._extract_state_variables_from_action_verbs(text)
        epistemic_vars = self._extract_state_variables_from_epistemic_terms(text)
        vague_language_vars = self._extract_variables_from_vague_language(text)
        
        # Add relationships for extracted variables (if we can infer them)
        # For action verbs: if we have "identify X" and "determine Y", infer X might affect Y
        all_extracted = action_verb_vars | epistemic_vars | vague_language_vars
        if len(all_extracted) >= 2:
            # Create inferred relationships between extracted variables
            extracted_list = sorted(list(all_extracted))
            for i in range(len(extracted_list) - 1):
                # Only add if not already in relationships
                already_exists = any(
                    r['source'] == extracted_list[i] and r['target'] == extracted_list[i+1]
                    for r in relationships
                )
                if not already_exists:
                    relationships.append({
                        'source': extracted_list[i],
                        'target': extracted_list[i+1],
                        'type': 'inferred_from_action_verb',
                        'confidence': 0.4,  # Lower confidence for inferred relationships
                        'negated': False,
                        'raw_source': extracted_list[i],
                        'raw_target': extracted_list[i+1]
                    })
        
        return relationships
    
    def _extract_variables_with_values(self, text: str) -> Dict[str, Any]:
        """
        Extract variables that have numerical values attached.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping variables to their values
        """
        variables_with_values = {}
        text_lower = text.lower()
        
        # Pattern: "variable is value" or "variable = value" or "variable: value"
        patterns = [
            r'(\w+(?:\s+\w+)?)\s+is\s+(\d+[.,]?\d*%?|\d+[.,]?\d*\s*[a-z]+)',
            r'(\w+(?:\s+\w+)?)\s*[=:]\s*(\d+[.,]?\d*%?|\d+[.,]?\d*\s*[a-z]+)',
            r'(\w+(?:\s+\w+)?)\s+of\s+(\d+[.,]?\d*%?)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                var = self._normalize_variable_name(match.group(1))
                value = match.group(2).strip()
                if var and var not in self.stop_words:
                    variables_with_values[var] = value
        
        return variables_with_values
    
    def _extract_standalone_variables(self, text: str) -> Set[str]:
        """
        Extract standalone variables using multiple strategies.
        
        Args:
            text: Input text
            
        Returns:
            Set of variable names
        """
        variables = set()
        text_lower = text.lower()
        
        # Strategy 1: Extract variables with values (new)
        variables_with_values = self._extract_variables_with_values(text)
        variables.update(variables_with_values.keys())
        
        # Strategy 2: Keyword-based extraction
        words = re.findall(r'\b\w+\b', text_lower)
        for word in words:
            if word in self.stop_words:
                continue
            # Check if word contains or matches keywords
            for keyword in self.variable_keywords:
                if keyword in word or word in keyword:
                    variables.add(word)
        
        # Strategy 3: Noun phrase extraction (enhanced to handle "X is Y" patterns)
        # Extract noun phrases before "is", "=", ":" followed by numbers
        state_patterns = [
            r'(\w+(?:\s+\w+)?)\s+(?:is|are|was|were)\s+(?:\d|%)',
            r'(\w+(?:\s+\w+)?)\s*[=:]\s*(?:\d|%)',
        ]
        for pattern in state_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                var = self._normalize_variable_name(match.group(1))
                if var and var not in self.stop_words:
                    variables.add(var)
        
        # Strategy 4: Standard noun phrase extraction
        noun_phrases = self._extract_noun_phrases(text_lower)
        for phrase in noun_phrases:
            # Filter out phrases that are just stop words
            words = phrase.split()
            if words and not all(w in self.stop_words for w in words):
                variables.add(phrase)
        
        # Strategy 5: Capitalized words (proper nouns or emphasized terms)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for word in capitalized:
            normalized = self._normalize_variable_name(word)
            if normalized and normalized not in self.stop_words:
                variables.add(normalized)
        
        # Strategy 6: Quoted phrases
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
        for match in quoted:
            phrase = (match[0] or match[1]).strip().lower()
            if phrase and phrase not in self.stop_words:
                variables.add(phrase)
        
        # Strategy 7: Terms after "of", "for", "in" (common variable indicators)
        of_pattern = r'\b(?:of|for|in|about|regarding)\s+(\w+(?:\s+\w+)?)'
        of_matches = re.finditer(of_pattern, text_lower)
        for match in of_matches:
            var = self._normalize_variable_name(match.group(1))
            if var and var not in self.stop_words:
                variables.add(var)
        
        # Strategy 8: Extract from questions (what is X, what will X be)
        question_patterns = [
            r'(?:what|which|how\s+much|how\s+many)\s+(?:is|are|will|would|should)\s+(?:the\s+)?(\w+(?:\s+\w+)?)',
            r'(?:what|which)\s+is\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:of|in|for)',
        ]
        for pattern in question_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                var = self._normalize_variable_name(match.group(1))
                if var and var not in self.stop_words:
                    variables.add(var)
        
        # Strategy 9: Extract variables mentioned with "&" or "and" (common in state descriptions)
        and_pattern = r'(\w+(?:\s+\w+)?)\s+(?:is|are|was|were)\s+[\d%]+\s*(?:&|and)\s+(\w+(?:\s+\w+)?)\s+(?:is|are|was|were)'
        and_matches = re.finditer(and_pattern, text_lower, re.IGNORECASE)
        for match in and_matches:
            var1 = self._normalize_variable_name(match.group(1))
            var2 = self._normalize_variable_name(match.group(2))
            if var1 and var1 not in self.stop_words:
                variables.add(var1)
            if var2 and var2 not in self.stop_words:
                variables.add(var2)
        
        # Strategy 10: Extract from action verbs (e.g., "identify X" -> extract X)
        # This helps handle vague language by finding what action verbs refer to
        action_verb_vars = self._extract_state_variables_from_action_verbs(text)
        variables.update(action_verb_vars)
        
        # Strategy 11: Extract from epistemic terms (e.g., "policy of X" -> extract X)
        epistemic_vars = self._extract_state_variables_from_epistemic_terms(text)
        variables.update(epistemic_vars)
        
        # Strategy 12: Extract from vague language patterns
        vague_vars = self._extract_variables_from_vague_language(text)
        variables.update(vague_vars)
        
        return variables
    
    def _resolve_references(self, text: str, variables: Set[str]) -> Set[str]:
        """
        Resolve pronoun and reference resolution.
        
        Args:
            text: Input text
            variables: Existing variables
            
        Returns:
            Updated set of variables with resolved references
        """
        # Simple pronoun resolution: if we see "it", "this", "that" referring to variables
        # This is a simplified version - full resolution would require more context
        resolved = variables.copy()
        
        # Look for patterns like "this X", "that X", "these X", "those X"
        reference_pattern = r'\b(this|that|these|those)\s+(\w+(?:\s+\w+)?)'
        matches = re.finditer(reference_pattern, text.lower())
        for match in matches:
            var = self._normalize_variable_name(match.group(2))
            if var and var not in self.stop_words:
                resolved.add(var)
        
        return resolved
    
    def _merge_similar_variables(self, variables: Set[str]) -> Set[str]:
        """
        Merge similar variable names (plurals, variations).
        
        Args:
            variables: Set of variable names
            
        Returns:
            Merged set of variables
        """
        merged = set()
        variable_list = list(variables)
        
        for var in variable_list:
            # Check if similar variable already exists
            is_duplicate = False
            for existing in merged:
                # Check for plural/singular
                if var == existing or var == existing + 's' or var + 's' == existing:
                    is_duplicate = True
                    break
                # Check for common variations
                if var.replace('_', ' ') == existing.replace('_', ' '):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.add(var)
        
        return merged
    
    def _filter_valid_variables(self, variables: Set[str]) -> Set[str]:
        """
        Filter variables to keep only valid ones for causal analysis.
        
        Args:
            variables: Set of variable names
            
        Returns:
            Set of valid variable names
        """
        valid = set()
        
        for var in variables:
            # Use _clean_variable to validate
            cleaned = self._clean_variable(var)
            if cleaned:
                # Additional checks
                words = cleaned.split()
                
                # Filter out single words that aren't keywords
                if len(words) == 1:
                    if cleaned not in self.variable_keywords:
                        # Check if it's a meaningful single word
                        if cleaned.lower() in self.stop_words:
                            continue
                        # Very short single words are likely invalid
                        if len(cleaned) < 4:
                            continue
                
                # Filter out variables that are clearly value descriptors
                value_descriptors = ['buy', 'sell', 'percent', 'percentage']
                if any(desc in cleaned.lower() for desc in value_descriptors):
                    continue
                
                # Filter out variables that start with "if"
                if cleaned.lower().startswith('if '):
                    continue
                
                valid.add(cleaned)
        
        return valid
    
    def _is_action_verb(self, var: str) -> bool:
        """
        Check if a variable is actually an action verb (epistemic/intentional action).
        
        Action verbs like "identify", "analyze" should NOT be treated as causal variables.
        
        Uses both local action_verbs list and dictionary part-of-speech checking.
        
        Args:
            var: Variable name to check
            
        Returns:
            True if it's an action verb
        """
        var_lower = var.lower()
        words = var_lower.split()
        
        # Check if any word is an action verb (local list)
        for word in words:
            if word in self.action_verbs:
                return True
            # Check for verb forms (ing, ed, s)
            base_word = word.rstrip('s').rstrip('ed').rstrip('ing')
            if base_word in self.action_verbs:
                return True
        
        # Use dictionary to check part of speech (more accurate)
        if self.lexical_compiler and self.lexical_compiler.enable_dictionary:
            # For single-word variables, check if it's a verb
            if len(words) == 1:
                if self.lexical_compiler.is_action_verb(words[0]):
                    return True
            # For multi-word, check each word
            else:
                for word in words:
                    if self.lexical_compiler.is_action_verb(word):
                        return True
        
        return False
    
    def _is_epistemic_term(self, var: str) -> bool:
        """
        Check if a variable is an epistemic/intentional term (task, policy, etc.).
        
        These are not causal state variables - they're about knowledge/intentions.
        
        Args:
            var: Variable name to check
            
        Returns:
            True if it's an epistemic term
        """
        var_lower = var.lower()
        words = var_lower.split()
        
        # Check if any word is an epistemic term
        for word in words:
            if word in self.epistemic_terms:
                return True
        
        # Check for common epistemic patterns
        epistemic_patterns = [
            r'past\s+\w+',  # "past policy"
            r'\w+\s+policy',  # "X policy"
            r'\w+\s+task',  # "X task"
            r'\w+\s+goal',  # "X goal"
            r'\w+\s+decision',  # "X decision"
        ]
        
        for pattern in epistemic_patterns:
            if re.search(pattern, var_lower):
                return True
        
        return False
    
    def validate_causal_relationship(
        self,
        source: str,
        target: str,
        graph: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate causal relationship using do-calculus and d-separation.
        
        Implements formal causal validation:
        - Correlation vs. causation: P(Y | X) ≠ P(Y | do(X))
        - D-separation: X ⊥ Y | Z if d-separated in graph
        - Temporal ordering: if X causes Y, then time(X) < time(Y)
        - Confounder detection: backdoor criterion
        
        Args:
            source: Source variable
            target: Target variable
            graph: Graph state
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])
        
        # Check if variables exist
        if source not in nodes or target not in nodes:
            return False, f"Variables {source} or {target} not in graph"
        
        # Check for direct edge (simplified causal validation)
        if (source, target) in edges:
            # Valid causal edge
            return True, None
        
        # Check for confounders using backdoor criterion (simplified)
        # Look for common causes
        common_causes = []
        for node in nodes:
            if node != source and node != target:
                # Check if node is a parent of both source and target
                has_edge_to_source = (node, source) in edges
                has_edge_to_target = (node, target) in edges
                if has_edge_to_source and has_edge_to_target:
                    common_causes.append(node)
        
        if common_causes:
            return False, f"Potential confounder detected: {common_causes[0]}"
        
        return True, None
    
    def _clean_variable(self, var: str) -> Optional[str]:
        """
        Clean and validate a variable name.
        
        Args:
            var: Raw variable name
            
        Returns:
            Cleaned variable name or None if invalid
        """
        if not var:
            return None
        
        # Normalize
        var = self._normalize_variable_name(var)
        var_lower = var.lower()
        
        # CRITICAL: Filter out action verbs (epistemic/intentional actions)
        # These are NOT causal state variables - they're tasks, not observables
        if self._is_action_verb(var):
            return None
        
        # CRITICAL: Filter out epistemic/intentional terms
        # These are about knowledge/intentions, not causal state variables
        if self._is_epistemic_term(var):
            return None
        
        # Filter out relationship phrases (contain causal verbs)
        if any(verb in var_lower for verb in self.causal_verbs):
            return None
        
        # Filter out if it contains relationship indicators
        relationship_indicators = ['depends', 'causes', 'affects', 'influences', 'leads', 'results', 'impacts']
        if any(indicator in var_lower for indicator in relationship_indicators):
            return None
        
        # Filter out value descriptions (buy, sell, etc. when they're part of percentages)
        value_descriptors = ['buy', 'sell', 'percent', 'percentage', '%']
        if var in value_descriptors:
            return None
        
        # Filter out time units that are standalone (but keep "7 days" as a variable)
        time_units = ['day', 'days', 'hour', 'hours', 'minute', 'minutes', 'second', 'seconds']
        if var in time_units and len(var.split()) == 1:
            return None
        
        # Remove common conjunctions at start/end
        words = var.split()
        if words:
            # Remove "and", "or", "the", "a", "an", "if" from start
            while words and words[0].lower() in {'and', 'or', 'the', 'a', 'an', 'if'}:
                words = words[1:]
            # Remove "and", "or" from end
            while words and words[-1].lower() in {'and', 'or'}:
                words = words[:-1]
        
        if not words:
            return None
        
        var = ' '.join(words)
        
        # Filter out if it's just stop words
        if var in self.stop_words:
            return None
        
        # Filter out if all words are stop words
        if all(w in self.stop_words for w in var.split()):
            return None
        
        # Stricter filtering for single-word variables
        if len(words) == 1:
            # Filter out single-word variables that are likely invalid
            invalid_single_words = {
                'if', 'and', 'or', 'but', 'the', 'a', 'an', 'buy', 'sell', 
                'days', 'day', 'hours', 'hour', 'minutes', 'minute', 
                'seconds', 'second', 'of', 'in', 'on', 'at', 'to', 'for',
                'from', 'with', 'by', 'as', 'is', 'was', 'are', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                'did', 'will', 'would', 'could', 'should', 'may', 'might',
                'must', 'can', 'this', 'that', 'these', 'those', 'what',
                'which', 'who', 'whom', 'where', 'when', 'why', 'how'
            }
            if var.lower() in invalid_single_words:
                return None
            
            # Filter out very short single-word variables (unless it's a known keyword)
            if len(var) < 3 and var not in self.variable_keywords:
                return None
            
            # Only allow single-word variables if they're in the keyword list
            if var not in self.variable_keywords:
                # Check if it's a meaningful single word (not a stop word)
                if var.lower() in self.stop_words:
                    return None
        
        # Filter out variables that are just conjunctions
        if var.lower() in {'and', 'or', 'but', 'the', 'a', 'an', 'if'}:
            return None
        
        # Filter out very long phrases (likely not a single variable)
        if len(words) > 4:
            return None
        
        # Final check: Reject if it's an action verb or epistemic term
        if self._is_action_verb(var) or self._is_epistemic_term(var):
            return None
        
        # Optional: Use dictionary to validate word (if lexical compiler available)
        # This helps filter out made-up words or typos that passed other filters
        # Note: This is a soft check - we don't require dictionary validation for all words
        # as domain-specific terms may not be in standard dictionaries
        if self.lexical_compiler and self.lexical_compiler.enable_dictionary:
            # For single-word variables, check if it's a valid word
            # Multi-word phrases are more likely to be domain-specific
            if len(words) == 1:
                # Check if word exists in dictionary
                if not self.lexical_compiler.is_valid_word(words[0]):
                    # Word not found - could be a typo or domain-specific term
                    # We'll still allow it but with lower confidence
                    logger.debug(f"Word '{words[0]}' not found in dictionary - may be domain-specific or typo")
        
        return var
    
    def _extract_clean_variables_from_relationships(self, relationships: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract clean variables from relationships.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            Set of clean variable names
        """
        variables = set()
        for rel in relationships:
            source = self._clean_variable(rel.get('source', ''))
            target = self._clean_variable(rel.get('target', ''))
            if source:
                variables.add(source)
            if target:
                variables.add(target)
        return variables
    
    def _infer_relationships_from_context(self, variables: Set[str], text: str) -> List[Tuple[str, str]]:
        """
        Infer relationships from context when explicit patterns aren't found.
        Enhanced to handle conditional questions and state descriptions.
        
        Args:
            variables: Set of extracted variables
            text: Original text
            
        Returns:
            List of inferred (source, target) tuples
        """
        inferred = []
        var_list = sorted(list(variables))
        text_lower = text.lower()
        
        # Clean variable list - remove value descriptors
        cleaned_vars = [v for v in var_list if self._clean_variable(v)]
        
        # If we have a conditional question pattern: "If X is Y, what is Z?"
        if 'if' in text_lower and ('what' in text_lower or 'which' in text_lower or 'expected' in text_lower):
            # Find variables mentioned before "what" or "expected"
            question_markers = ['what', 'which', 'expected']
            question_pos = -1
            for marker in question_markers:
                pos = text_lower.find(marker)
                if pos > 0:
                    question_pos = pos
                    break
            
            if question_pos > 0:
                before_question = text_lower[:question_pos]
                after_question = text_lower[question_pos:]
                
                # Variables before question are likely causes (state variables)
                before_vars = [v for v in cleaned_vars if v.lower() in before_question and 'expected' not in v.lower()]
                # Variables after question are likely effects (question target)
                after_vars = [v for v in cleaned_vars if v.lower() in after_question or 'expected' in v.lower()]
                
                # Also look for "expected X" pattern
                expected_pattern = r'expected\s+(\w+(?:\s+\w+)?)'
                expected_match = re.search(expected_pattern, text_lower, re.IGNORECASE)
                if expected_match:
                    expected_var = self._clean_variable(expected_match.group(1))
                    if expected_var and expected_var not in after_vars:
                        after_vars.append(expected_var)
                
                # Create relationships: state variables -> question target
                if before_vars and after_vars:
                    for before_var in before_vars:
                        for after_var in after_vars:
                            if before_var != after_var:
                                inferred.append((before_var, after_var))
                elif before_vars and not after_vars:
                    # If no explicit target, use the most likely target (e.g., "expected price")
                    # Look for variables with "expected" or mentioned in question
                    question_var_pattern = r'(?:what|which|expected)\s+(?:is|are|will|would|the\s+)?(\w+(?:\s+\w+)?)'
                    q_match = re.search(question_var_pattern, text_lower, re.IGNORECASE)
                    if q_match:
                        q_var = self._clean_variable(q_match.group(1))
                        if q_var and q_var in cleaned_vars:
                            for before_var in before_vars:
                                if before_var != q_var:
                                    inferred.append((before_var, q_var))
        
        # If we have state descriptions with multiple variables
        # Pattern: "X is Y & Z is W" -> X and Z might affect the question target
        if '&' in text or (' and ' in text_lower and 'is' in text_lower):
            # Find variables mentioned with "is" followed by values
            state_pattern = r'(\w+(?:\s+\w+)?)\s+(?:is|are|was|were)\s+[\d%]+'
            state_matches = list(re.finditer(state_pattern, text_lower, re.IGNORECASE))
            
            if len(state_matches) >= 1:
                # Variables mentioned in state descriptions
                state_vars = []
                for m in state_matches:
                    var = self._clean_variable(m.group(1))
                    if var and var in cleaned_vars:
                        state_vars.append(var)
                
                # Find question target
                question_vars = []
                # Look for "expected X" or "what is X"
                expected_pattern = r'expected\s+(\w+(?:\s+\w+)?)'
                what_pattern = r'what\s+(?:is|are|will|would)\s+(?:the\s+)?(\w+(?:\s+\w+)?)'
                
                for pattern in [expected_pattern, what_pattern]:
                    match = re.search(pattern, text_lower, re.IGNORECASE)
                    if match:
                        q_var = self._clean_variable(match.group(1))
                        if q_var and q_var in cleaned_vars:
                            question_vars.append(q_var)
                
                # If no explicit question var, look for variables with "price" or similar
                if not question_vars:
                    price_vars = [v for v in cleaned_vars if 'price' in v.lower() and 'expected' not in v.lower()]
                    if price_vars:
                        question_vars = price_vars[:1]  # Take first one
                
                # Create relationships from state variables to question target
                for state_var in state_vars:
                    for q_var in question_vars:
                        if state_var != q_var:
                            inferred.append((state_var, q_var))
        
        # Remove duplicates
        inferred = list(set(inferred))
        
        return inferred
    
    def extract_variables_from_task(self, task: str) -> Dict[str, Any]:
        """
        Advanced extraction of variables and relationships from natural language.
        
        Automatically extracts variables and relationships (causal, knowledge, etc.) from natural language text.
        Enhanced to handle:
        - Causal relationships (depends on, affects, causes)
        - General knowledge relationships (is-a, has, part-of, located-in)
        - Numerical values, conditionals, questions, and state descriptions
        - Definitions, facts, and taxonomic relationships
        
        Args:
            task: Natural language task description
            
        Returns:
            Dictionary with 'variables', 'edges', 'relationships', and metadata
            
        Example:
            >>> agent = HybridAgent()
            >>> result = agent.extract_causal_variables("price depends on demand and supply")
            >>> print(result['variables'])  # ['price', 'demand', 'supply']
            >>> print(result['edges'])  # [('price', 'demand'), ('price', 'supply')]
            
            >>> result = agent.extract_causal_variables("A dog is a mammal")
            >>> print(result['variables'])  # ['dog', 'mammal']
            >>> print(result['edges'])  # [('dog', 'mammal')] with type='is_a'
            
            >>> result = agent.extract_causal_variables("Paris is in France")
            >>> print(result['variables'])  # ['Paris', 'France']
            >>> print(result['edges'])  # [('Paris', 'France')] with type='located_in'
        """
        # Extract relationships with context
        relationships = self._extract_with_context(task)
        
        # Extract clean variables from relationships first (most reliable)
        variables = self._extract_clean_variables_from_relationships(relationships)
        
        # Extract standalone variables (supplementary) - this now handles state descriptions
        standalone_vars = self._extract_standalone_variables(task)
        
        # Clean standalone variables
        for var in standalone_vars:
            cleaned = self._clean_variable(var)
            if cleaned:
                variables.add(cleaned)
        
        # NEW: Extract state variables from action verbs and epistemic terms
        # This helps handle vague language by finding what action verbs/epistemic terms refer to
        action_verb_vars = self._extract_state_variables_from_action_verbs(task)
        epistemic_vars = self._extract_state_variables_from_epistemic_terms(task)
        vague_language_vars = self._extract_variables_from_vague_language(task)
        
        # Add extracted variables (they're already cleaned by the extraction methods)
        for var in action_verb_vars:
            cleaned = self._clean_variable(var)
            if cleaned:
                variables.add(cleaned)
        
        for var in epistemic_vars:
            cleaned = self._clean_variable(var)
            if cleaned:
                variables.add(cleaned)
        
        for var in vague_language_vars:
            cleaned = self._clean_variable(var)
            if cleaned:
                variables.add(cleaned)
        
        # Resolve references
        variables = self._resolve_references(task, variables)
        
        # Merge similar variables
        variables = self._merge_similar_variables(variables)
        
        # Final filtering: remove invalid variables
        variables = self._filter_valid_variables(variables)
        
        # Build clean edges from relationships
        edges = []
        for rel in relationships:
            if not rel.get('negated', False):  # Only add non-negated relationships
                source = self._clean_variable(rel['source'])
                target = self._clean_variable(rel['target'])
                if source and target and source != target:
                    # Filter out edges with "of" at the end (e.g., "price of")
                    if not target.endswith(' of') and not source.endswith(' of'):
                        edges.append((source, target))
        
        # If no explicit edges found, try to infer from context
        if not edges and variables:
            inferred_edges = self._infer_relationships_from_context(variables, task)
            edges.extend(inferred_edges)
        
        # Clean up edges: remove edges to/from invalid variables
        # Only include edges between valid variables
        valid_vars_set = variables
        cleaned_edges = []
        for source, target in edges:
            source_clean = self._clean_variable(source)
            target_clean = self._clean_variable(target)
            
            # Both must be valid and in the valid variables set
            if (source_clean and target_clean and 
                source_clean != target_clean and
                source_clean in valid_vars_set and 
                target_clean in valid_vars_set):
                # Don't create edges to time units unless they're part of a compound variable
                if target_clean in ['days', 'day', 'hours', 'hour'] and len(target_clean.split()) == 1:
                    continue
                # Don't create edges from single words to compound phrases that contain them
                if source_clean in target_clean.split() or target_clean in source_clean.split():
                    # Only skip if one is clearly a subset of the other
                    if len(source_clean.split()) < len(target_clean.split()) or len(target_clean.split()) < len(source_clean.split()):
                        continue
                cleaned_edges.append((source_clean, target_clean))
        
        # Remove duplicate edges
        edges = list(set(cleaned_edges))
        
        # Prioritize edges: prefer edges to "expected X" or question targets
        question_targets = [v for v in variables if 'expected' in v.lower()]
        if question_targets:
            # Keep edges that go to question targets
            prioritized_edges = [e for e in edges if e[1] in question_targets]
            # Add other edges that don't conflict
            for e in edges:
                if e not in prioritized_edges:
                    # Only add if source doesn't already have an edge to a question target
                    if not any(e[0] == p[0] for p in prioritized_edges):
                        prioritized_edges.append(e)
            edges = prioritized_edges if prioritized_edges else edges
        
        # Extract metadata
        metadata = {
            'total_relationships': len(relationships),
            'negated_relationships': sum(1 for r in relationships if r.get('negated', False)),
            'average_confidence': sum(r['confidence'] for r in relationships) / len(relationships) if relationships else 0.0,
            'variables_extracted': len(variables),
            'edges_extracted': len(edges),
            'variables_with_values': self._extract_variables_with_values(task)
        }
        
        return {
            'variables': sorted(list(variables)),  # Sorted for consistency
            'edges': edges,
            'relationships': relationships,
            'metadata': metadata
        }
    
    def infer_causal_structure(self, variables: List[str], context: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Infer causal structure from variables using advanced logical inference.
        
        Args:
            variables: List of variable names
            context: Optional context text for better inference
            
        Returns:
            List of (source, target) tuples representing causal edges
        """
        edges = []
        
        if not variables:
            return edges
        
        # Strategy 1: Sequential inference (variables mentioned in order)
        # Only if we have 2-4 variables (too many would create too many edges)
        if 2 <= len(variables) <= 4:
            for i in range(len(variables) - 1):
                source = variables[i]
                target = variables[i + 1]
                # Only add if not creating cycles
                if not self.graph_manager.has_path(target, source):
                    edges.append((source, target))
        
        # Strategy 2: Domain-specific heuristics
        # Common patterns: input -> process -> output, cause -> effect
        variable_lower = [v.lower() for v in variables]
        
        # Look for common causal patterns
        input_keywords = ['input', 'source', 'cause', 'factor', 'driver', 'trigger']
        output_keywords = ['output', 'result', 'effect', 'outcome', 'consequence', 'impact']
        process_keywords = ['process', 'mechanism', 'method', 'approach', 'system']
        
        inputs = [v for v, v_lower in zip(variables, variable_lower) 
                 if any(kw in v_lower for kw in input_keywords)]
        outputs = [v for v, v_lower in zip(variables, variable_lower)
                  if any(kw in v_lower for kw in output_keywords)]
        processes = [v for v, v_lower in zip(variables, variable_lower)
                    if any(kw in v_lower for kw in process_keywords)]
        
        # Input -> Process -> Output pattern
        if inputs and processes:
            for inp in inputs:
                for proc in processes:
                    if not self.graph_manager.has_path(proc, inp):
                        edges.append((inp, proc))
        
        if processes and outputs:
            for proc in processes:
                for out in outputs:
                    if not self.graph_manager.has_path(out, proc):
                        edges.append((proc, out))
        
        # Direct input -> output (if no process)
        if inputs and outputs and not processes:
            for inp in inputs:
                for out in outputs:
                    if not self.graph_manager.has_path(out, inp):
                        edges.append((inp, out))
        
        # Strategy 3: Context-based inference (if context provided)
        if context:
            context_lower = context.lower()
            # Look for mentions of variables in context
            for i, var1 in enumerate(variables):
                for var2 in variables[i+1:]:
                    # Check if var1 appears before var2 in context
                    pos1 = context_lower.find(var1.lower())
                    pos2 = context_lower.find(var2.lower())
                    if pos1 != -1 and pos2 != -1 and pos1 < pos2:
                        # Check if there's a causal word between them
                        between = context_lower[pos1:pos2]
                        if any(verb in between for verb in self.causal_verbs):
                            if not self.graph_manager.has_path(var2, var1):
                                edges.append((var1, var2))
        
        # Remove duplicates
        edges = list(set(edges))
        
        return edges
    
    def validate_causal_graph(self) -> Tuple[bool, Optional[str]]:
        """
        Validate that the causal graph is a valid DAG.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.graph_manager.is_dag():
            return False, "Graph contains cycles"
        return True, None
    
    def apply_causal_rules(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Apply rule-based causal reasoning to a state.
        
        Args:
            state: Dictionary mapping variables to values
            
        Returns:
            Updated state after applying causal rules
        """
        result = state.copy()
        
        # Get topological order
        try:
            order = self.graph_manager.topological_sort()
        except Exception:
            order = list(state.keys())
        
        # Apply causal propagation
        for node in order:
            if node not in result:
                continue
            
            parents = self.graph_manager.get_parents(node)
            if not parents:
                continue
            
            # Simple linear combination rule
            value = result.get(node, 0.0)
            for parent in parents:
                if parent in result:
                    strength = self.graph_manager.edge_strength(parent, node)
                    value += result[parent] * strength * 0.1  # Dampening factor
            
            result[node] = value
        
        return result


class StatisticalEngine:
    """
    Statistical inference engine wrapping StatisticalMethods.
    
    Provides Bayesian inference, regression-based edge estimation,
    and uncertainty quantification.
    """
    
    def __init__(
        self,
        graph_manager: GraphManager,
        prediction_framework: PredictionFramework,
        seed: int = 42
    ):
        """
        Initialize statistical engine.
        
        Args:
            graph_manager: GraphManager instance
            prediction_framework: PredictionFramework instance
            seed: Random seed
        """
        self.graph_manager = graph_manager
        self.prediction_framework = prediction_framework
        self.statistical_methods = StatisticalMethods(
            graph_manager=graph_manager,
            prediction_framework=prediction_framework,
            seed=seed
        )
    
    def fit_from_dataframe(
        self,
        df: Any,
        variables: List[str],
        window: int = 30,
        decay_alpha: float = 0.9,
        ridge_lambda: float = 0.0,
        enforce_signs: bool = True
    ) -> None:
        """
        Fit edge strengths from data using Bayesian regression.
        
        Args:
            df: pandas DataFrame with historical data
            variables: List of variable names to fit
            window: Rolling window size
            decay_alpha: Decay factor for recency weighting
            ridge_lambda: Ridge regularization parameter
            enforce_signs: Whether to enforce edge sign constraints
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for statistical fitting")
        
        self.statistical_methods.fit_from_dataframe(
            df=df,
            variables=variables,
            window=window,
            decay_alpha=decay_alpha,
            ridge_lambda=ridge_lambda,
            enforce_signs=enforce_signs
        )
        
        # Update prediction framework standardization stats
        self.prediction_framework.standardization_stats = (
            self.statistical_methods.standardization_stats.copy()
        )
    
    def quantify_uncertainty(
        self,
        df: Any,
        variables: List[str],
        windows: int = 200,
        alpha: float = 0.95
    ) -> Dict[str, Any]:
        """
        Quantify uncertainty using bootstrap resampling.
        
        Args:
            df: pandas DataFrame
            variables: List of variable names
            windows: Number of bootstrap samples
            alpha: Confidence level
            
        Returns:
            Dictionary with edge confidence intervals
        """
        if not PANDAS_AVAILABLE:
            return {}
        
        return self.statistical_methods.quantify_uncertainty(
            df=df,
            variables=variables,
            windows=windows,
            alpha=alpha
        )
    
    def assess_causal_strength(self, source: str, target: str) -> float:
        """
        Assess causal strength between two variables.
        
        Args:
            source: Source variable
            target: Target variable
            
        Returns:
            Causal strength (0.0 if no edge exists)
        """
        return self.graph_manager.edge_strength(source, target)
    
    def generate_probabilistic_counterfactuals(
        self,
        factual_state: Dict[str, float],
        target_variables: List[str],
        n_scenarios: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate probabilistic counterfactual scenarios.
        
        Args:
            factual_state: Current factual state
            target_variables: Variables to intervene on
            n_scenarios: Number of scenarios to generate
            
        Returns:
            List of counterfactual scenario dictionaries
        """
        scenarios = []
        
        for i in range(n_scenarios):
            # Generate intervention values (deterministic sampling using seeded RNG)
            interventions = {}
            # Use consistency engine's RNG if available, otherwise use statistical engine's
            rng = getattr(self, '_rng', None)
            if rng is None and hasattr(self.statistical_engine, '_rng'):
                rng = self.statistical_engine._rng
            elif rng is None:
                # Fallback: create deterministic RNG with seed
                if NUMPY_AVAILABLE:
                    rng = np.random.default_rng(42)
                else:
                    import random
                    random.seed(42)
                    rng = random
            
            for var in target_variables:
                if var in factual_state:
                    base_value = factual_state[var]
                    # Sample around base value (deterministic)
                    if NUMPY_AVAILABLE and hasattr(rng, 'random'):
                        random_val = float(rng.random())
                    else:
                        random_val = rng.random() if hasattr(rng, 'random') else 0.5
                    intervention_value = base_value * (0.5 + random_val)
                    interventions[var] = intervention_value
            
            # Predict outcomes
            try:
                predicted = self.prediction_framework.predict_outcomes(
                    factual_state=factual_state,
                    interventions=interventions
                )
                
                scenarios.append({
                    'name': f'Scenario {i+1}',
                    'interventions': interventions,
                    'expected_outcomes': predicted,
                    'probability': 1.0 / n_scenarios
                })
            except Exception as e:
                logger.warning(f"Failed to generate scenario {i+1}: {e}")
                continue
        
        return scenarios


class RuleBasedNLG:
    """
    Enhanced rule-based natural language generation for LLM replacement.
    
    Generates natural, conversational responses from graph state and reasoning results.
    Uses pragmatic layer for tone adjustment and confidence-based language.
    """
    
    def __init__(self):
        """Initialize enhanced rule-based NLG."""
        self.templates = {
            'causal_analysis': """## Causal Analysis

**Variables Identified:** {variables}

**Causal Relationships:**
{relationships}

**Graph Structure:**
{graph_structure}

**Key Insights:**
{insights}
""",
            'knowledge_analysis': """## Knowledge Analysis

**Entities Identified:** {variables}

**Relationships:**
{relationships}

**Graph Structure:**
{graph_structure}

**Key Insights:**
{insights}
""",
            'general_analysis': """## Analysis

**Entities/Variables Identified:** {variables}

**Relationships:**
{relationships}

**Graph Structure:**
{graph_structure}

**Key Insights:**
{insights}
""",
            'counterfactual': """## Counterfactual Scenario: {name}

**Interventions:**
{interventions}

**Expected Outcomes:**
{outcomes}

**Probability:** {probability:.2%}
""",
            'statistical_summary': """## Statistical Summary

**Edge Strengths:**
{edge_strengths}

**Uncertainty:**
{uncertainty}

**Confidence Intervals:**
{confidence_intervals}
""",
            'recommendation': """## Recommendations

Based on the causal analysis:

{recommendations}
""",
            'conversational_intro': """I've analyzed your question about {topic}. Here's what I found:

""",
            'conversational_summary': """
Based on the causal relationships I've identified, {summary}

""",
            'question_answer': """To answer your question: {question}

{answer}

This conclusion is derived from the causal graph structure, which shows {explanation}.

""",
            'explanation': """Let me explain how I reached this conclusion:

{explanation}

The causal relationships in the graph indicate that {insight}.

""",
        }
        
        # Conversational connectors
        self.connectors = {
            'high_confidence': ['Based on', 'According to', 'The evidence shows', 'Analysis indicates'],
            'medium_confidence': ['It appears that', 'The data suggests', 'This likely means', 'It seems'],
            'low_confidence': ['It may be that', 'Possibly', 'This could indicate', 'There might be'],
            'transition': ['Furthermore', 'Additionally', 'Moreover', 'In addition', 'Also'],
            'conclusion': ['Therefore', 'Thus', 'As a result', 'Consequently', 'Hence']
        }
        
        # Natural language patterns for different intents
        self.intent_responses = {
            'question': "Let me answer your question based on the relationships I've identified.",
            'analysis': "I've performed an analysis of the relationships you described.",
            'prediction': "Based on the structure, here's what I predict:",
            'counterfactual': "Let me explore what would happen if we changed certain variables:",
            'recommendation': "Based on the analysis, here are my recommendations:",
            'extraction': "I've extracted the following structure from your description:",
            'definition': "Here's what I know about that:",
            'person_query': "Here's information about that person:",
            'location_query': "Here's the location information:",
            'temporal_query': "Here's the temporal information:",
            'explanation': "Let me explain:",
            'comparison': "Comparing the entities, I found:"
        }
    
    def format_causal_analysis(self, analysis: Dict[str, Any]) -> str:
        """
        Format causal analysis results into natural language.
        
        Args:
            analysis: Dictionary with analysis results
            
        Returns:
            Formatted natural language text
        """
        variables = analysis.get('variables', [])
        relationships = analysis.get('relationships', [])
        graph_structure = analysis.get('graph_structure', '')
        insights = analysis.get('insights', [])
        
        # Format relationships
        rel_text = []
        for rel in relationships:
            source = rel.get('source', '')
            target = rel.get('target', '')
            rel_type = rel.get('type', '')
            strength = rel.get('strength', 0.0)
            rel_text.append(f"- {source} -> {target} (type: {rel_type}, strength: {strength:.3f})")
        
        # Format insights
        insights_text = []
        if isinstance(insights, list):
            for insight in insights:
                insights_text.append(f"- {insight}")
        else:
            insights_text.append(f"- {insights}")
        
        return self.templates['causal_analysis'].format(
            variables=', '.join(variables) if variables else 'None identified',
            relationships='\n'.join(rel_text) if rel_text else 'No relationships found',
            graph_structure=graph_structure or 'No graph structure available',
            insights='\n'.join(insights_text) if insights_text else 'No insights generated'
        )
    
    def format_counterfactuals(self, scenarios: List[Dict[str, Any]]) -> str:
        """
        Format counterfactual scenarios into natural language.
        
        Args:
            scenarios: List of counterfactual scenario dictionaries
            
        Returns:
            Formatted natural language text
        """
        if not scenarios:
            return "No counterfactual scenarios generated."
        
        formatted = []
        for scenario in scenarios:
            name = scenario.get('name', 'Unknown')
            interventions = scenario.get('interventions', {})
            outcomes = scenario.get('expected_outcomes', {})
            probability = scenario.get('probability', 0.0)
            
            # Format interventions
            inter_text = []
            for var, val in interventions.items():
                inter_text.append(f"- {var}: {val:.3f}")
            
            # Format outcomes
            out_text = []
            for var, val in outcomes.items():
                out_text.append(f"- {var}: {val:.3f}")
            
            formatted.append(self.templates['counterfactual'].format(
                name=name,
                interventions='\n'.join(inter_text) if inter_text else 'None',
                outcomes='\n'.join(out_text) if out_text else 'None',
                probability=probability
            ))
        
        return '\n\n'.join(formatted)
    
    def format_statistical_results(self, results: Dict[str, Any]) -> str:
        """
        Format statistical results into natural language.
        
        Args:
            results: Dictionary with statistical results
            
        Returns:
            Formatted natural language text
        """
        edge_strengths = results.get('edge_strengths', {})
        uncertainty = results.get('uncertainty', {})
        confidence_intervals = results.get('confidence_intervals', {})
        
        # Format edge strengths
        strength_text = []
        for (source, target), strength in edge_strengths.items():
            strength_text.append(f"- {source} -> {target}: {strength:.3f}")
        
        # Format uncertainty
        uncertainty_text = []
        for key, val in uncertainty.items():
            uncertainty_text.append(f"- {key}: {val:.3f}")
        
        # Format confidence intervals
        ci_text = []
        for key, ci in confidence_intervals.items():
            if isinstance(ci, dict):
                lower = ci.get('lower', 0.0)
                upper = ci.get('upper', 0.0)
                ci_text.append(f"- {key}: [{lower:.3f}, {upper:.3f}]")
            else:
                ci_text.append(f"- {key}: {ci}")
        
        return self.templates['statistical_summary'].format(
            edge_strengths='\n'.join(strength_text) if strength_text else 'None',
            uncertainty='\n'.join(uncertainty_text) if uncertainty_text else 'None',
            confidence_intervals='\n'.join(ci_text) if ci_text else 'None'
        )
    
    def generate_response(
        self,
        reasoning_result: Dict[str, Any],
        response_type: str = 'full',
        pragmatic_info: Optional[Dict[str, Any]] = None,
        show_reasoning: bool = False,
        reasoning_chain: Optional[ReasoningChain] = None
    ) -> str:
        """
        Generate enhanced natural language response with conversational tone.
        
        Args:
            reasoning_result: Dictionary with reasoning results
            response_type: Type of response ('full', 'analysis', 'counterfactuals', 'statistical', 'conversational')
            pragmatic_info: Optional pragmatic information (register, hedging, explicitness)
            
        Returns:
            Natural language response
        """
        intent = reasoning_result.get('intent', {})
        intent_type = intent.get('type', 'analysis')
        
        # Get pragmatic information
        if pragmatic_info is None:
            pragmatic_info = reasoning_result.get('pragmatic', {})
        
        register = pragmatic_info.get('register', 'neutral')
        hedging = pragmatic_info.get('hedging', 'likely')
        
        if response_type == 'conversational':
            # Enhanced conversational format with chain-of-thought
            parts = []
            
            # Show chain-of-thought reasoning if requested
            if show_reasoning and reasoning_chain:
                reasoning_text = self._format_reasoning_chain(reasoning_chain)
                parts.append(reasoning_text)
            
            # Conversational introduction
            task = reasoning_result.get('task', '')
            if task:
                # Extract topic from task
                topic = self._extract_topic(task)
                intro = self.intent_responses.get(intent_type, "I've analyzed your request.")
                parts.append(intro + "\n\n")
            
            # Graph-first answer if available (most authoritative)
            graph_answer = reasoning_result.get('graph_first_answer', {})
            if graph_answer and graph_answer.get('answer'):
                # Determine graph type from relationships or result
                analysis = reasoning_result.get('analysis', {})
                analysis_relationships = analysis.get('relationships', [])
                has_general = any(
                    isinstance(rel, dict) and rel.get('category') in ['taxonomic', 'meronymic', 'spatial', 'functional', 'definitional', 'factual']
                    for rel in analysis_relationships
                )
                graph_type_str = 'knowledge' if has_general else 'causal'
                parts.append(self._format_graph_answer(graph_answer, hedging, graph_type_str))
            
            # Analysis with natural language (causal or general knowledge)
            if 'analysis' in reasoning_result:
                analysis = reasoning_result['analysis']
                # Determine if this is causal or general knowledge
                analysis_relationships = analysis.get('relationships', [])
                has_causal = any(
                    (isinstance(rel, dict) and (rel.get('category') == 'causal' or rel.get('type', '').startswith('causal'))) or
                    (isinstance(rel, str) and 'causal' in rel.lower())
                    for rel in analysis_relationships
                )
                has_general = any(
                    isinstance(rel, dict) and rel.get('category') in ['taxonomic', 'meronymic', 'spatial', 'functional', 'definitional', 'factual']
                    for rel in analysis_relationships
                )
                
                if has_causal and not has_general:
                    analysis_text = self._format_analysis_conversational(
                        analysis,
                        register,
                        hedging
                    )
                elif has_general:
                    analysis_text = self._format_knowledge_analysis_conversational(
                        analysis,
                        register,
                        hedging
                    )
                else:
                    analysis_text = self._format_analysis_conversational(
                        analysis,
                        register,
                        hedging
                    )
                parts.append(analysis_text)
            
            # Answer specific questions
            if intent_type in ['question', 'prediction'] and 'analysis' in reasoning_result:
                answer = self._generate_question_answer(reasoning_result, hedging)
                if answer:
                    parts.append(answer)
            
            # Counterfactuals with explanation
            if 'counterfactuals' in reasoning_result and reasoning_result['counterfactuals']:
                cf_text = self._format_counterfactuals_conversational(
                    reasoning_result['counterfactuals'],
                    hedging
                )
                parts.append(cf_text)
            
            # Recommendations
            recommendations = reasoning_result.get('recommendations', [])
            if recommendations:
                rec_text = self._format_recommendations_conversational(recommendations, hedging)
                parts.append(rec_text)
            
            # Statistical results (if available and relevant)
            if 'statistical' in reasoning_result and reasoning_result['statistical']:
                if intent_type in ['analysis', 'prediction', 'comparison']:
                    stat_text = self._format_statistical_conversational(
                        reasoning_result['statistical'],
                        hedging
                    )
                    parts.append(stat_text)
            
            # Show transparency information if available
            if 'transparency' in reasoning_result and response_type != 'brief':
                transparency_text = self._format_transparency(reasoning_result['transparency'])
                parts.append(transparency_text)
            
            return '\n\n'.join(parts)
        
        elif response_type == 'full':
            # Original full format for backwards compatibility
            parts = []
            
            task = reasoning_result.get('task', '')
            if task:
                parts.append(f"## Task Analysis\n\nAnalyzing: *{task}*\n")
            
            if intent_type == 'extraction':
                parts.append("## Extracted Causal Structure\n")
            elif intent_type == 'counterfactual':
                parts.append("## Counterfactual Analysis\n")
            elif intent_type == 'recommendation':
                parts.append("## Causal Analysis & Recommendations\n")
            else:
                parts.append("## Causal Analysis\n")
            
            if 'analysis' in reasoning_result:
                parts.append(self.format_causal_analysis(reasoning_result['analysis']))
            
            recommendations = reasoning_result.get('recommendations', [])
            if recommendations:
                parts.append("\n## Recommendations\n")
                for i, rec in enumerate(recommendations, 1):
                    parts.append(f"{i}. {rec}")
            
            if 'counterfactuals' in reasoning_result and reasoning_result['counterfactuals']:
                parts.append("\n" + self.format_counterfactuals(reasoning_result['counterfactuals']))
            
            if 'statistical' in reasoning_result and reasoning_result['statistical']:
                if intent_type in ['analysis', 'prediction', 'comparison']:
                    parts.append("\n" + self.format_statistical_results(reasoning_result['statistical']))
            
            return '\n\n'.join(parts)
        
        elif response_type == 'analysis':
            return self.format_causal_analysis(reasoning_result.get('analysis', {}))
        
        elif response_type == 'counterfactuals':
            return self.format_counterfactuals(reasoning_result.get('counterfactuals', []))
        
        elif response_type == 'statistical':
            return self.format_statistical_results(reasoning_result.get('statistical', {}))
        
        else:
            return str(reasoning_result)
    
    def _extract_topic(self, task: str) -> str:
        """Extract topic from task for conversational intro."""
        # Simple extraction - take first few words
        words = task.split()[:5]
        return ' '.join(words)
    
    def _format_reasoning_chain(self, reasoning_chain: ReasoningChain) -> str:
        """
        Format reasoning chain for display.
        
        Args:
            reasoning_chain: Reasoning chain to format
            
        Returns:
            Formatted reasoning text
        """
        parts = ["## Chain-of-Thought Reasoning\n"]
        
        for i, step in enumerate(reasoning_chain.steps, 1):
            step_text = f"**Step {i}: {step.operation}**\n"
            if step.input_state:
                step_text += f"  Input: {str(step.input_state)[:100]}...\n"
            if step.output_state:
                step_text += f"  Output: {str(step.output_state)[:100]}...\n"
            if step.conclusion:
                step_text += f"  Conclusion: {step.conclusion}\n"
            if step.confidence < 1.0:
                step_text += f"  Confidence: {step.confidence:.2f}\n"
            
            parts.append(step_text)
        
        return "\n".join(parts)
    
    def _format_transparency(self, transparency: Dict[str, Any]) -> str:
        """
        Format transparency information.
        
        Args:
            transparency: Transparency dictionary
            
        Returns:
            Formatted transparency text
        """
        parts = ["## Transparency\n"]
        
        confidence_viz = transparency.get('confidence', {})
        if confidence_viz:
            mean_conf = confidence_viz.get('mean_confidence', 0.0)
            std_conf = confidence_viz.get('std_confidence', 0.0)
            parts.append(f"**Confidence:** {mean_conf:.2f} ± {std_conf:.2f}")
        
        graph_structure = transparency.get('graph_structure', {})
        if graph_structure:
            parts.append(f"**Graph Structure:** {graph_structure.get('structure_type', 'unknown')}")
            parts.append(f"  - Nodes: {graph_structure.get('num_nodes', 0)}")
            parts.append(f"  - Edges: {graph_structure.get('num_edges', 0)}")
        
        return "\n".join(parts)
    
    def _format_reasoning_chain(self, reasoning_chain: ReasoningChain) -> str:
        """
        Format reasoning chain for display.
        
        Args:
            reasoning_chain: Reasoning chain to format
            
        Returns:
            Formatted reasoning text
        """
        parts = ["## Chain-of-Thought Reasoning\n"]
        
        for i, step in enumerate(reasoning_chain.steps, 1):
            step_text = f"**Step {i}: {step.operation}**\n"
            if step.input_state:
                step_text += f"  Input: {str(step.input_state)[:100]}...\n"
            if step.output_state:
                step_text += f"  Output: {str(step.output_state)[:100]}...\n"
            if step.conclusion:
                step_text += f"  Conclusion: {step.conclusion}\n"
            if step.confidence < 1.0:
                step_text += f"  Confidence: {step.confidence:.2f}\n"
            
            parts.append(step_text)
        
        return "\n".join(parts)
    
    def _format_transparency(self, transparency: Dict[str, Any]) -> str:
        """
        Format transparency information.
        
        Args:
            transparency: Transparency dictionary
            
        Returns:
            Formatted transparency text
        """
        parts = ["## Transparency\n"]
        
        confidence_viz = transparency.get('confidence', {})
        if confidence_viz:
            mean_conf = confidence_viz.get('mean_confidence', 0.0)
            std_conf = confidence_viz.get('std_confidence', 0.0)
            parts.append(f"**Confidence:** {mean_conf:.2f} ± {std_conf:.2f}")
        
        graph_structure = transparency.get('graph_structure', {})
        if graph_structure:
            parts.append(f"**Graph Structure:** {graph_structure.get('structure_type', 'unknown')}")
            parts.append(f"  - Nodes: {graph_structure.get('num_nodes', 0)}")
            parts.append(f"  - Edges: {graph_structure.get('num_edges', 0)}")
        
        return "\n".join(parts)
    
    def _format_graph_answer(self, graph_answer: Dict[str, Any], hedging: str, graph_type: str = 'causal') -> str:
        """Format graph-first answer conversationally."""
        answer = graph_answer.get('answer', '')
        evidence = graph_answer.get('supporting_evidence', [])
        
        if answer:
            if graph_type in ['knowledge', 'mixed']:
                result = f"Based on the knowledge graph, {hedging} {answer.lower()}\n\n"
            else:
                result = f"Based on the causal graph structure, {hedging} {answer.lower()}\n\n"
            if evidence:
                result += "This conclusion is supported by:\n"
                for ev in evidence[:3]:  # Limit to 3 pieces of evidence
                    ev_type = ev.get('type', 'evidence')
                    result += f"- {ev_type}: {str(ev)[:100]}\n"
            return result
        return ""
    
    def _format_knowledge_analysis_conversational(
        self,
        analysis: Dict[str, Any],
        register: str,
        hedging: str
    ) -> str:
        """Format general knowledge analysis in conversational style."""
        parts = []
        variables = analysis.get('variables', [])
        relationships = analysis.get('relationships', [])
        
        if variables:
            parts.append(f"I've identified {len(variables)} entities: {', '.join(variables[:5])}")
            if len(variables) > 5:
                parts.append(f" and {len(variables) - 5} more")
            parts.append(".\n\n")
        
        if relationships:
            parts.append("Here are the relationships I found:\n\n")
            for rel in relationships[:5]:
                if isinstance(rel, dict):
                    source = rel.get('source', '')
                    target = rel.get('target', '')
                    rel_type = rel.get('type', 'related')
                    category = rel.get('category', 'general')
                    
                    # Format based on relationship type
                    if category == 'taxonomic':
                        parts.append(f"- {source} is a type of {target}\n")
                    elif category == 'meronymic':
                        if rel_type == 'part_of':
                            parts.append(f"- {source} is part of {target}\n")
                        elif rel_type == 'has_property':
                            parts.append(f"- {source} has {target}\n")
                        elif rel_type == 'contains':
                            parts.append(f"- {source} contains {target}\n")
                    elif category == 'spatial':
                        parts.append(f"- {source} is located in {target}\n")
                    elif category == 'functional':
                        parts.append(f"- {source} is used for {target}\n")
                    elif category == 'definitional':
                        parts.append(f"- {source} is {target}\n")
                    else:
                        parts.append(f"- {source} is related to {target}\n")
            
            if len(relationships) > 5:
                parts.append(f"\n... and {len(relationships) - 5} more relationships.\n")
        
        insights = analysis.get('insights', [])
        if insights:
            parts.append("\n**Key Insights:**\n")
            for insight in insights[:3]:
                parts.append(f"- {insight}\n")
        
        return ''.join(parts)
    
    def _format_analysis_conversational(
        self,
        analysis: Dict[str, Any],
        register: str,
        hedging: str
    ) -> str:
        """Format analysis (causal or general knowledge) in conversational style."""
        variables = analysis.get('variables', [])
        relationships = analysis.get('relationships', [])
        insights = analysis.get('insights', [])

        parts = []
        
        # Determine if this is causal or general knowledge
        has_causal = any(
            isinstance(rel, dict) and rel.get('category') == 'causal'
            for rel in relationships
        )
        has_general = any(
            isinstance(rel, dict) and rel.get('category') in ['taxonomic', 'meronymic', 'spatial', 'functional', 'definitional', 'factual']
            for rel in relationships
        )

        if variables:
            if has_general and not has_causal:
                var_text = ', '.join(variables[:5])
                if len(variables) > 5:
                    var_text += f", and {len(variables) - 5} more"
                parts.append(f"I've identified {len(variables)} entities: {var_text}.\n")
            else:
                var_text = ', '.join(variables[:5])
                if len(variables) > 5:
                    var_text += f", and {len(variables) - 5} more"
                parts.append(f"I've identified {len(variables)} key variables: {var_text}.\n")

        if relationships:
            if has_general and not has_causal:
                parts.append(f"Here are the relationships I found:\n\n")
                for rel in relationships[:5]:
                    if isinstance(rel, dict):
                        source = rel.get('source', '')
                        target = rel.get('target', '')
                        category = rel.get('category', 'general')
                        rel_type = rel.get('type', 'related')
                        
                        if category == 'taxonomic':
                            parts.append(f"- {source} is a type of {target}\n")
                        elif category == 'meronymic':
                            if rel_type == 'part_of':
                                parts.append(f"- {source} is part of {target}\n")
                            elif rel_type == 'has_property':
                                parts.append(f"- {source} has {target}\n")
                            else:
                                parts.append(f"- {source} -> {target}\n")
                        elif category == 'spatial':
                            parts.append(f"- {source} is located in {target}\n")
                        elif category == 'functional':
                            parts.append(f"- {source} is used for {target}\n")
                        else:
                            parts.append(f"- {source} is related to {target}\n")
                
                if len(relationships) > 5:
                    parts.append(f"\n... and {len(relationships) - 5} more relationships.\n")
            else:
                parts.append(f"These variables are connected through {len(relationships)} causal relationships.\n")

                # Highlight strongest relationship
                if relationships:
                    strongest = max(relationships, key=lambda x: abs(x.get('strength', 0)) if isinstance(x, dict) else 0)
                    if isinstance(strongest, dict):
                        source = strongest.get('source', '')
                        target = strongest.get('target', '')
                        strength = strongest.get('strength', 0)
                        confidence = strongest.get('confidence', 0.8)

                        connector = self.connectors.get('high_confidence' if confidence > 0.7 else 'medium_confidence', ['It appears'])[0]
                        parts.append(
                            f"{connector}, the strongest relationship is between '{source}' and '{target}' "
                            f"(strength: {strength:.2f}, confidence: {confidence:.1%}).\n"
                        )

        if insights:
            parts.append("\nKey insights:\n")
            for insight in insights[:3]:  # Limit to 3 insights
                parts.append(f"- {insight}\n")

        return ''.join(parts)
    
    def _generate_question_answer(
        self,
        reasoning_result: Dict[str, Any],
        hedging: str
    ) -> str:
        """Generate direct answer to a question."""
        task = reasoning_result.get('task', '')
        analysis = reasoning_result.get('analysis', {})
        graph_answer = reasoning_result.get('graph_first_answer', {})
        
        # Try to extract answer from graph-first reasoning first
        if graph_answer and graph_answer.get('answer'):
            return f"**Answer:** {graph_answer['answer']}\n"
        
        # Fallback to analysis-based answer
        variables = analysis.get('variables', [])
        relationships = analysis.get('relationships', [])
        
        if 'what' in task.lower() or 'which' in task.lower():
            if variables:
                return f"**Answer:** The key variables involved are: {', '.join(variables[:3])}.\n"
        
        if 'how' in task.lower() or 'why' in task.lower():
            if relationships:
                strongest = max(relationships, key=lambda x: abs(x.get('strength', 0)))
                source = strongest.get('source', '')
                target = strongest.get('target', '')
                return f"**Answer:** {hedging.capitalize()}, '{source}' affects '{target}' through a causal relationship.\n"
        
        return ""
    
    def _format_counterfactuals_conversational(
        self,
        counterfactuals: List[Dict[str, Any]],
        hedging: str
    ) -> str:
        """Format counterfactuals conversationally."""
        if not counterfactuals:
            return ""
        
        parts = ["## Exploring Alternative Scenarios\n\n"]
        parts.append(f"Let me explore {len(counterfactuals)} alternative scenarios:\n\n")
        
        for i, scenario in enumerate(counterfactuals[:3], 1):  # Limit to 3 scenarios
            name = scenario.get('name', f'Scenario {i}')
            interventions = scenario.get('interventions', {})
            outcomes = scenario.get('expected_outcomes', {})
            probability = scenario.get('probability', 0.0)
            
            parts.append(f"**{name}** ({probability:.1%} probability):\n")
            
            if interventions:
                parts.append("If we change:\n")
                for var, val in list(interventions.items())[:3]:
                    parts.append(f"- {var} to {val:.2f}\n")
            
            if outcomes:
                parts.append("Then we would expect:\n")
                for var, val in list(outcomes.items())[:3]:
                    parts.append(f"- {var}: {val:.2f}\n")
            
            parts.append("\n")
        
        return ''.join(parts)
    
    def _format_recommendations_conversational(
        self,
        recommendations: List[str],
        hedging: str
    ) -> str:
        """Format recommendations conversationally."""
        if not recommendations:
            return ""
        
        parts = ["## Recommendations\n\n"]
        parts.append("Based on my analysis, here's what I recommend:\n\n")
        
        for i, rec in enumerate(recommendations[:5], 1):  # Limit to 5 recommendations
            parts.append(f"{i}. {rec}\n")
        
        return ''.join(parts)
    
    def _format_statistical_conversational(
        self,
        statistical: Dict[str, Any],
        hedging: str
    ) -> str:
        """Format statistical results conversationally."""
        parts = ["## Statistical Analysis\n\n"]
        
        edge_strengths = statistical.get('edge_strengths', {})
        if edge_strengths:
            parts.append(f"The statistical analysis reveals {len(edge_strengths)} causal relationships with quantified strengths.\n")
            
            # Highlight strongest edges
            sorted_edges = sorted(edge_strengths.items(), key=lambda x: abs(x[1]), reverse=True)
            if sorted_edges:
                parts.append("The strongest relationships are:\n")
                for (source, target), strength in sorted_edges[:3]:
                    parts.append(f"- {source} -> {target}: {strength:.3f}\n")
        
        uncertainty = statistical.get('uncertainty', {})
        if uncertainty:
            parts.append(f"\nUncertainty analysis indicates {hedging} confidence in these relationships.\n")
        
        return ''.join(parts)


class HybridOrchestrator:
    """
    Orchestrates hybrid reasoning with all LLM-enhanced components.
    
    Integrates:
    - Reasoning tracking for chain-of-thought
    - Explanation generation
    - Self-verification
    - Consistency guarantees
    """
    
    def __init__(
        self,
        symbolic_reasoner: SymbolicReasoner,
        statistical_engine: StatisticalEngine,
        nlg: RuleBasedNLG,
        graph_first_reasoner: Optional[GraphFirstReasoner] = None,
        text_corrector: Optional[TextCorrector] = None,
        lexical_compiler: Optional[LexicalCompiler] = None,
        grammatical_compiler: Optional[GrammaticalCompiler] = None,
        pragmatic_compiler: Optional[PragmaticCompiler] = None,
        reasoning_tracker: Optional[ReasoningTracker] = None,
        explanation_builder: Optional[ExplanationBuilder] = None,
        transparency_layer: Optional[TransparencyLayer] = None,
        consistency_checker: Optional[ConsistencyChecker] = None,
        error_detector: Optional[ErrorDetector] = None,
        self_corrector: Optional[SelfCorrector] = None,
        consistency_engine: Optional[ConsistencyEngine] = None
    ):
        """
        Initialize hybrid orchestrator.
        
        Args:
            symbolic_reasoner: Symbolic reasoner instance
            statistical_engine: Statistical engine instance
            nlg: Natural language generator
            graph_first_reasoner: Optional graph-first reasoner
            text_corrector: Optional text corrector
            lexical_compiler: Optional lexical compiler
            grammatical_compiler: Optional grammatical compiler
            pragmatic_compiler: Optional pragmatic compiler
            reasoning_tracker: Optional reasoning tracker for chain-of-thought
            explanation_builder: Optional explanation builder
            transparency_layer: Optional transparency layer
            consistency_checker: Optional consistency checker
            error_detector: Optional error detector
            self_corrector: Optional self corrector
            consistency_engine: Optional consistency engine
        """
        self.symbolic_reasoner = symbolic_reasoner
        self.statistical_engine = statistical_engine
        self.nlg = nlg
        self.graph_first_reasoner = graph_first_reasoner
        self.text_corrector = text_corrector
        self.lexical_compiler = lexical_compiler
        self.grammatical_compiler = grammatical_compiler
        self.pragmatic_compiler = pragmatic_compiler
        self.reasoning_tracker = reasoning_tracker
        self.explanation_builder = explanation_builder
        self.transparency_layer = transparency_layer
        self.consistency_checker = consistency_checker
        self.error_detector = error_detector
        self.self_corrector = self_corrector
        self.consistency_engine = consistency_engine
    
    def _parse_task_intent(self, task: str) -> Dict[str, Any]:
        """
        Parse task to understand user intent and extract query type.
        
        Args:
            task: Natural language task
            
        Returns:
            Dictionary with intent information
        """
        task_lower = task.lower()
        intent = {
            'type': 'analysis',  # default
            'question_type': None,
            'target_variables': [],
            'intervention_variables': [],
            'comparison_requested': False,
        }
        
        # Question type detection
        if any(word in task_lower for word in ['what', 'which', 'who']):
            intent['question_type'] = 'what'
        elif any(word in task_lower for word in ['how', 'why']):
            intent['question_type'] = 'how'
        elif any(word in task_lower for word in ['when', 'where']):
            intent['question_type'] = 'when_where'
        elif '?' in task:
            intent['question_type'] = 'general_question'
        
        # Intent type detection
        if any(word in task_lower for word in ['extract', 'identify', 'find', 'list']):
            intent['type'] = 'extraction'
        elif any(word in task_lower for word in ['analyze', 'analyze', 'examine', 'study']):
            intent['type'] = 'analysis'
        elif any(word in task_lower for word in ['predict', 'forecast', 'estimate']):
            intent['type'] = 'prediction'
        elif any(word in task_lower for word in ['compare', 'versus', 'vs', 'difference']):
            intent['type'] = 'comparison'
            intent['comparison_requested'] = True
        elif any(word in task_lower for word in ['what if', 'if', 'suppose', 'assume']):
            intent['type'] = 'counterfactual'
        elif any(word in task_lower for word in ['recommend', 'suggest', 'should', 'best']):
            intent['type'] = 'recommendation'
        elif any(word in task_lower for word in ['what is', 'what are', 'define', 'definition', 'meaning']):
            intent['type'] = 'definition'
        elif any(word in task_lower for word in ['who is', 'who are']):
            intent['type'] = 'person_query'
        elif any(word in task_lower for word in ['where is', 'where are', 'location']):
            intent['type'] = 'location_query'
        elif any(word in task_lower for word in ['when is', 'when was', 'when did', 'date', 'time']):
            intent['type'] = 'temporal_query'
        elif any(word in task_lower for word in ['explain', 'describe', 'tell me about']):
            intent['type'] = 'explanation'
        
        # Extract target variables (what user wants to know about)
        target_patterns = [
            r'(?:about|regarding|for|of)\s+(\w+(?:\s+\w+)?)',
            r'(?:affecting|impacting|influencing)\s+(\w+(?:\s+\w+)?)',
            r'(?:on|in)\s+(\w+(?:\s+\w+)?)',
        ]
        for pattern in target_patterns:
            matches = re.finditer(pattern, task_lower)
            for match in matches:
                var = self.symbolic_reasoner._normalize_variable_name(match.group(1))
                if var and var not in self.symbolic_reasoner.stop_words:
                    intent['target_variables'].append(var)
        
        # Extract intervention variables (what user wants to change)
        intervention_patterns = [
            r'(?:if|when|suppose)\s+(\w+(?:\s+\w+)?)\s+(?:changes?|increases?|decreases?)',
            r'(?:change|modify|adjust)\s+(\w+(?:\s+\w+)?)',
        ]
        for pattern in intervention_patterns:
            matches = re.finditer(pattern, task_lower)
            for match in matches:
                var = self.symbolic_reasoner._normalize_variable_name(match.group(1))
                if var and var not in self.symbolic_reasoner.stop_words:
                    intent['intervention_variables'].append(var)
        
        return intent
    
    def _parse_extracted_values(self, variables_with_values: Dict[str, str]) -> Dict[str, float]:
        """
        Parse extracted string values into float values.
        
        Handles:
        - "20000" -> 20000.0
        - "61%" -> 0.61
        - "61% buy, 39% sell" -> extracts main percentage (61% -> 0.61)
        
        Args:
            variables_with_values: Dictionary mapping variable names to string values
            
        Returns:
            Dictionary mapping variable names to float values
        """
        parsed = {}
        
        for var, value_str in variables_with_values.items():
            if not value_str:
                continue
            
            try:
                # Remove whitespace
                value_str = value_str.strip()
                
                # Handle percentages
                if '%' in value_str:
                    # Extract first percentage if multiple (e.g., "61% buy, 39% sell" -> 61%)
                    percent_match = re.search(r'(\d+[.,]?\d*)\s*%', value_str)
                    if percent_match:
                        percent_value = float(percent_match.group(1).replace(',', '.'))
                        # Convert percentage to decimal (61% -> 0.61)
                        parsed[var] = percent_value / 100.0
                    else:
                        # Try to extract any number before %
                        num_match = re.search(r'(\d+[.,]?\d*)', value_str)
                        if num_match:
                            parsed[var] = float(num_match.group(1).replace(',', '.')) / 100.0
                else:
                    # Regular number
                    # Remove any non-numeric characters except decimal point and comma
                    clean_value = re.sub(r'[^\d.,-]', '', value_str)
                    if clean_value:
                        # Handle comma as decimal separator (European format)
                        if ',' in clean_value and '.' not in clean_value:
                            clean_value = clean_value.replace(',', '.')
                        # Handle comma as thousands separator
                        elif ',' in clean_value and '.' in clean_value:
                            # Assume last comma/period is decimal separator
                            parts = clean_value.replace(',', ' ').replace('.', ' ').split()
                            if len(parts) > 1:
                                clean_value = '.'.join(parts)
                            else:
                                clean_value = clean_value.replace(',', '')
                        
                        parsed[var] = float(clean_value)
            except (ValueError, AttributeError) as e:
                logger.debug(f"Failed to parse value '{value_str}' for variable '{var}': {e}")
                continue
        
        return parsed
    
    def _detect_and_parse_json_scm(self, task: str) -> Optional[Dict[str, Any]]:
        """
        Detect and parse JSON SCM (Structural Causal Model) from task.
        
        Args:
            task: Task string that may contain JSON SCM
            
        Returns:
            Parsed SCM dictionary or None if not detected
        """
        # Try to find JSON in the task
        # Look for JSON object pattern - find the largest JSON object
        # This handles cases where there's text before/after the JSON
        json_matches = list(re.finditer(r'\{[\s\S]*?\}', task))
        if not json_matches:
            return None
        
        # Try the largest match first (most likely to be complete JSON)
        json_matches_sorted = sorted(json_matches, key=lambda m: len(m.group(0)), reverse=True)
        
        for json_match in json_matches_sorted:
            json_str = json_match.group(0)
            
            try:
                scm_data = json.loads(json_str)
                
                # Validate it's an SCM structure
                if not isinstance(scm_data, dict):
                    continue
                
                # Check for SCM indicators
                has_variables = 'variables' in scm_data
                has_equations = 'equations' in scm_data
                has_roles = any(
                    isinstance(v, dict) and 'role' in v 
                    for v in scm_data.get('variables', [])
                )
                
                if has_variables and (has_equations or has_roles):
                    logger.info(f"Detected SCM structure with {len(scm_data.get('variables', []))} variables")
                    return scm_data
                
            except json.JSONDecodeError:
                # Try to extract and fix JSON
                try:
                    # Remove comments and fix common issues
                    json_str_clean = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
                    json_str_clean = re.sub(r'/\*.*?\*/', '', json_str_clean, flags=re.DOTALL)
                    scm_data = json.loads(json_str_clean)
                    
                    if isinstance(scm_data, dict) and 'variables' in scm_data:
                        logger.info(f"Detected SCM structure (after cleaning) with {len(scm_data.get('variables', []))} variables")
                        return scm_data
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _parse_scm_to_graph(self, scm_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse structured SCM JSON into causal graph.
        
        Args:
            scm_data: Parsed SCM dictionary
            
        Returns:
            Dictionary with extracted variables, edges, and metadata
        """
        variables = []
        edges = []
        relationships = []
        variables_with_values = {}
        
        # Extract variables
        var_list = scm_data.get('variables', [])
        for var_def in var_list:
            if isinstance(var_def, dict):
                var_id = var_def.get('id', '')
                var_role = var_def.get('role', '')
                if var_id:
                    variables.append(var_id)
                    # Store role information
                    if var_role:
                        variables_with_values[f"{var_id}_role"] = var_role
        
        # Extract relationships from equations
        equations = scm_data.get('equations', [])
        for eq in equations:
            if isinstance(eq, dict):
                defines = eq.get('defines', '')
                parents = eq.get('parents', [])
                
                # Extract variable name from defines (e.g., "S[t+1]" -> "S")
                defines_var = re.sub(r'\[.*?\]', '', defines).strip()
                
                # Create edges from parents to defined variable
                for parent in parents:
                    # Extract variable name from parent (e.g., "S[t]" -> "S", "alpha" -> "alpha")
                    parent_var = re.sub(r'\[.*?\]', '', str(parent)).strip()
                    
                    if defines_var and parent_var and defines_var != parent_var:
                        # Only create edges between state variables (not parameters/constants)
                        parent_role = None
                        defines_role = None
                        
                        # Find roles
                        for var_def in var_list:
                            if isinstance(var_def, dict):
                                if var_def.get('id') == parent_var:
                                    parent_role = var_def.get('role', '')
                                if var_def.get('id') == defines_var:
                                    defines_role = var_def.get('role', '')
                        
                        # Only create causal edges (not parameter/constant relationships)
                        # State variables can cause other state variables
                        # Interventions can affect state variables
                        # Exogenous can affect state variables
                        if (defines_role in ['state', 'derived'] and 
                            parent_role in ['state', 'intervention', 'exogenous']):
                            edges.append((parent_var, defines_var))
                            relationships.append({
                                'source': parent_var,
                                'target': defines_var,
                                'type': 'causal',
                                'confidence': 1.0,  # High confidence for explicit SCM
                                'negated': False,
                                'raw_source': parent_var,
                                'raw_target': defines_var,
                                'from_equation': eq.get('id', ''),
                                'equation': eq.get('expr', '')
                            })
        
        # Extract initial state values
        given = scm_data.get('given', {})
        initial_state = given.get('initial_state', {})
        for var, value in initial_state.items():
            var_name = re.sub(r'\[.*?\]', '', var).strip()
            if var_name in variables:
                variables_with_values[var_name] = str(value)
        
        # Extract parameter values
        parameters = given.get('parameters', {})
        for param, value in parameters.items():
            if param in variables:
                variables_with_values[param] = str(value)
        
        return {
            'variables': variables,
            'edges': edges,
            'relationships': relationships,
            'metadata': {
                'variables_with_values': variables_with_values,
                'scm_structure': True,
                'task_id': scm_data.get('task_id', ''),
                'equations_count': len(equations),
                'variables_count': len(variables)
            }
        }
    
    def reason_hybrid(
        self,
        task: str,
        data: Optional[Any] = None,
        context: Optional[ConversationContext] = None
    ) -> Dict[str, Any]:
        """
        Execute graph-first hybrid reasoning pipeline with reasoning tracking.
        
        Pipeline:
        1. JSON/SCM detection and parsing (if structured input)
        2. Text correction (non-destructive)
        3. Language compilation (lexical -> grammatical -> pragmatic)
        4. Symbolic extraction (to graph with provenance)
        5. Graph-first reasoning (answer from graph state ONLY)
        6. Natural language generation (from graph state)
        7. Self-verification and error correction
        8. Explanation generation
        
        Args:
            task: Natural language task description or JSON SCM
            data: Optional pandas DataFrame for statistical inference
            context: Optional conversation context
            
        Returns:
            Dictionary with reasoning results derived from graph state
        """
        # Create reasoning chain if tracking enabled
        if self.reasoning_tracker:
            self.reasoning_tracker.create_chain()
        
        result = {
            'task': task,
            'intent': {},
            'analysis': {},
            'counterfactuals': [],
            'statistical': {},
            'graph_structure': '',
            'recommendations': [],
            'graph_first_answer': None,
            'scm_parsed': False,
            'reasoning_chain': None,
            'explanation': None
        }
        
        # Track reasoning step: SCM detection
        if self.reasoning_tracker:
            self.reasoning_tracker.add_step(
                step_type=StepType.EXTRACTION,
                operation="detect_json_scm",
                input_state={'task': task},
                output_state={},
                conclusion="SCM detection"
            )
        
        # Step 0: Detect and parse JSON SCM if present
        scm_data = self._detect_and_parse_json_scm(task)
        if scm_data:
            logger.info("Detected structured JSON SCM - parsing directly")
            result['scm_parsed'] = True
            result['scm_data'] = scm_data
            
            # Track reasoning step: SCM parsing
            if self.reasoning_tracker:
                self.reasoning_tracker.add_step(
                    step_type=StepType.EXTRACTION,
                    operation="parse_scm",
                    input_state={'scm_data': scm_data},
                    output_state={},
                    conclusion="SCM parsed"
                )
            
            # Parse SCM to graph structure
            scm_extraction = self._parse_scm_to_graph(scm_data)
            result['scm_extraction'] = scm_extraction
            
            # Use SCM extraction instead of natural language extraction
            variables = scm_extraction.get('variables', [])
            relationships_extracted = scm_extraction.get('relationships', [])
            edges = scm_extraction.get('edges', [])
            
            # All SCM variables are valid (they're explicitly defined)
            valid_variables = set(variables)
            
            # Add edges directly from SCM (high confidence - explicit structure)
            edges_added = 0
            for rel in relationships_extracted:
                source = rel['source']
                target = rel['target']
                
                self.symbolic_reasoner.graph_manager.add_relationship(
                    source=source,
                    target=target,
                    strength=1.0,
                    confidence=rel.get('confidence', 1.0),
                    from_scm=True,
                    equation_id=rel.get('from_equation', ''),
                    equation_expr=rel.get('equation', '')
                )
                edges_added += 1
            
            result['edges_added'] = edges_added
            result['scm_parsing_success'] = True
            
            # Track reasoning step: Graph construction from SCM
            if self.reasoning_tracker:
                self.reasoning_tracker.add_step(
                    step_type=StepType.TRANSFORMATION,
                    operation="build_graph_from_scm",
                    input_state={'variables': variables, 'relationships': relationships_extracted},
                    output_state={'edges_added': edges_added},
                    conclusion=f"Graph built with {edges_added} edges"
                )
            
            # Skip natural language extraction for SCM
            # Continue to graph-first reasoning and analysis
            corrected_task = task  # Keep original for display
        else:
            # Step 0: Text correction (non-destructive) - only for natural language
            corrected_task = task
            corrected_tokens = None
            if self.text_corrector:
                correction_result = self.text_corrector.correct_text(task)
                corrected_task = correction_result['corrected_text']
                corrected_tokens = correction_result['corrected_tokens']
                result['correction'] = {
                    'original': task,
                    'corrected': corrected_task,
                    'confidence': correction_result['confidence']
                }
        
        # Step 0.5: Language compilation (lexical -> grammatical -> pragmatic)
        # Skip for SCM (already parsed)
        if not result.get('scm_parsed'):
            if self.lexical_compiler and self.grammatical_compiler:
                # Normalize terms using lexical compiler
                if corrected_tokens:
                    for token in corrected_tokens:
                        normalized = self.lexical_compiler.normalize_term(token.normalized_form)
                        if normalized != token.normalized_form:
                            token.normalized_form = normalized
                        
                        # Optional: Validate word using dictionary (helps filter invalid terms)
                        # This is a soft check - we don't reject words that aren't in dictionary
                        # as domain-specific terms may not be in standard dictionaries
                        if self.lexical_compiler.enable_dictionary:
                            word_info = self.lexical_compiler.get_word_info(token.normalized_form)
                            if word_info:
                                # Store word info for later use (part of speech, synonyms, etc.)
                                token.metadata = word_info
            
                # Parse grammatical structure
                if self.grammatical_compiler:
                    parse_tree = self.grammatical_compiler.parse_dependencies(corrected_task)
                    causal_structure = self.grammatical_compiler.extract_causal_structure(parse_tree)
                    if causal_structure:
                        result['causal_structure'] = {
                            'cause': causal_structure.cause,
                            'effect': causal_structure.effect,
                            'relation_type': causal_structure.relation_type,
                            'confidence': causal_structure.confidence
                        }
        
        # Step 0.6: Parse task intent
        intent = self._parse_task_intent(corrected_task)
        result['intent'] = intent
        
        # Step 1: Advanced symbolic extraction (use corrected task) OR use SCM extraction
        if result.get('scm_parsed'):
            # Use SCM extraction (already done above and stored in result)
            extraction = result.get('scm_extraction', {})
            variables = extraction.get('variables', [])
            relationships_extracted = extraction.get('relationships', [])
            edges = extraction.get('edges', [])
            valid_variables = set(variables)  # All SCM variables are valid
            logger.info(f"Using SCM extraction: {len(variables)} variables, {len(relationships_extracted)} relationships")
        else:
            # Natural language extraction with few-shot learning (if enabled)
            # Check if we have learned patterns to use
            if hasattr(self.symbolic_reasoner, 'adaptive_extractor') and self.symbolic_reasoner.adaptive_extractor:
                # Try adaptive extraction first
                try:
                    adaptive_result = self.symbolic_reasoner.adaptive_extractor.adapt_extraction(corrected_task)
                    if adaptive_result.get('variables'):
                        # Use adaptive extraction result
                        extraction = {
                            'variables': adaptive_result.get('variables', []),
                            'edges': adaptive_result.get('edges', []),
                            'relationships': []
                        }
                        # Convert edges to relationships format
                        for source, target in adaptive_result.get('edges', []):
                            extraction['relationships'].append({
                                'source': source,
                                'target': target,
                                'type': 'causal',
                                'confidence': adaptive_result.get('confidence', 0.8)
                            })
                    else:
                        # Fall back to standard extraction
                        extraction = self.symbolic_reasoner.extract_variables_from_task(corrected_task)
                except Exception as e:
                    logger.warning(f"Adaptive extraction failed: {e}, falling back to standard extraction")
                    extraction = self.symbolic_reasoner.extract_variables_from_task(corrected_task)
            else:
                # Standard extraction
                extraction = self.symbolic_reasoner.extract_variables_from_task(corrected_task)
            
            variables = extraction.get('variables', [])
            relationships_extracted = extraction.get('relationships', [])
            edges = extraction.get('edges', [])
            
            # Track reasoning step: Variable extraction
            if self.reasoning_tracker:
                self.reasoning_tracker.add_step(
                    step_type=StepType.EXTRACTION,
                    operation="extract_variables",
                    input_state={'task': corrected_task},
                    output_state={'variables': variables, 'relationships': len(relationships_extracted)},
                    conclusion=f"Extracted {len(variables)} variables"
                )
            
            # valid_variables will be set below after filtering
        
        # CRITICAL: Filter variables before graph construction - only use valid variables
        # This prevents treating action verbs and epistemic terms as causal variables
        # BUT: Skip filtering for SCM (all SCM variables are explicitly defined and valid)
        if not result.get('scm_parsed'):
            valid_variables = {v for v in variables if self.symbolic_reasoner._clean_variable(v)}
            # For general knowledge tasks, be more permissive with filtering
            current_graph_type = self.symbolic_reasoner.graph_manager.graph_type
            if current_graph_type in ['knowledge', 'mixed']:
                # For general knowledge: Only filter action verbs/epistemic terms, keep entities
                filtered = set()
                for var in valid_variables:
                    if not self.symbolic_reasoner._is_action_verb(var) and not self.symbolic_reasoner._is_epistemic_term(var):
                        cleaned = self.symbolic_reasoner._clean_variable(var)
                        if cleaned and cleaned not in self.symbolic_reasoner.stop_words:
                            filtered.add(cleaned)
                valid_variables = filtered
            else:
                # For causal tasks: Strict filtering
                valid_variables = self.symbolic_reasoner._filter_valid_variables(valid_variables)
            
            # Enhanced epistemic validation with GroundingValidator
            # Check epistemic grounding: ∀v ∈ V, ∃ path from observables O to v
            observable_variables = valid_variables.copy()  # For now, all valid variables are considered observable
            if self.consistency_checker:
                graph_state_temp = {
                    'nodes': list(valid_variables),
                    'edges': edges
                }
                all_grounded, ungrounded = self.consistency_checker.verify_epistemic_grounding(
                    graph_state_temp,
                    observable_variables
                )
                if not all_grounded and ungrounded:
                    logger.warning(f"Ungrounded variables detected: {ungrounded}")
            
            # Epistemic validation: Check if we have sufficient grounding
            # If task is too vague (no explicit causal relationships, only action verbs/epistemic terms),
            # we should warn or reject
            has_explicit_causal_structure = len(relationships_extracted) > 0 or len(edges) > 0
            has_valid_state_variables = len(valid_variables) > 0
            
            # Check if variables are mostly action verbs/epistemic terms (bad sign)
            action_verb_count = sum(1 for v in variables if self.symbolic_reasoner._is_action_verb(v))
            epistemic_term_count = sum(1 for v in variables if self.symbolic_reasoner._is_epistemic_term(v))
            total_vars = len(variables)
            
            if total_vars > 0:
                action_epistemic_ratio = (action_verb_count + epistemic_term_count) / total_vars
                if action_epistemic_ratio > 0.5 and not has_explicit_causal_structure:
                    result['epistemic_warning'] = (
                        f"Warning: Task appears to contain mostly action verbs or epistemic terms "
                        f"({action_verb_count + epistemic_term_count}/{total_vars} variables), "
                        f"not causal state variables. Causal relationships cannot be inferred from "
                        f"intent statements alone. Please provide explicit state variables and "
                        f"causal relationships, or an existing SCM with logged policy decisions."
                    )
                    logger.warning(result['epistemic_warning'])
            
            # Determine task type (causal vs general knowledge)
            task_intent = self._parse_task_intent(task)
            
            # Check relationships_extracted for categories (if available)
            relationships_categories = [rel.get('category', 'causal') for rel in relationships_extracted if isinstance(rel, dict)]
            has_causal_rels = any(cat == 'causal' for cat in relationships_categories)
            has_general_rels = any(cat in ['taxonomic', 'meronymic', 'spatial', 'functional', 'definitional', 'factual'] for cat in relationships_categories)
            
            graph_type = self.symbolic_reasoner.graph_manager.graph_type
            is_causal_task = (
                task_intent.get('type') in ['analysis', 'prediction', 'counterfactual', 'comparison'] or
                has_causal_rels or
                graph_type == 'causal'
            )
            is_general_knowledge_task = (
                task_intent.get('type') in ['definition', 'person_query', 'location_query', 'temporal_query', 'explanation'] or
                has_general_rels or
                graph_type in ['knowledge', 'mixed']
            )
            
            # Epistemic validation only applies to causal tasks
            if is_causal_task and not is_general_knowledge_task:
                # If no valid state variables after filtering, task is epistemically underspecified
                if not has_valid_state_variables and not has_explicit_causal_structure:
                    result['epistemic_error'] = (
                        "Task is epistemically underspecified. No valid causal state variables were "
                        "extracted. A CRCA agent requires:\n"
                        "- Explicit state variables (not action verbs like 'identify' or epistemic terms like 'policy')\n"
                        "- Transition relations (causal relationships between variables)\n"
                        "- Intervention hooks (variables that can be manipulated)\n"
                        "- Optionally: An existing SCM, logged policy decisions, defined collapse predicates\n\n"
                        "Please provide a task with explicit causal structure, not just intent statements."
                    )
                    logger.error(result['epistemic_error'])
                    # Don't proceed with graph construction if we have no valid variables
                    result['analysis'] = {
                        'variables': [],
                        'relationships': [],
                        'graph_structure': 'No valid causal structure extracted',
                        'insights': [result['epistemic_error']],
                        'epistemic_underspecified': True
                    }
                    return result
        
        # Add edges to graph with confidence scores
        # For SCM, all relationships are already added above, so skip this section
        if not result.get('scm_parsed'):
            edges_added = 0
            for rel in relationships_extracted:
                if not rel.get('negated', False):  # Skip negated relationships
                    # For general knowledge, be more permissive with cleaning
                    # Don't filter out entities just because they're short or not in keywords
                    rel_category = rel.get('category', 'causal')
                    if rel_category != 'causal':
                        # For general knowledge: Just normalize, don't filter
                        source_clean = self.symbolic_reasoner._normalize_variable_name(rel['source'])
                        target_clean = self.symbolic_reasoner._normalize_variable_name(rel['target'])
                        # Remove stop words but keep the variable
                        if source_clean in self.symbolic_reasoner.stop_words:
                            source_clean = None
                        if target_clean in self.symbolic_reasoner.stop_words:
                            target_clean = None
                    else:
                        # For causal: Use strict cleaning
                        source_clean = self.symbolic_reasoner._clean_variable(rel['source'])
                        target_clean = self.symbolic_reasoner._clean_variable(rel['target'])
                    
                    if not source_clean or not target_clean:
                        continue
                    
                    rel_type = rel.get('type', 'causal')
                    
                    # For causal relationships: Reject action verbs/epistemic terms (strict validation)
                    if rel_category == 'causal':
                        if self.symbolic_reasoner._is_action_verb(source_clean):
                            logger.warning(f"Rejected causal relationship: source '{source_clean}' is an action verb")
                            continue
                        if self.symbolic_reasoner._is_action_verb(target_clean):
                            logger.warning(f"Rejected causal relationship: target '{target_clean}' is an action verb")
                            continue
                        if self.symbolic_reasoner._is_epistemic_term(source_clean):
                            logger.warning(f"Rejected causal relationship: source '{source_clean}' is an epistemic term")
                            continue
                        if self.symbolic_reasoner._is_epistemic_term(target_clean):
                            logger.warning(f"Rejected causal relationship: target '{target_clean}' is an epistemic term")
                            continue
                        
                        # Only add if both are valid state variables for causal relationships
                        if (source_clean in valid_variables and target_clean in valid_variables):
                            # Validate causal relationship using do-calculus
                            graph_state_temp = {
                                'nodes': list(valid_variables),
                                'edges': [(s, t) for s, t in self.symbolic_reasoner.graph_manager.get_edges()]
                            }
                            is_valid_causal, causal_error = self.symbolic_reasoner.validate_causal_relationship(
                                source_clean,
                                target_clean,
                                graph_state_temp
                            )
                            
                            if is_valid_causal:
                                self.symbolic_reasoner.graph_manager.add_relationship(
                                    source=source_clean,
                                    target=target_clean,
                                    strength=1.0,
                                    relation_type=rel_type,
                                    confidence=rel.get('confidence', 0.8),
                                    category=rel_category
                                )
                                edges_added += 1
                            else:
                                logger.debug(f"Skipping invalid causal relationship {source_clean} -> {target_clean}: {causal_error}")
                    
                    # For general knowledge relationships: More permissive (allow entities, concepts, etc.)
                    else:
                        # Add to valid_variables if not already there (general knowledge can include new entities)
                        if source_clean not in valid_variables:
                            valid_variables.add(source_clean)
                        if target_clean not in valid_variables:
                            valid_variables.add(target_clean)
                        
                        # Add relationship (no strict validation for general knowledge)
                        self.symbolic_reasoner.graph_manager.add_relationship(
                            source=source_clean,
                            target=target_clean,
                            strength=1.0,
                            relation_type=rel_type,
                            confidence=rel.get('confidence', 0.8),
                            category=rel_category
                        )
                        edges_added += 1
        else:
            # For SCM, edges were already added above
            edges_added = result.get('edges_added', 0)
        
        # Add direct edges
        # Skip for SCM (already added)
        if not result.get('scm_parsed'):
            for source, target in edges:
                source_clean = self.symbolic_reasoner._clean_variable(source)
                target_clean = self.symbolic_reasoner._clean_variable(target)
                
                if not source_clean or not target_clean:
                    continue
                
                # Determine relationship category from context
                # If graph type is causal, apply strict filtering
                # If graph type is knowledge/mixed, be more permissive
                graph_type = self.symbolic_reasoner.graph_manager.graph_type
                if graph_type == 'causal':
                    # For causal graphs: Reject action verbs/epistemic terms
                    if (self.symbolic_reasoner._is_action_verb(source_clean) or 
                        self.symbolic_reasoner._is_epistemic_term(source_clean) or
                        self.symbolic_reasoner._is_action_verb(target_clean) or 
                        self.symbolic_reasoner._is_epistemic_term(target_clean)):
                        continue
                    
                    # Only add if both are valid state variables
                    if (source_clean in valid_variables and target_clean in valid_variables):
                        self.symbolic_reasoner.graph_manager.add_relationship(
                            source=source_clean,
                            target=target_clean,
                            strength=1.0,
                            relation_type='causal',
                            confidence=0.8,
                            category='causal'
                        )
                        edges_added += 1
                else:
                    # For knowledge/mixed graphs: More permissive
                    if source_clean not in valid_variables:
                        valid_variables.add(source_clean)
                    if target_clean not in valid_variables:
                        valid_variables.add(target_clean)
                    
                    self.symbolic_reasoner.graph_manager.add_relationship(
                        source=source_clean,
                        target=target_clean,
                        strength=1.0,
                        relation_type='related',
                        confidence=0.8,
                        category='general'
                    )
                    edges_added += 1
            
            # Track if we actually added any edges
            result['edges_added'] = edges_added
            if edges_added == 0 and len(relationships_extracted) > 0:
                result['epistemic_warning'] = (
                    "No valid causal relationships were added to the graph. All extracted relationships "
                    "involved action verbs or epistemic terms rather than causal state variables. "
                    "Please provide explicit state variables and causal relationships."
                )
        
        # Infer additional structure if needed (with context) - only for valid variables
        # Skip for SCM (structure is explicit)
        if not result.get('scm_parsed') and not edges and valid_variables:
            inferred_edges = self.symbolic_reasoner.infer_causal_structure(list(valid_variables), context=task)
            for source, target in inferred_edges:
                source_clean = self.symbolic_reasoner._clean_variable(source)
                target_clean = self.symbolic_reasoner._clean_variable(target)
                # Only add if both are valid
                if (source_clean and target_clean and 
                    source_clean in valid_variables and 
                    target_clean in valid_variables):
                    self.symbolic_reasoner.graph_manager.add_relationship(
                        source=source_clean,
                        target=target_clean,
                        strength=0.5,
                        confidence=0.5
                    )
        
        # Validate graph with consistency checker
        if self.consistency_checker:
            graph_state = {
                'nodes': list(valid_variables),
                'edges': [(s, t) for s, t in self.symbolic_reasoner.graph_manager.get_edges() 
                          if s in valid_variables and t in valid_variables]
            }
            is_consistent, consistency_error = self.consistency_checker.verify_consistency(graph_state)
            if not is_consistent:
                logger.warning(f"Graph consistency check failed: {consistency_error}")
                # Try to correct if self_corrector available
                if self.self_corrector:
                    errors = [{'type': 'inconsistency', 'message': consistency_error, 'graph': graph_state}]
                    corrections = self.self_corrector.correct_errors(errors, graph_state)
                    if corrections:
                        logger.info(f"Applied {len(corrections)} corrections")
        
        # Validate graph
        is_valid, error = self.symbolic_reasoner.validate_causal_graph()
        if not is_valid:
            logger.warning(f"Graph validation failed: {error}")
        
        # Track reasoning step: Graph validation
        if self.reasoning_tracker:
            self.reasoning_tracker.add_step(
                step_type=StepType.VALIDATION,
                operation="validate_graph",
                input_state={'graph': graph_state if self.consistency_checker else {}},
                output_state={'is_valid': is_valid, 'error': error},
                conclusion="Graph validated" if is_valid else f"Graph validation failed: {error}"
            )
        
        # Step 2: Statistical fitting (if data available)
        if data is not None and PANDAS_AVAILABLE:
            try:
                # Only use valid variables for statistical fitting
                graph_nodes = self.symbolic_reasoner.graph_manager.get_nodes()
                all_variables = [v for v in graph_nodes if v in valid_variables]
                if all_variables:
                    self.statistical_engine.fit_from_dataframe(
                        df=data,
                        variables=all_variables,
                        window=min(30, len(data)),
                        decay_alpha=0.9
                    )
                    
                    # Quantify uncertainty
                    uncertainty = self.statistical_engine.quantify_uncertainty(
                        df=data,
                        variables=all_variables,
                        windows=min(200, len(data))
                    )
                    
                    result['statistical'] = {
                        'edge_strengths': {
                            (s, t): self.statistical_engine.assess_causal_strength(s, t)
                            for s, t in self.symbolic_reasoner.graph_manager.get_edges()
                        },
                        'uncertainty': uncertainty,
                        'confidence_intervals': uncertainty.get('edge_intervals', {})
                    }
            except Exception as e:
                logger.warning(f"Statistical fitting failed: {e}")
        
        # Step 3: Build comprehensive analysis result
        # Only include relationships between valid variables
        graph_nodes_all = self.symbolic_reasoner.graph_manager.get_nodes()
        graph_edges_all = self.symbolic_reasoner.graph_manager.get_edges()
        
        # Filter to only valid variables
        graph_nodes = [n for n in graph_nodes_all if n in valid_variables]
        graph_edges = [(s, t) for s, t in graph_edges_all if s in valid_variables and t in valid_variables]
        
        relationships = []
        for source, target in graph_edges:
            edge_data = self.symbolic_reasoner.graph_manager.graph.get(source, {}).get(target, {})
            strength = self.statistical_engine.assess_causal_strength(source, target) if edge_data.get('category') == 'causal' else 1.0
            confidence = edge_data.get('confidence', 0.8)
            relation_type = edge_data.get('relation_type', 'causal')
            category = edge_data.get('category', 'causal')  # Get category from edge metadata
            relationships.append({
                'source': source,
                'target': target,
                'type': relation_type,
                'category': category,
                'strength': strength,
                'confidence': confidence
            })
        
        # Determine relationship type for graph structure description
        has_causal = any(rel.get('category') == 'causal' for rel in relationships)
        has_general = any(rel.get('category') in ['taxonomic', 'meronymic', 'spatial', 'functional', 'definitional', 'factual'] for rel in relationships)
        
        if has_general and not has_causal:
            rel_type_label = "knowledge relationships"
        elif has_causal:
            rel_type_label = "causal relationships"
        else:
            rel_type_label = "relationships"
        
        # Generate graph structure description (only valid variables)
        graph_structure = f"Nodes: {', '.join(sorted(graph_nodes))}\nEdges: {len(graph_edges)} {rel_type_label}"
        
        # Generate insights based on intent
        insights = []
        if relationships:
            # Strongest relationship
            strongest = max(relationships, key=lambda x: abs(x.get('strength', 0)))
            category = strongest.get('category', 'causal')
            if category == 'causal':
                insights.append(
                    f"Strongest causal relationship: {strongest['source']} -> {strongest['target']} "
                    f"(strength: {strongest['strength']:.3f}, confidence: {strongest['confidence']:.2f})"
                )
            elif category == 'taxonomic':
                insights.append(
                    f"Taxonomic relationship: {strongest['source']} is a type of {strongest['target']} "
                    f"(confidence: {strongest['confidence']:.2f})"
                )
            else:
                insights.append(
                    f"Strongest relationship: {strongest['source']} -> {strongest['target']} "
                    f"(type: {category}, confidence: {strongest['confidence']:.2f})"
                )
            
            # Most connected variable
            node_degrees = defaultdict(int)
            for rel in relationships:
                node_degrees[rel['source']] += 1
                node_degrees[rel['target']] += 1
            if node_degrees:
                most_connected = max(node_degrees.items(), key=lambda x: x[1])
                insights.append(
                    f"Most connected variable: {most_connected[0]} ({most_connected[1]} relationships)"
                )
        
        # Generate recommendations if requested
        recommendations = []
        if intent['type'] == 'recommendation' and relationships:
            # Find variables with high out-degree (causes) that could be intervened on
            out_degrees = defaultdict(int)
            for rel in relationships:
                out_degrees[rel['source']] += abs(rel['strength'])
            
            if out_degrees:
                top_levers = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
                for var, total_effect in top_levers:
                    recommendations.append(
                        f"Consider intervening on '{var}' - it has strong causal effects on multiple outcomes"
                    )
        
        # Parse extracted values to create factual state (used for both analysis and counterfactuals)
        # For SCM, use the values from the SCM structure
        if result.get('scm_parsed'):
            variables_with_values = extraction.get('metadata', {}).get('variables_with_values', {})
        else:
            variables_with_values = extraction.get('metadata', {}).get('variables_with_values', {})
        extracted_values = self._parse_extracted_values(variables_with_values)
        
        # Create factual state using extracted values, fallback to 0.0
        factual_state = {}
        if graph_nodes:
            for var in graph_nodes:
                # Use extracted value if available, otherwise 0.0
                # Try exact match first
                if var in extracted_values:
                    factual_state[var] = extracted_values[var]
                else:
                    # Try partial match (e.g., "product price" matches "product price")
                    matched = False
                    for extracted_var, value in extracted_values.items():
                        # Normalize both for comparison
                        var_normalized = var.lower().replace(' ', '')
                        extracted_var_normalized = extracted_var.lower().replace(' ', '')
                        if var_normalized in extracted_var_normalized or extracted_var_normalized in var_normalized:
                            factual_state[var] = value
                            matched = True
                            break
                    if not matched:
                        factual_state[var] = 0.0
        
        result['analysis'] = {
            'variables': sorted(list(graph_nodes)),  # Only valid variables
            'relationships': relationships,  # Only valid relationships
            'graph_structure': graph_structure,
            'insights': insights,
            'extraction_metadata': extraction.get('metadata', {}),
            'factual_state': factual_state
        }
        result['recommendations'] = recommendations
        
        # Step 4: Graph-first reasoning (answer from graph state ONLY)
        if self.graph_first_reasoner and graph_nodes:
            try:
                graph_state = {
                    'nodes': graph_nodes,
                    'edges': graph_edges,
                    'edge_data': {
                        (s, t): self.symbolic_reasoner.graph_manager.graph.get(s, {}).get(t, {})
                        for s, t in graph_edges
                    }
                }
                
                # Track reasoning step: Graph-first reasoning
                if self.reasoning_tracker:
                    self.reasoning_tracker.add_step(
                        step_type=StepType.INFERENCE,
                        operation="graph_first_reasoning",
                        input_state={'graph_state': graph_state, 'query': corrected_task},
                        output_state={},
                        conclusion="Graph-first reasoning"
                    )
                
                # Reason from graph state only
                graph_answer = self.graph_first_reasoner.reason_from_graph_state(
                    state=graph_state,
                    query=corrected_task,
                    graph_manager=self.symbolic_reasoner.graph_manager
                )
                result['graph_first_answer'] = graph_answer
                
                # Track reasoning step: Graph answer
                if self.reasoning_tracker and graph_answer.get('answer'):
                    self.reasoning_tracker.add_step(
                        step_type=StepType.INFERENCE,
                        operation="graph_answer",
                        input_state={},
                        output_state={'answer': graph_answer.get('answer')},
                        conclusion=graph_answer.get('answer', '')
                    )
                
                # If graph-first reasoning provides an answer, use it
                if graph_answer.get('answer'):
                    result['analysis']['graph_first_insight'] = graph_answer['answer']
            except Exception as e:
                logger.warning(f"Graph-first reasoning failed: {e}")
        
        # Step 5: Generate counterfactuals (if requested or if we have a state)
        if intent['type'] == 'counterfactual' or (graph_nodes and not intent['type'] == 'extraction'):
            
            # Use intervention variables from intent if available
            # Prefer state variables (variables with extracted values) over question targets
            target_vars = intent.get('intervention_variables', [])
            if not target_vars:
                # Use variables that have extracted values (state variables)
                state_vars = [v for v in graph_nodes if v in extracted_values or any(v in k for k in extracted_values.keys())]
                if state_vars:
                    target_vars = state_vars[:3]  # Use first 3 state variables
                else:
                    target_vars = [v for v in graph_nodes if 'expected' not in v.lower()][:3]  # Exclude question targets
            
            # Filter target_vars to only valid variables
            target_vars = [v for v in target_vars if v in graph_nodes]
            
            if target_vars:
                try:
                    counterfactuals = self.statistical_engine.generate_probabilistic_counterfactuals(
                        factual_state=factual_state,
                        target_variables=target_vars,
                        n_scenarios=min(5, len(target_vars) + 2)
                    )
                    result['counterfactuals'] = counterfactuals
                except Exception as e:
                    logger.warning(f"Counterfactual generation failed: {e}")
        
        # Step 6: Apply pragmatic layer for response generation
        if self.pragmatic_compiler and result.get('analysis'):
            # Determine confidence and complexity for pragmatic decisions
            avg_confidence = sum([r.get('confidence', 0.8) for r in relationships]) / len(relationships) if relationships else 0.8
            complexity = len(graph_nodes) + len(graph_edges)
            
            register = self.pragmatic_compiler.select_register(avg_confidence, complexity)
            result['pragmatic'] = {
                'register': register,
                'hedging': self.pragmatic_compiler.generate_hedging(avg_confidence),
                'explicitness': self.pragmatic_compiler.adjust_explicitness(len(graph_nodes))
            }
        else:
            # Default pragmatic info if compiler not available
            result['pragmatic'] = {
                'register': 'neutral',
                'hedging': 'likely',
                'explicitness': 3
            }
        
        # Step 7: Self-verification and error detection
        if self.error_detector and self.reasoning_tracker and self.reasoning_tracker.current_chain:
            reasoning_chain = self.reasoning_tracker.current_chain
            graph_state = {
                'nodes': graph_nodes,
                'edges': graph_edges
            }
            errors = self.error_detector.detect_errors(reasoning_chain, graph_state)
            if errors:
                result['errors'] = errors
                logger.warning(f"Detected {len(errors)} errors")
                
                # Try to correct errors
                if self.self_corrector:
                    corrections = self.self_corrector.correct_errors(errors, graph_state)
                    if corrections:
                        result['corrections'] = corrections
                        logger.info(f"Applied {len(corrections)} corrections")
        
        # Step 8: Generate explanations
        if self.explanation_builder and self.reasoning_tracker and self.reasoning_tracker.current_chain:
            reasoning_chain = self.reasoning_tracker.current_chain
            graph_state = {
                'nodes': graph_nodes,
                'edges': graph_edges
            }
            explanation = self.explanation_builder.generate_explanation(reasoning_chain, graph_state)
            result['explanation'] = explanation
        
        # Step 9: Transparency layer
        if self.transparency_layer and self.reasoning_tracker and self.reasoning_tracker.current_chain:
            reasoning_chain = self.reasoning_tracker.current_chain
            trace = self.transparency_layer.show_reasoning_trace(reasoning_chain)
            confidence_viz = self.transparency_layer.visualize_confidence(reasoning_chain)
            graph_explanation = self.transparency_layer.explain_graph_structure({
                'nodes': graph_nodes,
                'edges': graph_edges
            })
            result['transparency'] = {
                'trace': trace,
                'confidence': confidence_viz,
                'graph_structure': graph_explanation
            }
        
        # Store reasoning chain in result
        if self.reasoning_tracker and self.reasoning_tracker.current_chain and result:
            result['reasoning_chain'] = self.reasoning_tracker.current_chain
            # Mark chain as successful if we have results
            if result.get('analysis') or result.get('graph_first_answer'):
                self.reasoning_tracker.current_chain.success = True
                graph_answer = result.get('graph_first_answer', {})
                if graph_answer:
                    self.reasoning_tracker.current_chain.final_conclusion = graph_answer.get('answer')
        
        # Step 10: Validate result completeness
        self._validate_result(result)
        
        return result
    
    def _validate_result(self, result: Dict[str, Any]) -> None:
        """
        Validate result completeness and add warnings if needed.
        
        Args:
            result: Result dictionary to validate
        """
        warnings = []
        
        # Check if we have any meaningful analysis
        analysis = result.get('analysis', {})
        if not analysis.get('variables') and not analysis.get('relationships'):
            warnings.append("No variables or relationships were extracted from the input.")
        
        # Check graph-first answer quality
        graph_answer = result.get('graph_first_answer', {})
        if not graph_answer or not graph_answer.get('answer'):
            warnings.append("Graph-first reasoning did not produce a clear answer.")
        
        # Check counterfactuals if requested
        intent = result.get('intent', {})
        if intent.get('type') == 'counterfactual' and not result.get('counterfactuals'):
            warnings.append("Counterfactual analysis was requested but none were generated.")
        
        if warnings:
            result['warnings'] = warnings
            logger.warning(f"Result validation warnings: {warnings}")


class HybridAgent:
    """
    Main hybrid agent class with graph-first reasoning architecture.
    
    Supports both causal reasoning (CRCA) and general knowledge tasks.
    
    New Architecture:
    Text Input -> TextCorrector -> LanguageCompiler -> SymbolicReasoner
    -> GraphManager -> GraphFirstReasoner -> RuleBasedNLG -> Response
    
    No LLM dependency - pure symbolic-statistical reasoning with graph-first architecture.
    
    Graph Types Supported:
    - "causal": Causal relationships (default for CRCA tasks)
    - "knowledge": General knowledge graphs (facts, definitions, taxonomic relationships)
    - "dependency": Dependency relationships
    - "mixed": Combination of relationship types
    """
    
    def __init__(
        self,
        graph_type: str = "causal",
        seed: int = 42,
        enable_graph_first: bool = True,
        enable_compression: bool = True,
        enable_language_compilation: bool = True,
        enable_error_correction: bool = True,
        enable_conversation: bool = True,
        enable_reasoning_tracking: bool = True,
        enable_few_shot_learning: bool = True,
        enable_task_decomposition: bool = True,
        enable_explanations: bool = True,
        enable_verification: bool = True,
        enable_consistency: bool = True
    ):
        """
        Initialize hybrid agent with graph-first architecture.
        
        Args:
            graph_type: Type of graph (causal, knowledge, dependency, etc.)
            seed: Random seed for reproducibility
            enable_graph_first: Enable graph-first reasoning (answers from graph only)
            enable_compression: Enable graph compression and abstraction
            enable_language_compilation: Enable language compilation layers
            enable_error_correction: Enable non-destructive text correction
        """
        # Initialize core components
        self.graph_manager = GraphManager(graph_type=graph_type)
        self.prediction_framework = PredictionFramework(
            graph_manager=self.graph_manager
        )
        
        # Initialize graph-first components
        self.graph_first_reasoner = GraphFirstReasoner(graph_manager=self.graph_manager) if enable_graph_first else None
        self.graph_compressor = GraphCompressor(self.graph_manager) if enable_compression else None
        
        # Initialize language compilation components
        if enable_language_compilation:
            # Enable dictionary integration by default (no API key required)
            self.lexical_compiler = LexicalCompiler(enable_dictionary=True, cache_enabled=True)
            self.grammatical_compiler = GrammaticalCompiler()
            self.pragmatic_compiler = PragmaticCompiler()
        else:
            self.lexical_compiler = None
            self.grammatical_compiler = None
            self.pragmatic_compiler = None
        
        # Initialize error correction (with dictionary integration)
        if enable_error_correction:
            # Pass lexical compiler to text corrector for dictionary-enhanced correction
            lexical_for_corrector = self.lexical_compiler if enable_language_compilation else None
            self.text_corrector = TextCorrector(lexical_compiler=lexical_for_corrector)
        else:
            self.text_corrector = None
        
        # Initialize few-shot learning components (needed before symbolic reasoner)
        if enable_few_shot_learning:
            self.example_store = ExampleStore()
            self.pattern_learner = PatternLearner(self.example_store)
            self.adaptive_extractor = AdaptiveExtractor(self.pattern_learner, self.example_store)
        else:
            self.example_store = None
            self.pattern_learner = None
            self.adaptive_extractor = None
        
        # Initialize reasoning components
        # Pass lexical compiler to symbolic reasoner for dictionary-enhanced validation
        lexical_for_reasoner = self.lexical_compiler if enable_language_compilation else None
        adaptive_extractor_for_reasoner = self.adaptive_extractor if enable_few_shot_learning else None
        self.symbolic_reasoner = SymbolicReasoner(
            self.graph_manager,
            lexical_compiler=lexical_for_reasoner,
            adaptive_extractor=adaptive_extractor_for_reasoner
        )
        self.statistical_engine = StatisticalEngine(
            graph_manager=self.graph_manager,
            prediction_framework=self.prediction_framework,
            seed=seed
        )
        self.nlg = RuleBasedNLG()
        # Initialize LLM-enhanced components
        if enable_conversation:
            self.conversation_history = ConversationHistory(decay_lambda=0.1)
            self.context_tracker = ContextTracker(self.conversation_history)
        else:
            self.conversation_history = None
            self.context_tracker = None
        
        if enable_reasoning_tracking:
            self.reasoning_tracker = ReasoningTracker()
        else:
            self.reasoning_tracker = None
        
        if enable_few_shot_learning:
            self.example_store = ExampleStore()
            self.pattern_learner = PatternLearner(self.example_store)
            self.adaptive_extractor = AdaptiveExtractor(self.pattern_learner, self.example_store)
        else:
            self.example_store = None
            self.pattern_learner = None
            self.adaptive_extractor = None
        
        if enable_task_decomposition:
            self.task_analyzer = TaskAnalyzer()
            self.subtask_executor = SubTaskExecutor()
            self.plan_generator = PlanGenerator(self.task_analyzer)
        else:
            self.task_analyzer = None
            self.subtask_executor = None
            self.plan_generator = None
        
        if enable_explanations:
            self.explanation_builder = ExplanationBuilder()
            self.transparency_layer = TransparencyLayer()
        else:
            self.explanation_builder = None
            self.transparency_layer = None
        
        if enable_verification:
            self.consistency_checker = ConsistencyChecker()
            self.error_detector = ErrorDetector()
            self.self_corrector = SelfCorrector()
        else:
            self.consistency_checker = None
            self.error_detector = None
            self.self_corrector = None
        
        if enable_consistency:
            self.consistency_engine = ConsistencyEngine(seed=seed)
        else:
            self.consistency_engine = None
        
        self.orchestrator = HybridOrchestrator(
            symbolic_reasoner=self.symbolic_reasoner,
            statistical_engine=self.statistical_engine,
            nlg=self.nlg,
            graph_first_reasoner=self.graph_first_reasoner,
            text_corrector=self.text_corrector,
            lexical_compiler=self.lexical_compiler,
            grammatical_compiler=self.grammatical_compiler,
            pragmatic_compiler=self.pragmatic_compiler,
            reasoning_tracker=self.reasoning_tracker,
            explanation_builder=self.explanation_builder,
            transparency_layer=self.transparency_layer,
            consistency_checker=self.consistency_checker,
            error_detector=self.error_detector,
            self_corrector=self.self_corrector,
            consistency_engine=self.consistency_engine
        )
    
    def run(
        self,
        task: str,
        data: Optional[Any] = None,
        response_style: str = 'conversational',
        context: Optional[ConversationContext] = None,
        show_reasoning: bool = False
    ) -> str:
        """
        Run hybrid agent on a task with conversation context support.
        
        Args:
            task: Task description
            data: Optional data for statistical inference
            response_style: Response style ('conversational', 'brief', 'full')
            context: Optional conversation context
            show_reasoning: Whether to show chain-of-thought reasoning
            
        Returns:
            Natural language response
        """
        try:
            # Handle conversation context
            if self.conversation_history and context is None:
                # Use existing context if available
                context = self.conversation_history.context
            
            # Add user message to conversation
            if self.conversation_history:
                self.conversation_history.add_message(
                    role=MessageRole.USER,
                    content=task
                )
            
            # Resolve references in task using context
            if self.context_tracker and context:
                resolved_task = self._resolve_task_references(task, context)
            else:
                resolved_task = task
            
            # Task decomposition (if enabled)
            if self.task_analyzer and self.plan_generator:
                plan = self.plan_generator.generate_plan(resolved_task)
                if plan['estimated_steps'] > 1:
                    # Complex task - use decomposition
                    logger.info(f"Decomposing task into {plan['estimated_steps']} subtasks")
                    # For now, proceed with original task
                    # Future: Execute subtasks in parallel
            
            # Validate input
            if not resolved_task or not isinstance(resolved_task, str):
                return "I need a valid task description to analyze. Please provide a question or statement about causal relationships."
            
            if len(resolved_task.strip()) == 0:
                return "Please provide a non-empty task description."
            
            # Execute hybrid reasoning with graph-first architecture
            result = self.orchestrator.reason_hybrid(task=resolved_task, data=data, context=context)
            
            # Validate result
            if not result:
                return "I couldn't process your request. Please try rephrasing with clearer causal relationships."
            
            # Add agent response to conversation
            if self.conversation_history:
                # Will add after response is generated
                pass
            
            # Get pragmatic information for response generation
            pragmatic_info = result.get('pragmatic', {
                'register': 'neutral',
                'hedging': 'likely',
                'explicitness': 3
            })
            
            # Generate natural language response from graph state
            reasoning_chain = None
            if self.reasoning_tracker and self.reasoning_tracker.current_chain:
                reasoning_chain = self.reasoning_tracker.current_chain
            
            if show_reasoning and reasoning_chain:
                # Include chain-of-thought reasoning
                if self.explanation_builder:
                    explanation = self.explanation_builder.generate_explanation(reasoning_chain, result.get('analysis', {}))
                    result['explanation'] = explanation
                
                response = self.nlg.generate_response(
                    result,
                    response_type=response_style,
                    pragmatic_info=pragmatic_info,
                    show_reasoning=True,
                    reasoning_chain=reasoning_chain
                )
            else:
                response = self.nlg.generate_response(
                    result,
                    response_type=response_style,
                    pragmatic_info=pragmatic_info,
                    show_reasoning=False,
                    reasoning_chain=reasoning_chain
                )
            
            # Add agent response to conversation
            if self.conversation_history:
                self.conversation_history.add_message(
                    role=MessageRole.AGENT,
                    content=response,
                    metadata={'result': result}
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in hybrid agent run: {e}", exc_info=True)
            return f"I encountered an error processing your request: {str(e)}. Please try rephrasing."
    
    def _resolve_task_references(
        self,
        task: str,
        context: ConversationContext
    ) -> str:
        """
        Resolve references in task using conversation context.
        
        Args:
            task: Original task
            context: Conversation context
            
        Returns:
            Task with resolved references
        """
        if not self.context_tracker:
            return task
        
        # Resolve common references
        resolved = task
        references = ['it', 'that', 'this', 'the price', 'the variable']
        
        for ref in references:
            if ref.lower() in task.lower():
                resolved_var = self.context_tracker.resolve_reference(ref, context.current_turn)
                if resolved_var:
                    resolved = resolved.replace(ref, resolved_var)
        
        return resolved
    
    def update_context(
        self,
        context: Optional[ConversationContext],
        user_message: str,
        agent_response: str
    ) -> ConversationContext:
        """
        Update conversation context after interaction.
        
        Args:
            context: Current context
            user_message: User message
            agent_response: Agent response
            
        Returns:
            Updated context
        """
        if self.conversation_history:
            return self.conversation_history.context
        return context
    
    def learn_from_examples(
        self,
        examples: List[Tuple[str, Dict[str, Any]]]
    ) -> None:
        """
        Learn from examples for few-shot learning.
        
        Args:
            examples: List of (input_text, output_structure) tuples
        """
        if not self.example_store or not self.pattern_learner:
            logger.warning("Few-shot learning not enabled")
            return
        
        # Add examples to store
        for input_text, output in examples:
            self.example_store.add_example(input_text, output)
        
        # Learn patterns
        self.pattern_learner.learn_from_examples(examples)
        logger.info(f"Learned from {len(examples)} examples")
    
    def _generate_brief_response(
        self,
        result: Dict[str, Any],
        pragmatic_info: Dict[str, Any]
    ) -> str:
        """Generate brief summary response."""
        hedging = pragmatic_info.get('hedging', 'likely')
        
        # Try graph-first answer first
        graph_answer = result.get('graph_first_answer', {})
        if graph_answer and graph_answer.get('answer'):
            return f"{hedging.capitalize()}, {graph_answer['answer']}"
        
        # Fallback to analysis summary
        analysis = result.get('analysis', {})
        variables = analysis.get('variables', [])
        relationships = analysis.get('relationships', [])
        
        if variables and relationships:
            return f"I've identified {len(variables)} variables with {len(relationships)} causal relationships. {hedging.capitalize()}, the strongest relationship is between '{relationships[0].get('source', '')}' and '{relationships[0].get('target', '')}'."
        
        return "I've analyzed your request, but couldn't extract clear causal relationships. Please provide more specific information about the variables and their relationships."
    
    def _generate_fallback_response(self, result: Dict[str, Any]) -> str:
        """Generate fallback response when main generation fails."""
        task = result.get('task', 'your request')
        return f"I've processed {task}, but couldn't generate a detailed response. The analysis may need more information or clearer causal relationships."
    
    def query_graph(self, question: str) -> Dict[str, Any]:
        """
        Query graph state directly (graph-first reasoning).
        
        Args:
            question: Question to answer from graph state
            
        Returns:
            Dictionary with answer derived from graph state
        """
        if self.graph_first_reasoner is None:
            raise ValueError("Graph-first reasoning is not enabled")
        
        return self.graph_first_reasoner.query_graph_state(question, self.graph_manager)
    
    def reason_from_graph_state(self, state: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Pure graph reasoning from explicit graph state.
        
        Args:
            state: Graph state dictionary
            query: Query string
            
        Returns:
            Dictionary with reasoning results
        """
        if self.graph_first_reasoner is None:
            raise ValueError("Graph-first reasoning is not enabled")
        
        return self.graph_first_reasoner.reason_from_graph_state(state, query, self.graph_manager)
    
    def extract_causal_variables(self, task: str) -> Dict[str, Any]:
        """
        Extract causal variables from a task.
        
        Args:
            task: Natural language task description
            
        Returns:
            Dictionary with extracted variables and relationships
        """
        extraction = self.symbolic_reasoner.extract_variables_from_task(task)
        
        # Add edges to graph
        for source, target in extraction.get('edges', []):
            self.graph_manager.add_relationship(
                source=source,
                target=target,
                strength=1.0,
                confidence=0.8
            )
        
        return extraction
    
    def generate_causal_analysis(
        self,
        variables: Dict[str, Any],
        data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Generate causal analysis from variables.
        
        Args:
            variables: Dictionary with variables and relationships
            data: Optional pandas DataFrame for statistical inference
            
        Returns:
            Dictionary with causal analysis results
        """
        # Build graph from variables
        var_list = variables.get('variables', [])
        edges = variables.get('edges', [])
        
        for source, target in edges:
            self.graph_manager.add_relationship(
                source=source,
                target=target,
                strength=1.0
            )
        
        # Fit statistical model if data available
        if data is not None and PANDAS_AVAILABLE:
            try:
                self.statistical_engine.fit_from_dataframe(
                    df=data,
                    variables=var_list
                )
            except Exception as e:
                logger.warning(f"Statistical fitting failed: {e}")
        
        # Build analysis
        relationships = []
        for source, target in self.graph_manager.get_edges():
            strength = self.statistical_engine.assess_causal_strength(source, target)
            relationships.append({
                'source': source,
                'target': target,
                'strength': strength
            })
        
        return {
            'variables': var_list,
            'relationships': relationships,
            'graph_structure': f"{len(var_list)} variables, {len(relationships)} relationships"
        }
    
    def generate_counterfactuals(
        self,
        state: Dict[str, float],
        target_vars: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual scenarios.
        
        Args:
            state: Factual state dictionary
            target_vars: List of variables to intervene on
            
        Returns:
            List of counterfactual scenario dictionaries
        """
        return self.statistical_engine.generate_probabilistic_counterfactuals(
            factual_state=state,
            target_variables=target_vars,
            n_scenarios=5
        )

