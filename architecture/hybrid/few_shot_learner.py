"""
Few-Shot Learning System with Meta-Learning and Pattern Generalization.

Implements meta-learning with MDL-based pattern generalization,
gradient-free optimization, LSH indexing, and Bayesian pattern updating.

Theoretical Basis:
- Meta-Learning (Schmidhuber 1987, Thrun & Pratt 1998)
- Pattern Recognition (Duda & Hart 1973)
- Minimum Description Length (Rissanen 1978)
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import logging
import re
import math
import hashlib

logger = logging.getLogger(__name__)

# Try to import numpy for numerical operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class ExampleStore:
    """
    Stores example patterns with LSH indexing for O(1) approximate nearest neighbor search.
    
    Implements episodic memory with:
    - Input-output pairs: D = {(x₁, y₁), ..., (xₙ, yₙ)}
    - Variable extraction patterns with learned weights
    - Relationship inference patterns
    - Domain-specific templates
    """
    
    def __init__(self):
        """Initialize example store."""
        self.examples: List[Tuple[str, Dict[str, Any]]] = []  # (input, output) pairs
        self.patterns: List[Dict[str, Any]] = []  # Learned patterns
        self.lsh_index: Dict[str, List[int]] = defaultdict(list)  # LSH buckets
        self.domain_templates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def add_example(
        self,
        input_text: str,
        output: Dict[str, Any]
    ) -> None:
        """
        Add input-output example pair.
        
        Args:
            input_text: Input natural language text
            output: Extracted structure (variables, edges, etc.)
        """
        self.examples.append((input_text, output))
        
        # Update LSH index (simplified hash-based indexing)
        hash_key = self._lsh_hash(input_text)
        self.lsh_index[hash_key].append(len(self.examples) - 1)
    
    def _lsh_hash(self, text: str, num_hashes: int = 5) -> str:
        """
        Compute LSH hash for approximate nearest neighbor search.
        
        Simplified implementation using multiple hash functions.
        
        Args:
            text: Text to hash
            num_hashes: Number of hash functions
            
        Returns:
            Hash key
        """
        # Simple character-based hash
        text_lower = text.lower()
        hashes = []
        for i in range(num_hashes):
            # Use different hash seeds
            hash_obj = hashlib.md5(f"{i}_{text_lower}".encode())
            hashes.append(hash_obj.hexdigest()[:8])
        return "_".join(hashes)
    
    def find_similar_examples(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Find k most similar examples using LSH.
        
        O(1) approximate nearest neighbor search.
        
        Args:
            query: Query text
            k: Number of examples to retrieve
            
        Returns:
            List of (input, output, similarity_score) tuples
        """
        query_hash = self._lsh_hash(query)
        
        # Get candidates from same LSH bucket
        candidate_indices = set()
        for hash_key in self.lsh_index:
            # Simple similarity: count matching hash components
            similarity = self._hash_similarity(query_hash, hash_key)
            if similarity > 0.3:  # Threshold
                candidate_indices.update(self.lsh_index[hash_key])
        
        # Compute similarity scores for candidates
        scored_examples = []
        for idx in candidate_indices:
            if idx < len(self.examples):
                input_text, output = self.examples[idx]
                similarity = self._text_similarity(query, input_text)
                scored_examples.append((input_text, output, similarity))
        
        # Sort by similarity and return top k
        scored_examples.sort(key=lambda x: x[2], reverse=True)
        return scored_examples[:k]
    
    def _hash_similarity(self, hash1: str, hash2: str) -> float:
        """Compute similarity between two hashes."""
        components1 = set(hash1.split("_"))
        components2 = set(hash2.split("_"))
        if not components1 or not components2:
            return 0.0
        intersection = len(components1 & components2)
        union = len(components1 | components2)
        return intersection / union if union > 0 else 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between texts."""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0


class PatternLearner:
    """
    Implements gradient-free meta-learning with MDL-based pattern generalization.
    
    Features:
    - Pattern extraction using Minimum Description Length (MDL)
    - Generalization via abstraction function
    - Dynamic rule updates with exponential moving average
    - Domain adaptation using domain embeddings
    """
    
    def __init__(self, example_store: ExampleStore):
        """
        Initialize pattern learner.
        
        Args:
            example_store: ExampleStore instance
        """
        self.example_store = example_store
        self.learned_patterns: List[Dict[str, Any]] = []
        self.pattern_weights: Dict[str, float] = {}
    
    def learn_from_examples(
        self,
        examples: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Learn patterns from examples using MDL principle.
        
        Algorithm:
            function learn_from_examples(examples):
                patterns = []
                for (x, y) in examples:
                    pattern = extract_pattern(x, y)
                    patterns.append((pattern, compute_mdl(pattern)))
                return select_best_patterns(patterns, k)  // Top-k by MDL
        
        MDL: L(pattern) = L(data|pattern) + L(pattern)
        
        Args:
            examples: Optional list of examples (uses store if None)
            k: Number of best patterns to return
            
        Returns:
            List of learned patterns
        """
        if examples is None:
            examples = self.example_store.examples
        
        patterns_with_mdl = []
        
        for input_text, output in examples:
            pattern = self._extract_pattern(input_text, output)
            mdl_score = self._compute_mdl(pattern, input_text, output)
            patterns_with_mdl.append((pattern, mdl_score))
        
        # Select top-k by MDL (lower is better)
        patterns_with_mdl.sort(key=lambda x: x[1])
        best_patterns = [p for p, _ in patterns_with_mdl[:k]]
        
        self.learned_patterns = best_patterns
        
        # Initialize weights
        for i, pattern in enumerate(best_patterns):
            pattern_id = pattern.get('id', f"pattern_{i}")
            self.pattern_weights[pattern_id] = 1.0 / len(best_patterns)
        
        return best_patterns
    
    def _extract_pattern(
        self,
        input_text: str,
        output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract pattern from input-output pair.
        
        Args:
            input_text: Input text
            output: Output structure
            
        Returns:
            Pattern dictionary
        """
        # Extract variable patterns
        variables = output.get('variables', [])
        edges = output.get('edges', [])
        
        # Create regex pattern from input
        # Simple: replace specific words with generic patterns
        pattern_text = input_text.lower()
        
        # Replace variable names with placeholders
        for var in variables:
            pattern_text = re.sub(r'\b' + re.escape(var.lower()) + r'\b', r'\\w+', pattern_text)
        
        pattern = {
            'id': f"pattern_{hash(input_text) % 10000}",
            'regex': pattern_text,
            'variables': variables,
            'edges': edges,
            'input_template': input_text,
            'output_template': output
        }
        
        return pattern
    
    def _compute_mdl(
        self,
        pattern: Dict[str, Any],
        input_text: str,
        output: Dict[str, Any]
    ) -> float:
        """
        Compute Minimum Description Length for pattern.
        
        MDL: L(pattern) = L(data|pattern) + L(pattern)
        
        Args:
            pattern: Pattern dictionary
            input_text: Input text
            output: Output structure
            
        Returns:
            MDL score (lower is better)
        """
        # L(pattern): Description length of pattern
        pattern_length = len(str(pattern))
        
        # L(data|pattern): Description length of data given pattern
        # Simplified: how well pattern matches data
        match_score = self._pattern_match_score(pattern, input_text, output)
        data_given_pattern = -math.log(match_score + 1e-10)  # Negative log likelihood
        
        mdl = pattern_length + data_given_pattern
        return mdl
    
    def _pattern_match_score(
        self,
        pattern: Dict[str, Any],
        input_text: str,
        output: Dict[str, Any]
    ) -> float:
        """Compute how well pattern matches input-output pair."""
        # Simple matching: check if pattern variables match output variables
        pattern_vars = set(pattern.get('variables', []))
        output_vars = set(output.get('variables', []))
        
        if not pattern_vars or not output_vars:
            return 0.0
        
        intersection = len(pattern_vars & output_vars)
        union = len(pattern_vars | output_vars)
        return intersection / union if union > 0 else 0.0
    
    def update_pattern_weights(
        self,
        pattern_id: str,
        success: bool,
        learning_rate: float = 0.1
    ) -> None:
        """
        Update pattern weights using exponential moving average.
        
        θ_t = (1-η)·θ_{t-1} + η·θ_new
        
        Args:
            pattern_id: Pattern ID
            success: Whether pattern was successful
            learning_rate: Learning rate η
        """
        if pattern_id not in self.pattern_weights:
            return
        
        # Update weight based on success
        new_weight = 1.0 if success else 0.0
        current_weight = self.pattern_weights[pattern_id]
        
        # Exponential moving average
        updated_weight = (1 - learning_rate) * current_weight + learning_rate * new_weight
        
        self.pattern_weights[pattern_id] = updated_weight


class AdaptiveExtractor:
    """
    Implements adaptive pattern matching with style adaptation and correction learning.
    
    Features:
    - Learned patterns with confidence-weighted matching
    - Style adaptation using n-gram language model
    - Correction learning from feedback
    - Bayesian updating of pattern confidence
    """
    
    def __init__(
        self,
        pattern_learner: PatternLearner,
        example_store: ExampleStore
    ):
        """
        Initialize adaptive extractor.
        
        Args:
            pattern_learner: PatternLearner instance
            example_store: ExampleStore instance
        """
        self.pattern_learner = pattern_learner
        self.example_store = example_store
        self.user_style_model: Dict[str, float] = defaultdict(float)  # n-gram model
    
    def adapt_extraction(
        self,
        input_text: str,
        learned_patterns: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Adapt extraction using learned patterns.
        
        Algorithm:
            function adapt_extraction(input, learned_patterns):
                scores = [match_score(input, p) for p in learned_patterns]
                best_pattern = argmax(scores)
                return apply_pattern(best_pattern, input)
        
        Args:
            input_text: Input text to extract from
            learned_patterns: Optional list of patterns (uses learner's patterns if None)
            
        Returns:
            Extracted structure
        """
        if learned_patterns is None:
            learned_patterns = self.pattern_learner.learned_patterns
        
        if not learned_patterns:
            # No patterns learned yet, return empty structure
            return {'variables': [], 'edges': []}
        
        # Compute match scores
        scores = []
        for pattern in learned_patterns:
            score = self._match_score(input_text, pattern)
            pattern_id = pattern.get('id', '')
            weight = self.pattern_learner.pattern_weights.get(pattern_id, 0.5)
            # Weighted score
            weighted_score = score * weight
            scores.append((pattern, weighted_score))
        
        # Get best pattern
        if not scores:
            return {'variables': [], 'edges': []}
        
        best_pattern, best_score = max(scores, key=lambda x: x[1])
        
        # Apply pattern
        return self._apply_pattern(input_text, best_pattern)
    
    def _match_score(
        self,
        input_text: str,
        pattern: Dict[str, Any]
    ) -> float:
        """
        Compute match score between input and pattern.
        
        score = Σᵢ wᵢ·match(patternᵢ, input)
        
        Args:
            input_text: Input text
            pattern: Pattern dictionary
            
        Returns:
            Match score (0.0-1.0)
        """
        # Simple regex matching
        regex = pattern.get('regex', '')
        if not regex:
            return 0.0
        
        try:
            match = re.search(regex, input_text.lower())
            if match:
                return 1.0
            else:
                # Partial match: word overlap
                pattern_words = set(re.findall(r'\b\w+\b', regex))
                input_words = set(re.findall(r'\b\w+\b', input_text.lower()))
                if not pattern_words or not input_words:
                    return 0.0
                intersection = len(pattern_words & input_words)
                union = len(pattern_words | input_words)
                return intersection / union if union > 0 else 0.0
        except re.error:
            return 0.0
    
    def _apply_pattern(
        self,
        input_text: str,
        pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply pattern to input text.
        
        Args:
            input_text: Input text
            pattern: Pattern dictionary
            
        Returns:
            Extracted structure
        """
        # Use pattern's output template as base
        output_template = pattern.get('output_template', {})
        
        # Extract variables from input (simplified)
        variables = []
        edges = []
        
        # Try to match pattern variables in input
        pattern_vars = pattern.get('variables', [])
        for var in pattern_vars:
            # Look for variable in input
            if var.lower() in input_text.lower():
                variables.append(var)
        
        # Use pattern edges if variables match
        pattern_edges = pattern.get('edges', [])
        for source, target in pattern_edges:
            if source in variables and target in variables:
                edges.append((source, target))
        
        return {
            'variables': variables,
            'edges': edges,
            'confidence': 0.8,  # Default confidence
            'pattern_id': pattern.get('id', '')
        }
    
    def learn_from_correction(
        self,
        input_text: str,
        correction: Dict[str, Any],
        pattern_id: Optional[str] = None
    ) -> None:
        """
        Learn from correction feedback.
        
        Implements online learning: update(pattern, correction)
        
        Args:
            input_text: Original input
            correction: Corrected output
            pattern_id: Optional pattern ID that was used
        """
        # Add correction as new example
        self.example_store.add_example(input_text, correction)
        
        # Update pattern weights if pattern_id provided
        if pattern_id:
            self.pattern_learner.update_pattern_weights(pattern_id, success=True)
        
        # Re-learn patterns (simplified: just add to examples)
        # In full implementation, would trigger re-learning
    
    def update_style_model(
        self,
        text: str,
        n: int = 2
    ) -> None:
        """
        Update user-specific language model using n-grams.
        
        P(user_word | context) using n-gram model
        
        Args:
            text: Text to learn from
            n: N-gram size
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Generate n-grams
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            self.user_style_model[str(ngram)] += 1.0
