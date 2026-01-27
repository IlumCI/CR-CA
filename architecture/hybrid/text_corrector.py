"""
Non-destructive text correction pipeline for CRCA hybrid agent.

Provides three-layer correction:
1. Orthographic normalization: Spelling errors
2. Informal compression handling: Abbreviations, shortcuts
3. Grammar recovery: Recover structure, not polish

All corrections are non-destructive - original form is preserved
with confidence scores and provenance.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
import re

from schemas.hybrid import AnnotatedToken, DependencyTree
from utils.edit_distance import find_closest_match, damerau_levenshtein_distance

logger = logging.getLogger(__name__)


class TextCorrector:
    """
    Non-destructive text corrector with three correction layers.
    
    Correction produces annotated tokens with:
    - original_form: What user typed
    - normalized_form: Corrected version
    - confidence: Correction confidence
    - correction_type: Type of correction
    - provenance: Why correction was made
    - metadata: Optional dictionary information
    """
    
    def __init__(
        self,
        vocabulary: Optional[List[str]] = None,
        lexical_compiler: Optional[Any] = None
    ):
        """
        Initialize text corrector.
        
        Args:
            vocabulary: Optional vocabulary list for spelling correction
            lexical_compiler: Optional LexicalCompiler instance for dictionary lookups
        """
        self.vocabulary = set(vocabulary) if vocabulary else set()
        self.lexical_compiler = lexical_compiler
        self.abbreviations = self._build_abbreviation_dict()
        self.common_words = self._build_common_words()
    
    def _build_abbreviation_dict(self) -> Dict[str, str]:
        """Build dictionary of common abbreviations."""
        return {
            "depnds": "depends",
            "demnad": "demand",
            "prce": "price",
            "qulity": "quality",
            "affcts": "affects",
            "influnces": "influences",
            "causes": "causes",  # Already correct, but included for completeness
        }
    
    def _build_common_words(self) -> set:
        """Build set of common English words."""
        return {
            "the", "a", "an", "and", "or", "but", "if", "then", "when",
            "what", "which", "how", "why", "where", "who",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might",
            "cause", "causes", "effect", "effects", "affect", "affects",
            "influence", "influences", "depend", "depends", "lead", "leads",
            "result", "results", "impact", "impacts", "determine", "determines"
        }
    
    def correct_orthographic(self, text: str) -> List[AnnotatedToken]:
        """
        Correct orthographic (spelling) errors.
        
        Args:
            text: Text to correct
            
        Returns:
            List of AnnotatedToken objects
        """
        tokens = text.split()
        corrected_tokens = []
        
        for token in tokens:
            # Clean token (remove punctuation for matching)
            clean_token = re.sub(r'[^\w]', '', token.lower())
            
            # Check if token is already correct
            if clean_token in self.common_words or clean_token in self.vocabulary:
                corrected_tokens.append(AnnotatedToken(
                    original_form=token,
                    normalized_form=token,
                    confidence=1.0,
                    correction_type="none"
                ))
                continue
            
            # Try to find closest match
            candidates = list(self.common_words) + list(self.vocabulary)
            match, distance = find_closest_match(clean_token, candidates, max_distance=2)
            
            # If dictionary is available, also check if word is valid
            word_info = None
            if self.lexical_compiler and self.lexical_compiler.enable_dictionary:
                word_info = self.lexical_compiler.get_word_info(clean_token)
                if word_info and word_info.get('found'):
                    # Word is valid, no correction needed
                    match = clean_token
                    distance = 0
            
            if match and distance <= 2:
                # Calculate confidence based on distance
                confidence = max(0.0, 1.0 - (distance / 3.0))
                
                # Preserve original capitalization if possible
                normalized = match
                if token[0].isupper():
                    normalized = match.capitalize()
                
                # Store dictionary info in metadata if available
                metadata = None
                if word_info:
                    metadata = word_info
                
                corrected_tokens.append(AnnotatedToken(
                    original_form=token,
                    normalized_form=normalized,
                    confidence=confidence,
                    correction_type="spelling" if distance > 0 else "none",
                    provenance=f"Edit distance: {distance}" if distance > 0 else "Valid word (dictionary verified)",
                    metadata=metadata
                ))
            else:
                # No correction found, but check dictionary to see if it's a valid word
                if self.lexical_compiler and self.lexical_compiler.enable_dictionary:
                    word_info = self.lexical_compiler.get_word_info(clean_token)
                    if word_info and word_info.get('found'):
                        # Word is valid, just not in our local vocabulary
                        metadata = word_info
                        corrected_tokens.append(AnnotatedToken(
                            original_form=token,
                            normalized_form=token,
                            confidence=0.8,  # Medium confidence - valid word but not in local vocab
                            correction_type="none",
                            provenance="Valid word (dictionary verified)",
                            metadata=metadata
                        ))
                    else:
                        # Word not found in dictionary either
                        corrected_tokens.append(AnnotatedToken(
                            original_form=token,
                            normalized_form=token,
                            confidence=0.3,  # Low confidence - unknown word
                            correction_type="none",
                            provenance="Unknown word, not found in dictionary"
                        ))
                else:
                    # No dictionary available
                    corrected_tokens.append(AnnotatedToken(
                        original_form=token,
                        normalized_form=token,
                        confidence=0.5,  # Low confidence - unknown word
                        correction_type="none",
                        provenance="Unknown word, no correction found"
                    ))
        
        return corrected_tokens
    
    def expand_abbreviations(self, tokens: List[AnnotatedToken]) -> List[AnnotatedToken]:
        """
        Expand abbreviations and shortcuts.
        
        Args:
            tokens: List of AnnotatedToken objects
            
        Returns:
            List of AnnotatedToken objects with abbreviations expanded
        """
        expanded_tokens = []
        
        for token in tokens:
            original = token.original_form.lower()
            
            # Check abbreviation dictionary
            if original in self.abbreviations:
                expanded = self.abbreviations[original]
                
                # Preserve original capitalization
                if token.original_form[0].isupper():
                    expanded = expanded.capitalize()
                
                expanded_tokens.append(AnnotatedToken(
                    original_form=token.original_form,
                    normalized_form=expanded,
                    confidence=0.9,
                    correction_type="abbreviation",
                    provenance=f"Abbreviation expansion: {original} -> {expanded}"
                ))
            else:
                # No abbreviation found, keep token as-is
                expanded_tokens.append(token)
        
        return expanded_tokens
    
    def recover_grammar(self, tokens: List[AnnotatedToken]) -> DependencyTree:
        """
        Recover grammar structure from tokens (not polish, just structure).
        
        Args:
            tokens: List of AnnotatedToken objects
            
        Returns:
            DependencyTree representing recovered structure
        """
        words = [token.normalized_form for token in tokens]
        
        # Simple dependency recovery
        # In full implementation, would use more sophisticated parsing
        nodes = words
        edges = []
        root = None
        
        if len(words) >= 2:
            # Look for verb
            verbs = ["is", "are", "was", "were", "causes", "affects", "influences", "depends"]
            verb_idx = None
            for i, word in enumerate(words):
                if word.lower() in verbs:
                    verb_idx = i
                    break
            
            if verb_idx is not None:
                root = words[verb_idx]
                # Subject is before verb
                if verb_idx > 0:
                    edges.append((words[verb_idx - 1], words[verb_idx], "nsubj"))
                # Object is after verb
                if verb_idx < len(words) - 1:
                    edges.append((words[verb_idx], words[verb_idx + 1], "dobj"))
        
        if root is None and words:
            root = words[0]
        
        return DependencyTree(nodes=nodes, edges=edges, root=root)
    
    def correct_text(
        self,
        text: str,
        use_abbreviation_expansion: bool = True,
        use_grammar_recovery: bool = True
    ) -> Dict[str, Any]:
        """
        Complete correction pipeline.
        
        Args:
            text: Text to correct
            use_abbreviation_expansion: Whether to expand abbreviations
            use_grammar_recovery: Whether to recover grammar structure
            
        Returns:
            Dictionary with corrected tokens and dependency tree
        """
        # Step 1: Orthographic correction
        tokens = self.correct_orthographic(text)
        
        # Step 2: Abbreviation expansion
        if use_abbreviation_expansion:
            tokens = self.expand_abbreviations(tokens)
        
        # Step 3: Grammar recovery
        dependency_tree = None
        if use_grammar_recovery:
            dependency_tree = self.recover_grammar(tokens)
        
        return {
            "original_text": text,
            "corrected_tokens": tokens,
            "corrected_text": " ".join([t.normalized_form for t in tokens]),
            "dependency_tree": dependency_tree,
            "confidence": min([t.confidence for t in tokens]) if tokens else 1.0
        }
    
    def disambiguate_with_graph(
        self,
        tokens: List[AnnotatedToken],
        graph_manager: Any,  # GraphManager instance
        expected_pattern: Optional[str] = None
    ) -> List[AnnotatedToken]:
        """
        Use graph structure for context-aware disambiguation.
        
        Args:
            tokens: List of AnnotatedToken objects
            graph_manager: GraphManager instance for graph structure
            expected_pattern: Optional expected pattern (e.g., "depends on")
            
        Returns:
            List of disambiguated AnnotatedToken objects
        """
        disambiguated = []
        
        # Get variables from graph
        graph_variables = set(graph_manager.get_nodes())
        
        for token in tokens:
            normalized = token.normalized_form.lower()
            
            # If token matches a graph variable, increase confidence
            if normalized in {v.lower() for v in graph_variables}:
                disambiguated.append(AnnotatedToken(
                    original_form=token.original_form,
                    normalized_form=token.normalized_form,
                    confidence=min(1.0, token.confidence + 0.2),  # Boost confidence
                    correction_type=token.correction_type,
                    provenance=f"{token.provenance}; Graph variable match"
                ))
            else:
                # Try to find closest graph variable
                if graph_variables:
                    match, distance = find_closest_match(normalized, list(graph_variables), max_distance=2)
                    if match and distance <= 2:
                        disambiguated.append(AnnotatedToken(
                            original_form=token.original_form,
                            normalized_form=match,
                            confidence=0.7,
                            correction_type="graph_disambiguation",
                            provenance=f"Matched to graph variable '{match}' (distance: {distance})"
                        ))
                    else:
                        disambiguated.append(token)
                else:
                    disambiguated.append(token)
        
        return disambiguated
