"""
Language compilation system for CRCA hybrid agent.

Provides three-layer language compilation:
1. Lexical layer: Words and phrases (synonyms, hypernyms, vocabulary)
2. Grammatical layer: Sentence structure (dependency grammar, causal patterns)
3. Pragmatic layer: Tone and style (confidence-based language decisions)

All language knowledge is compiled into queryable structures at initialization,
not parsed at runtime.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import logging
import re
import time
from collections import defaultdict

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from schemas.hybrid import LexicalGraph, SynonymSet, DependencyTree, CausalStructure

logger = logging.getLogger(__name__)


class LexicalCompiler:
    """
    Compiles lexical knowledge (words and phrases) into queryable structures.
    
    Features:
    - Synonym sets and hypernym chains
    - Controlled vocabulary expansion
    - Term normalization (canonical forms)
    - Vocabulary validation
    """
    
    def __init__(self, enable_dictionary: bool = True, cache_enabled: bool = True):
        """
        Initialize lexical compiler with dictionary integration.
        
        Args:
            enable_dictionary: Enable online dictionary lookups
            cache_enabled: Enable caching of dictionary lookups
        """
        self.lexical_graph = LexicalGraph()
        self.enable_dictionary = enable_dictionary and REQUESTS_AVAILABLE
        self.cache_enabled = cache_enabled
        
        # Dictionary cache to avoid repeated API calls
        self._dictionary_cache: Dict[str, Dict[str, Any]] = {}
        self._dictionary_cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 86400  # 24 hours
        
        # Dictionary API endpoint (Free Dictionary API - no key required)
        self.dictionary_api_url = "https://api.dictionaryapi.dev/api/v2/entries/en"
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 100ms between requests
        
        self._build_basic_vocabulary()
    
    def _build_basic_vocabulary(self) -> None:
        """Build basic vocabulary from common causal terms."""
        # Basic causal vocabulary
        causal_terms = {
            "cause": {"cause", "causes", "caused", "causing", "causation"},
            "effect": {"effect", "effects", "affected", "affecting", "affects"},
            "influence": {"influence", "influences", "influenced", "influencing"},
            "determine": {"determine", "determines", "determined", "determining"},
            "depend": {"depend", "depends", "depended", "depending", "dependent"},
            "lead": {"lead", "leads", "led", "leading"},
            "result": {"result", "results", "resulted", "resulting"},
            "impact": {"impact", "impacts", "impacted", "impacting"},
            "drive": {"drive", "drives", "drove", "driving", "driven"},
            "control": {"control", "controls", "controlled", "controlling"}
        }
        
        for canonical, synonyms in causal_terms.items():
            self.add_synonym_set(canonical, synonyms)
    
    def compile_lexicon(self, sources: List[str]) -> LexicalGraph:
        """
        Compile lexicon from multiple sources.
        
        Args:
            sources: List of source identifiers (for future expansion)
            
        Returns:
            Compiled LexicalGraph
        """
        # For now, use built-in vocabulary
        # In future, can load from dictionaries, WordNet, etc.
        return self.lexical_graph
    
    def add_synonym_set(self, canonical: str, synonyms: Set[str]) -> None:
        """
        Add a set of synonyms.
        
        Args:
            canonical: Canonical form of the term
            synonyms: Set of synonymous terms
        """
        self.lexical_graph.synonym_sets[canonical] = synonyms.copy()
        self.lexical_graph.vocabulary.add(canonical)
        self.lexical_graph.vocabulary.update(synonyms)
    
    def expand_vocabulary(self, word: str) -> Set[str]:
        """
        Expand vocabulary for a word (get synonyms and related terms).
        
        Args:
            word: Word to expand
            
        Returns:
            Set of related terms (including the word itself)
        """
        expanded = {word}
        word_lower = word.lower()
        
        # Find canonical form
        for canonical, synonyms in self.lexical_graph.synonym_sets.items():
            if word_lower == canonical.lower() or word_lower in {s.lower() for s in synonyms}:
                expanded.add(canonical)
                expanded.update(synonyms)
        
        return expanded
    
    def normalize_term(self, term: str) -> str:
        """
        Normalize a term to its canonical form.
        
        Args:
            term: Term to normalize
            
        Returns:
            Canonical form of the term
        """
        term_lower = term.lower()
        
        # Check if term is already canonical
        if term_lower in {c.lower() for c in self.lexical_graph.synonym_sets.keys()}:
            return term_lower
        
        # Find canonical form in synonym sets
        for canonical, synonyms in self.lexical_graph.synonym_sets.items():
            if term_lower == canonical.lower():
                return canonical
            if term_lower in {s.lower() for s in synonyms}:
                return canonical
        
        # If not found, return original (lowercased)
        return term_lower
    
    def validate_vocabulary(self, term: str) -> bool:
        """
        Validate if a term is in the vocabulary.
        
        Uses both local vocabulary and online dictionary if enabled.
        
        Args:
            term: Term to validate
            
        Returns:
            True if term is in vocabulary, False otherwise
        """
        term_lower = term.lower()
        
        # Check local vocabulary first
        if term_lower in {t.lower() for t in self.lexical_graph.vocabulary}:
            return True
        
        # Check online dictionary if enabled
        if self.enable_dictionary:
            word_info = self._lookup_dictionary(term)
            if word_info and word_info.get('found'):
                # Add to vocabulary cache
                self.lexical_graph.vocabulary.add(term_lower)
                return True
        
        return False
    
    def _lookup_dictionary(self, word: str) -> Optional[Dict[str, Any]]:
        """
        Look up a word in the online dictionary.
        
        Uses Free Dictionary API (dictionaryapi.dev) - no API key required.
        
        Args:
            word: Word to look up
            
        Returns:
            Dictionary with word information or None if not found
        """
        if not self.enable_dictionary:
            return None
        
        word_lower = word.lower().strip()
        
        # Check cache first
        if self.cache_enabled and word_lower in self._dictionary_cache:
            cache_time = self._dictionary_cache_timestamps.get(word_lower, 0)
            if time.time() - cache_time < self._cache_ttl:
                return self._dictionary_cache[word_lower]
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            time.sleep(self._min_request_interval - time_since_last)
        
        try:
            # Free Dictionary API - no API key needed
            url = f"{self.dictionary_api_url}/{word_lower}"
            response = requests.get(url, timeout=5)
            self._last_request_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    # Extract word information
                    word_data = data[0]
                    
                    word_info = {
                        'found': True,
                        'word': word_data.get('word', word_lower),
                        'phonetic': word_data.get('phonetic', ''),
                        'meanings': [],
                        'synonyms': set(),
                        'antonyms': set(),
                        'part_of_speech': []
                    }
                    
                    # Extract meanings, synonyms, antonyms
                    for meaning in word_data.get('meanings', []):
                        pos = meaning.get('partOfSpeech', '')
                        word_info['part_of_speech'].append(pos)
                        
                        meaning_entry = {
                            'part_of_speech': pos,
                            'definitions': [],
                            'synonyms': [],
                            'antonyms': []
                        }
                        
                        for definition in meaning.get('definitions', []):
                            meaning_entry['definitions'].append({
                                'definition': definition.get('definition', ''),
                                'example': definition.get('example', '')
                            })
                        
                        # Collect synonyms and antonyms
                        for syn in meaning.get('synonyms', []):
                            word_info['synonyms'].add(syn.lower())
                            meaning_entry['synonyms'].append(syn.lower())
                        
                        for ant in meaning.get('antonyms', []):
                            word_info['antonyms'].add(ant.lower())
                            meaning_entry['antonyms'].append(ant.lower())
                        
                        word_info['meanings'].append(meaning_entry)
                    
                    # Cache the result
                    if self.cache_enabled:
                        self._dictionary_cache[word_lower] = word_info
                        self._dictionary_cache_timestamps[word_lower] = time.time()
                    
                    return word_info
            
            elif response.status_code == 404:
                # Word not found
                word_info = {'found': False, 'word': word_lower}
                if self.cache_enabled:
                    self._dictionary_cache[word_lower] = word_info
                    self._dictionary_cache_timestamps[word_lower] = time.time()
                return word_info
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"Dictionary lookup failed for '{word}': {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error in dictionary lookup for '{word}': {e}")
            return None
        
        return None
    
    def get_word_info(self, word: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive word information from dictionary.
        
        Args:
            word: Word to look up
            
        Returns:
            Dictionary with word information (definitions, synonyms, part of speech, etc.)
        """
        return self._lookup_dictionary(word)
    
    def is_valid_word(self, word: str) -> bool:
        """
        Check if a word exists in the dictionary.
        
        Args:
            word: Word to check
            
        Returns:
            True if word exists, False otherwise
        """
        if not word or len(word.strip()) == 0:
            return False
        
        # Check cache first
        word_lower = word.lower().strip()
        if self.cache_enabled and word_lower in self._dictionary_cache:
            cache_time = self._dictionary_cache_timestamps.get(word_lower, 0)
            if time.time() - cache_time < self._cache_ttl:
                cached = self._dictionary_cache[word_lower]
                return cached.get('found', False)
        
        # Look up in dictionary
        word_info = self._lookup_dictionary(word)
        return word_info is not None and word_info.get('found', False)
    
    def get_synonyms(self, word: str) -> Set[str]:
        """
        Get synonyms for a word using dictionary.
        
        Args:
            word: Word to get synonyms for
            
        Returns:
            Set of synonyms
        """
        synonyms = set()
        
        # Check local synonym sets first
        local_synonyms = self.expand_vocabulary(word)
        synonyms.update(local_synonyms)
        
        # Look up in dictionary
        if self.enable_dictionary:
            word_info = self._lookup_dictionary(word)
            if word_info and word_info.get('found'):
                dict_synonyms = word_info.get('synonyms', set())
                synonyms.update(dict_synonyms)
                
                # Add to local synonym sets for future use
                if dict_synonyms:
                    canonical = word_info.get('word', word.lower())
                    self.add_synonym_set(canonical, dict_synonyms)
        
        return synonyms
    
    def get_part_of_speech(self, word: str) -> List[str]:
        """
        Get part of speech for a word.
        
        Args:
            word: Word to check
            
        Returns:
            List of parts of speech (noun, verb, adjective, etc.)
        """
        if not self.enable_dictionary:
            return []
        
        word_info = self._lookup_dictionary(word)
        if word_info and word_info.get('found'):
            return word_info.get('part_of_speech', [])
        
        return []
    
    def is_action_verb(self, word: str) -> bool:
        """
        Check if a word is an action verb using dictionary.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is a verb, False otherwise
        """
        pos_list = self.get_part_of_speech(word)
        return 'verb' in pos_list
    
    def is_noun(self, word: str) -> bool:
        """
        Check if a word is a noun using dictionary.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is a noun, False otherwise
        """
        pos_list = self.get_part_of_speech(word)
        return 'noun' in pos_list


class GrammaticalCompiler:
    """
    Compiles grammatical knowledge (sentence structure) into queryable patterns.
    
    Features:
    - Dependency grammar rules
    - Causal expression patterns
    - Active/passive transformation
    - Tense and aspect for temporal causality
    """
    
    def __init__(self):
        """Initialize grammatical compiler with pattern definitions."""
        self.causal_patterns = self._build_causal_patterns()
        self.dependency_patterns = self._build_dependency_patterns()
    
    def _build_causal_patterns(self) -> List[Tuple[str, str, float]]:
        """
        Build patterns for causal expressions.
        
        Returns:
            List of (pattern, relation_type, confidence) tuples
        """
        return [
            # Direct causal verbs
            (r'(\w+(?:\s+\w+)?)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(\w+(?:\s+\w+)?)', 'causes', 0.95),
            (r'(\w+(?:\s+\w+)?)\s+(?:affects?|influences?|impacts?)\s+(\w+(?:\s+\w+)?)', 'affects', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+(?:depends?\s+on|depends?\s+upon)\s+(\w+(?:\s+\w+)?)', 'depends_on', 0.9),
            (r'(\w+(?:\s+\w+)?)\s+(?:determines?|controls?|drives?)\s+(\w+(?:\s+\w+)?)', 'determines', 0.95),
            
            # Passive voice
            (r'(\w+(?:\s+\w+)?)\s+is\s+(?:caused|affected|influenced|determined)\s+by\s+(\w+(?:\s+\w+)?)', 'caused_by', 0.95),
            (r'(\w+(?:\s+\w+)?)\s+results?\s+from\s+(\w+(?:\s+\w+)?)', 'results_from', 0.9),
            
            # Conditional
            (r'if\s+(\w+(?:\s+\w+)?)\s+then\s+(\w+(?:\s+\w+)?)', 'conditional', 0.85),
            (r'when\s+(\w+(?:\s+\w+)?)\s*,\s*(\w+(?:\s+\w+)?)', 'temporal', 0.8),
        ]
    
    def _build_dependency_patterns(self) -> List[Tuple[str, str]]:
        """
        Build dependency grammar patterns.
        
        Returns:
            List of (pattern, relation) tuples
        """
        return [
            # Subject-verb-object
            (r'(\w+)\s+(\w+)\s+(\w+)', 'SVO'),
            # Prepositional phrases
            (r'(\w+)\s+(?:in|on|at|by|with|for|from|to)\s+(\w+)', 'PREP'),
        ]
    
    def parse_dependencies(self, sentence: str) -> DependencyTree:
        """
        Parse sentence into dependency tree.
        
        Args:
            sentence: Sentence to parse
            
        Returns:
            DependencyTree representation
        """
        words = sentence.split()
        nodes = words
        edges = []
        
        # Simple dependency parsing (subject-verb-object)
        # In a full implementation, would use proper dependency parser
        if len(words) >= 3:
            # Assume first word is subject, second is verb, third is object
            edges.append((words[0], words[1], "nsubj"))  # subject
            edges.append((words[1], words[2], "dobj"))  # object
        
        return DependencyTree(nodes=nodes, edges=edges, root=words[0] if words else None)
    
    def extract_causal_structure(self, parse_tree: DependencyTree) -> Optional[CausalStructure]:
        """
        Extract causal structure from dependency parse tree.
        
        Args:
            parse_tree: DependencyTree to analyze
            
        Returns:
            CausalStructure if found, None otherwise
        """
        # Match against causal patterns
        sentence = ' '.join(parse_tree.nodes)
        
        for pattern, relation_type, confidence in self.causal_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    cause = match.group(1).strip()
                    effect = match.group(2).strip()
                    return CausalStructure(
                        cause=cause,
                        effect=effect,
                        relation_type=relation_type,
                        confidence=confidence
                    )
        
        return None
    
    def transform_voice(self, sentence: str, target_voice: str) -> str:
        """
        Transform sentence between active and passive voice.
        
        Args:
            sentence: Sentence to transform
            target_voice: Target voice ("active" or "passive")
            
        Returns:
            Transformed sentence
        """
        # Simplified transformation
        # In full implementation, would use proper grammar rules
        
        if target_voice == "passive":
            # Simple active to passive: "X causes Y" -> "Y is caused by X"
            active_pattern = r'(\w+(?:\s+\w+)?)\s+(causes?|affects?|influences?)\s+(\w+(?:\s+\w+)?)'
            match = re.search(active_pattern, sentence, re.IGNORECASE)
            if match:
                subject = match.group(1)
                verb = match.group(2)
                object_ = match.group(3)
                
                # Convert verb to past participle
                verb_map = {
                    "causes": "caused",
                    "cause": "caused",
                    "affects": "affected",
                    "affect": "affected",
                    "influences": "influenced",
                    "influence": "influenced"
                }
                past_participle = verb_map.get(verb.lower(), verb + "ed")
                
                return f"{object_} is {past_participle} by {subject}"
        
        return sentence


class PragmaticCompiler:
    """
    Compiles pragmatic knowledge (tone and style) based on graph properties.
    
    Maps graph properties to language decisions:
    - Low confidence → hedging ("may", "possibly", "uncertain")
    - High confidence → assertive ("will", "determines", "causes")
    - Complex graph → explanatory phrasing
    - Simple graph → concise phrasing
    """
    
    def __init__(self):
        """Initialize pragmatic compiler."""
        self.hedging_phrases = [
            "may", "might", "possibly", "potentially", "could",
            "uncertain", "unclear", "suggests", "indicates"
        ]
        self.assertive_phrases = [
            "will", "determines", "causes", "leads to", "results in",
            "clearly", "definitely", "certainly", "always"
        ]
    
    def select_register(self, confidence: float, complexity: int) -> str:
        """
        Select appropriate language register based on confidence and complexity.
        
        Args:
            confidence: Confidence level (0.0-1.0)
            complexity: Graph complexity (number of nodes/edges)
            
        Returns:
            Register type ("formal", "informal", "technical", "casual")
        """
        if confidence > 0.8 and complexity < 10:
            return "assertive"
        elif confidence < 0.5:
            return "hedging"
        elif complexity > 20:
            return "explanatory"
        else:
            return "neutral"
    
    def generate_hedging(self, confidence: float) -> str:
        """
        Generate hedging phrase based on confidence level.
        
        Args:
            confidence: Confidence level (0.0-1.0)
            
        Returns:
            Hedging phrase
        """
        if confidence > 0.7:
            return "likely"
        elif confidence > 0.5:
            return "possibly"
        elif confidence > 0.3:
            return "may"
        else:
            return "uncertain"
    
    def adjust_explicitness(self, depth: int) -> int:
        """
        Adjust explicitness level based on reasoning depth.
        
        Args:
            depth: Depth of reasoning chain
            
        Returns:
            Explicitness level (0-5, where 5 is most explicit)
        """
        if depth <= 1:
            return 1  # Concise
        elif depth <= 3:
            return 3  # Moderate
        else:
            return 5  # Very explicit
