"""
Hybrid Agent Dictionary Integration Demo

Demonstrates the online dictionary integration (Free Dictionary API - no API key required).
"""

from architecture.hybrid.hybrid_agent import HybridAgent
from architecture.hybrid.language_compiler import LexicalCompiler

# ============================================================================
# EXAMPLE 1: Dictionary Lookup
# ============================================================================

print("=== Example 1: Dictionary Lookup ===\n")

compiler = LexicalCompiler(enable_dictionary=True)

# Look up word information
word_info = compiler.get_word_info("price")
if word_info and word_info.get('found'):
    print(f"Word: {word_info['word']}")
    print(f"Part of Speech: {word_info['part_of_speech']}")
    print(f"Synonyms: {list(word_info['synonyms'])[:5]}")
    print()

# ============================================================================
# EXAMPLE 2: Validate Words
# ============================================================================

print("=== Example 2: Word Validation ===\n")

words_to_check = ["price", "demand", "xyzabc123", "identify", "policy"]

for word in words_to_check:
    is_valid = compiler.is_valid_word(word)
    pos = compiler.get_part_of_speech(word)
    is_verb = compiler.is_action_verb(word)
    is_noun = compiler.is_noun(word)
    
    print(f"'{word}':")
    print(f"  Valid: {is_valid}")
    print(f"  Part of Speech: {pos}")
    print(f"  Is Verb: {is_verb}")
    print(f"  Is Noun: {is_noun}")
    print()

# ============================================================================
# EXAMPLE 3: Get Synonyms
# ============================================================================

print("=== Example 3: Synonym Expansion ===\n")

word = "price"
synonyms = compiler.get_synonyms(word)
print(f"Synonyms for '{word}': {list(synonyms)[:10]}")
print()

# ============================================================================
# EXAMPLE 4: Dictionary-Enhanced Variable Filtering
# ============================================================================

print("=== Example 4: Dictionary-Enhanced Agent ===\n")

agent = HybridAgent(
    enable_language_compilation=True,  # Enables dictionary
    enable_error_correction=True
)

# Test with a task that has some invalid words
task = "If product prce is 20000 and demnad is 61%, what is the expected price?"

response = agent.run(task, response_style='conversational')
print(f"Task: {task}")
print(f"\nResponse:\n{response[:500]}...")
print()

# ============================================================================
# EXAMPLE 5: Dictionary Cache
# ============================================================================

print("=== Example 5: Dictionary Caching ===\n")

# First lookup (will make API call)
import time
start = time.time()
info1 = compiler.get_word_info("demand")
time1 = time.time() - start

# Second lookup (will use cache)
start = time.time()
info2 = compiler.get_word_info("demand")
time2 = time.time() - start

print(f"First lookup: {time1:.3f}s")
print(f"Cached lookup: {time2:.3f}s")
print(f"Cache speedup: {time1/time2:.1f}x faster")
print()

print("Dictionary integration complete!")
print("The hybrid agent now uses online dictionary (no API key required) for:")
print("- Word validation")
print("- Part of speech detection")
print("- Synonym expansion")
print("- Better spelling correction")
