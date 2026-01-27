"""
Enhanced Hybrid Agent - General Knowledge Examples

Demonstrates the hybrid agent's ability to handle general knowledge tasks,
not just causal reasoning. Shows taxonomic, spatial, temporal, and definitional relationships.
"""

from architecture.hybrid.hybrid_agent import HybridAgent

# ============================================================================
# EXAMPLE 1: Taxonomic Relationships (Is-A)
# ============================================================================

agent = HybridAgent(graph_type="knowledge")  # Use knowledge graph type
response = agent.run("A dog is a mammal. A mammal is an animal.")
print("=== Example 1: Taxonomic Relationships ===")
print(response)
print("\n")

# ============================================================================
# EXAMPLE 2: Spatial Relationships
# ============================================================================

response = agent.run("Paris is in France. France is in Europe.")
print("=== Example 2: Spatial Relationships ===")
print(response)
print("\n")

# ============================================================================
# EXAMPLE 3: Definitions
# ============================================================================

response = agent.run("What is a mammal?")
print("=== Example 3: Definition Query ===")
print(response)
print("\n")

# ============================================================================
# EXAMPLE 4: Mixed Causal and General Knowledge
# ============================================================================

agent_mixed = HybridAgent(graph_type="mixed")  # Mixed graph type
response = agent_mixed.run("Price depends on demand. A product is a good. Demand affects sales.")
print("=== Example 4: Mixed Relationships ===")
print(response)
print("\n")

# ============================================================================
# EXAMPLE 5: Part-Whole Relationships
# ============================================================================

response = agent.run("A wheel is part of a car. A car has an engine.")
print("=== Example 5: Part-Whole Relationships ===")
print(response)
print("\n")

# ============================================================================
# EXAMPLE 6: Temporal Relationships
# ============================================================================

response = agent.run("Spring occurs before summer. Summer happens after spring.")
print("=== Example 6: Temporal Relationships ===")
print(response)
print("\n")

# ============================================================================
# EXAMPLE 7: Functional Relationships
# ============================================================================

response = agent.run("A hammer is used for hitting nails. A pen functions as a writing tool.")
print("=== Example 7: Functional Relationships ===")
print(response)
print("\n")

# ============================================================================
# EXAMPLE 8: Complex General Knowledge Query
# ============================================================================

response = agent.run("What is Paris? Where is it located?")
print("=== Example 8: Complex General Knowledge Query ===")
print(response)
print("\n")

# ============================================================================
# EXAMPLE 9: Conversational General Knowledge
# ============================================================================

# First message
response1 = agent.run("A cat is a mammal.")
print("=== Example 9a: First Message ===")
print(response1)
print("\n")

# Follow-up with reference
response2 = agent.run("What else is a mammal?")
print("=== Example 9b: Follow-up with Reference ===")
print(response2)
print("\n")

# ============================================================================
# EXAMPLE 10: Show Reasoning for General Knowledge
# ============================================================================

response = agent.run("A bird is an animal. An animal is a living thing.", show_reasoning=True)
print("=== Example 10: General Knowledge with Reasoning ===")
print(response)
print("\n")
