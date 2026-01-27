"""
Enhanced Hybrid Agent Usage Examples

Demonstrates the enhanced hybrid agent that can replace LLMs entirely
for causal reasoning tasks with natural language interaction.
"""

from architecture.hybrid.hybrid_agent import HybridAgent

# ============================================================================
# EXAMPLE 1: Basic Usage (Conversational Response)
# ============================================================================

agent = HybridAgent()
response = agent.run("If product price is 20000 and demand is 61%, what is the expected price in 7 days?")
print("=== Example 1: Basic Usage ===")
print(response)
print("\n")

# ============================================================================
# EXAMPLE 2: Different Response Styles
# ============================================================================

task = "Product quality affects customer satisfaction, which influences sales revenue"

# Conversational (default - natural, LLM-like)
response_conv = agent.run(task, response_style='conversational')
print("=== Example 2a: Conversational Style ===")
print(response_conv)
print("\n")

# Brief summary
response_brief = agent.run(task, response_style='brief')
print("=== Example 2b: Brief Style ===")
print(response_brief)
print("\n")

# Full analysis
response_full = agent.run(task, response_style='full')
print("=== Example 2c: Full Style ===")
print(response_full[:500] + "...")  # Truncate for display
print("\n")

# ============================================================================
# EXAMPLE 3: Graph-First Query
# ============================================================================

# Build graph first
agent.run("Price depends on demand. Demand affects supply. Supply influences cost.")

# Query graph directly
graph_result = agent.query_graph("What is the relationship between price and cost?")
print("=== Example 3: Graph-First Query ===")
print(graph_result)
print("\n")

# ============================================================================
# EXAMPLE 4: Error Handling (Stability)
# ============================================================================

# Empty input
response = agent.run("")
print("=== Example 4a: Empty Input ===")
print(response)
print("\n")

# Invalid input
response = agent.run(None)  # Will be caught and handled gracefully
print("=== Example 4b: Invalid Input ===")
print(response)
print("\n")

# ============================================================================
# EXAMPLE 5: Complex Causal Chain
# ============================================================================

complex_task = """
If marketing spend increases by 20%, brand awareness improves, 
which leads to higher customer acquisition. Customer acquisition 
affects revenue, and revenue influences profit margins.
"""

response = agent.run(complex_task, response_style='conversational')
print("=== Example 5: Complex Causal Chain ===")
print(response)
print("\n")

# ============================================================================
# EXAMPLE 6: Question Answering
# ============================================================================

agent.run("Temperature affects pressure. Pressure influences flow rate.")

question = "How does temperature affect flow rate?"
response = agent.run(question, response_style='conversational')
print("=== Example 6: Question Answering ===")
print(response)
print("\n")

# ============================================================================
# EXAMPLE 7: Counterfactual Analysis
# ============================================================================

counterfactual_task = """
If we increase employee training by 30%, productivity improves by 15%.
Productivity affects output quality, which influences customer satisfaction.
What if we increase training by 50% instead?
"""

response = agent.run(counterfactual_task, response_style='conversational')
print("=== Example 7: Counterfactual Analysis ===")
print(response[:800] + "...")  # Truncate for display
print("\n")

# ============================================================================
# EXAMPLE 8: Text Correction (Handles Typos)
# ============================================================================

typo_task = "Prce depnds on demnad and suply affects cost"  # Intentional typos
response = agent.run(typo_task, response_style='conversational')
print("=== Example 8: Text Correction (Typos) ===")
print(response)
print("\n")

# ============================================================================
# EXAMPLE 9: Multi-Turn Conversation
# ============================================================================

print("=== Example 9: Multi-Turn Conversation ===")

# Turn 1: Establish context
agent.run("Customer satisfaction depends on product quality and service quality.")

# Turn 2: Ask question
response1 = agent.run("What affects customer satisfaction?", response_style='conversational')
print("Q: What affects customer satisfaction?")
print(f"A: {response1[:200]}...\n")

# Turn 3: Follow-up
response2 = agent.run("How does product quality affect it?", response_style='conversational')
print("Q: How does product quality affect it?")
print(f"A: {response2[:200]}...\n")

# ============================================================================
# EXAMPLE 10: Production-Ready Usage
# ============================================================================

def analyze_causal_relationship(description: str) -> str:
    """
    Production-ready function for causal analysis.
    
    Args:
        description: Natural language description of causal relationships
        
    Returns:
        Analysis result as natural language
    """
    agent = HybridAgent(
        enable_graph_first=True,
        enable_compression=True,
        enable_language_compilation=True,
        enable_error_correction=True
    )
    
    try:
        return agent.run(description, response_style='conversational')
    except Exception as e:
        return f"Analysis failed: {str(e)}"

# Usage
result = analyze_causal_relationship(
    "Employee motivation drives productivity. Productivity affects company revenue."
)
print("=== Example 10: Production-Ready Function ===")
print(result)
print("\n")

print("All examples completed successfully!")
print("\nThe hybrid agent can now replace LLMs for causal reasoning tasks!")
