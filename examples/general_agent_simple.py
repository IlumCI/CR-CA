"""
Ultra-Simple GeneralAgent Usage Examples

All examples are under 10 lines of code (LOC).
"""

# ============================================================================
# EXAMPLE 1: Simplest Possible (1 LOC)
# ============================================================================

from branches.general_agent import GeneralAgent
agent = GeneralAgent()
response = agent.run("Hello!")
print(response)

# ============================================================================
# EXAMPLE 2: With Model (2 LOC)
# ============================================================================

agent = GeneralAgent(model_name="gpt-4o")
response = agent.run("Explain causal reasoning.")
print(response)

# ============================================================================
# EXAMPLE 3: With Personality (2 LOC)
# ============================================================================

agent = GeneralAgent(personality="friendly")
response = agent.run("Tell me a joke.")
print(response)

# ============================================================================
# EXAMPLE 4: Using Factory Function (2 LOC)
# ============================================================================

from branches.general_agent import create_agent
agent = create_agent()
response = agent.run("What can you do?")
print(response)

# ============================================================================
# EXAMPLE 5: Using Class Method (2 LOC)
# ============================================================================

agent = GeneralAgent.create(model_name="gpt-4o", personality="technical")
response = agent.run("Analyze this business problem.")
print(response)

# ============================================================================
# EXAMPLE 6: Full Conversation (5 LOC)
# ============================================================================

agent = GeneralAgent()
agent.run("My name is Alice.")
response = agent.run("What's my name?")
print(response)

# ============================================================================
# EXAMPLE 7: Causal Reasoning Task (3 LOC)
# ============================================================================

agent = GeneralAgent()
response = agent.run("Extract causal variables for: product price depends on demand and supply")
print(response)

# ============================================================================
# EXAMPLE 8: Async Usage (4 LOC)
# ============================================================================

import asyncio
agent = GeneralAgent()
response = asyncio.run(agent.run_async("Hello!"))
print(response)

# ============================================================================
# EXAMPLE 9: Batch Processing (4 LOC)
# ============================================================================

agent = GeneralAgent()
tasks = ["What is AI?", "What is ML?", "What is causal reasoning?"]
results = agent.run_batch(tasks)
for r in results:
    print(r)

# ============================================================================
# EXAMPLE 10: Complete Workflow (8 LOC)
# ============================================================================

agent = GeneralAgent(model_name="gpt-4o-mini", personality="neutral")
task = "Help me understand what factors influence customer satisfaction"
response = agent.run(task)
print(f"Task: {task}\nResponse: {response}")
