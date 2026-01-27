# Quickstart Guide

Get started with CR-CA in minutes. This guide walks you through creating your first causal reasoning agent.

## Basic Usage

### 1. Import and Initialize

```python
from CRCA import CRCAAgent

# Create an agent
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    agent_name="my-first-agent"
)
```

### 2. Run a Simple Task

```python
# Ask the agent to analyze a causal relationship
result = agent.run(
    "What is the causal relationship between education and income?"
)

print(result)
```

### 3. Automatic Variable Extraction

CR-CA automatically extracts variables and causal relationships from natural language:

```python
task = """
Analyze how increasing the minimum wage affects employment rates.
Consider factors like inflation, business costs, and consumer spending.
"""

result = agent.run(task)

# The agent automatically:
# 1. Extracts variables (minimum_wage, employment_rate, inflation, etc.)
# 2. Identifies causal relationships
# 3. Builds a causal graph
# 4. Performs causal analysis
```

## Mathematical Foundation

CR-CA uses Structural Causal Models (SCMs) to represent causal relationships. An SCM is defined as:

$$(U, V, F)$$

Where:
- $U$: Exogenous (unobserved) variables
- $V$: Endogenous (observed) variables
- $F$: Structural equations defining causal relationships

The causal effect of intervention $do(X=x)$ on outcome $Y$ is computed using:

$$E[Y | do(X=x)] = \sum_{z} E[Y | X=x, Z=z] P(Z=z)$$

## Working with Causal Graphs

### View the Causal Graph

```python
# After running a task, access the causal graph
graph = agent.causal_graph

# Get all variables
variables = graph.get_variables()
print(f"Variables: {variables}")

# Get causal relationships
relationships = graph.get_relationships()
for parent, child, strength in relationships:
    print(f"{parent} -> {child} (strength: {strength})")
```

### Add Causal Relationships Manually

```python
# Add a causal relationship
agent.add_causal_relationship(
    parent="education",
    child="income",
    strength=0.6
)

# Add multiple relationships
agent.add_causal_relationships([
    ("education", "income", 0.6),
    ("experience", "income", 0.4),
    ("education", "experience", 0.3)
])
```

## Counterfactual Analysis

Generate "what-if" scenarios:

```python
# Generate counterfactuals
counterfactuals = agent.generate_counterfactuals(
    intervention={"education": "college"},
    outcome="income"
)

for scenario in counterfactuals:
    print(f"Scenario: {scenario.description}")
    print(f"Expected outcome: {scenario.expected_outcome}")
```

## Deterministic Simulation

Run deterministic causal simulations:

```python
# Simulate with specific variable values
simulation_result = agent.simulate(
    variables={
        "education": 16,  # years
        "experience": 5,  # years
    },
    target="income"
)

print(f"Simulated income: ${simulation_result:.2f}")
```

## Next Steps

- [Configuration](configuration.md) - Configure advanced settings
- [First Agent](first-agent.md) - Build a complete example
- [Core Modules](../core/crcagent/overview.md) - Learn about core functionality
