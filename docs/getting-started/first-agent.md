# Creating Your First Agent

This tutorial walks you through creating a complete causal reasoning agent from scratch.

## Example: Education and Income Analysis

Let's build an agent that analyzes the causal relationship between education and income.

### Step 1: Import and Initialize

```python
from CRCA import CRCAAgent

# Create the agent
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    agent_name="education-income-analyzer",
    max_loops=5
)
```

### Step 2: Define Your Causal Question

```python
task = """
Analyze the causal relationship between education and income.

Consider the following factors:
- Years of education
- Type of degree
- Work experience
- Industry sector
- Geographic location

Determine:
1. The direct causal effect of education on income
2. Mediating factors (experience, industry)
3. Confounding variables (geographic location, family background)
4. Counterfactual scenarios (what if education increased by 2 years?)
"""
```

### Step 3: Run the Analysis

```python
result = agent.run(task)
print(result)
```

The agent will automatically:
1. Extract variables from the task
2. Identify causal relationships
3. Build a causal graph
4. Perform causal inference
5. Generate counterfactuals

### Step 4: Examine the Causal Graph

```python
# Get the causal graph
graph = agent.causal_graph

# View all variables
variables = graph.get_variables()
print("Variables:", variables)

# View causal relationships
relationships = graph.get_relationships()
for parent, child, strength in relationships:
    print(f"{parent} -> {child}: {strength:.2f}")
```

### Step 5: Generate Counterfactuals

```python
# What if education increased by 2 years?
counterfactuals = agent.generate_counterfactuals(
    intervention={"education_years": "+2"},
    outcome="income"
)

for cf in counterfactuals:
    print(f"\nCounterfactual: {cf.description}")
    print(f"Expected income change: {cf.expected_outcome}")
```

### Step 6: Run Deterministic Simulation

```python
# Simulate with specific values
simulation = agent.simulate(
    variables={
        "education_years": 16,
        "experience_years": 5,
        "industry": "technology"
    },
    target="income"
)

print(f"Simulated income: ${simulation:,.2f}")
```

## Mathematical Foundation

The agent uses Structural Causal Models (SCMs) to represent causal relationships. For education and income:

$$Income = f(Education, Experience, Industry, Location, \epsilon)$$

Where $\epsilon$ represents unobserved factors.

The causal effect of education on income, controlling for confounders, is:

$$E[Income | do(Education = e)] = \sum_{z} E[Income | Education = e, Z = z] P(Z = z)$$

Where $Z$ represents confounders (experience, industry, location).

## Complete Example

```python
from CRCA import CRCAAgent

# Initialize agent
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    agent_name="education-income-analyzer"
)

# Define task
task = """
Analyze how increasing education from high school to college affects income,
controlling for work experience and industry sector.
"""

# Run analysis
result = agent.run(task)

# Get causal graph
graph = agent.causal_graph
print("Causal Graph Variables:", graph.get_variables())

# Generate counterfactuals
counterfactuals = agent.generate_counterfactuals(
    intervention={"education": "college"},
    outcome="income"
)

# Run simulation
simulation = agent.simulate(
    variables={
        "education": "college",
        "experience": 3,
        "industry": "technology"
    },
    target="income"
)

print(f"\nSimulated Income: ${simulation:,.2f}")
```

## Next Steps

- [Core Modules](../core/crcagent/overview.md) - Learn about core functionality
- [Examples](../examples/overview.md) - Explore more examples
- [API Reference](../api/crca/overview.md) - Complete API documentation
