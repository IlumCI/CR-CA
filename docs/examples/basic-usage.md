# Basic Usage Examples

Basic examples for getting started with CR-CA.

## Simple Causal Analysis

```python
from CRCA import CRCAAgent

agent = CRCAAgent(
    model_name="gpt-4o-mini",
    agent_name="basic-agent"
)

result = agent.run("What is the causal relationship between education and income?")
print(result)
```

## Causal Graph Construction

```python
# Add variables
agent.add_variable("education")
agent.add_variable("income")
agent.add_variable("experience")

# Add relationships
agent.add_causal_relationship("education", "income", strength=0.6)
agent.add_causal_relationship("experience", "income", strength=0.4)

# View graph
graph = agent.causal_graph
print(graph.get_variables())
```

## Counterfactual Analysis

```python
counterfactuals = agent.generate_counterfactuals(
    intervention={"education": "college"},
    outcome="income"
)

for cf in counterfactuals:
    print(f"Expected income: ${cf.expected_outcome:,.2f}")
```

## Mathematical Foundation

The causal effect is computed as:

$$E[Income | do(Education = college)] = \sum_{z} E[Income | Education = college, Z = z] P(Z = z)$$

## Next Steps

- [General Agent](general-agent/quickstart.md) - General agent examples
- [Hybrid Agent](hybrid-agent/auto-extraction.md) - Hybrid agent examples
