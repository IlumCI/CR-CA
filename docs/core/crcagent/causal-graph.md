# Causal Graph Management

CRCAAgent manages causal relationships through directed acyclic graphs (DAGs).

## Graph Structure

A causal graph $G = (V, E)$ consists of:
- **Vertices $V$**: Variables in the system
- **Edges $E$**: Causal relationships $(X \to Y)$

The graph must be acyclic: no directed paths from a variable to itself.

## Mathematical Foundation

For a causal graph, each variable $V_i$ has structural equation:

$$V_i = f_i(Pa(V_i), U_i)$$

Where:
- $Pa(V_i)$: Parents of $V_i$ in the graph
- $U_i$: Exogenous variable for $V_i$

## Building Graphs

### Automatic Construction

```python
agent = CRCAAgent(model_name="gpt-4o-mini")
result = agent.run("Analyze X -> Y -> Z")
# Graph is automatically constructed
```

### Manual Construction

```python
# Add variables
agent.add_variable("education")
agent.add_variable("income")
agent.add_variable("experience")

# Add causal relationships
agent.add_causal_relationship("education", "income", strength=0.6)
agent.add_causal_relationship("education", "experience", strength=0.4)
agent.add_causal_relationship("experience", "income", strength=0.5)
```

## Graph Operations

### Get Variables

```python
variables = agent.causal_graph.get_variables()
```

### Get Relationships

```python
relationships = agent.causal_graph.get_relationships()
for parent, child, strength in relationships:
    print(f"{parent} -> {child}: {strength}")
```

### Check Paths

```python
# Check if path exists
has_path = agent.causal_graph.has_path("X", "Y")

# Get all paths
paths = agent.causal_graph.get_paths("X", "Y")
```

## Causal Identification

The agent can identify causal effects using do-calculus rules:

$$P(Y | do(X=x)) = \sum_{z} P(Y | X=x, Z=z) P(Z=z)$$

Where $Z$ is a valid adjustment set.

## Graph Visualization

```python
# Export graph for visualization
graph_data = agent.causal_graph.export()
# Use with graph visualization libraries
```

## Next Steps

- [Deterministic Simulation](deterministic-simulation.md) - Use graphs for simulation
- [Counterfactuals](counterfactuals.md) - Generate counterfactuals from graphs
