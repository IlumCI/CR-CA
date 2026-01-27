# CRCAAgent Operations

Advanced operations including extraction, simulation, and counterfactuals.

## Variable Extraction

Methods for automatic variable extraction from natural language.

### Mathematical Foundation

Extraction identifies:

$$V = \{V_1, \ldots, V_n\} = Extract(T)$$

Where $T$ is the natural language task.

### Methods

- `extract_variables()`: Extract variables from text
- `extract_relationships()`: Extract causal relationships

## Deterministic Simulation

Methods for deterministic causal simulation.

### Mathematical Foundation

Simulation uses:

$$V_i(t+\Delta t) = V_i(t) + f_i(Pa(V_i(t)), U_i) \cdot \Delta t$$

For differential equation systems.

### Methods

- `simulate()`: Single-step simulation
- `simulate_time_series()`: Multi-step simulation

### Usage

```python
result = agent.simulate(
    variables={"education": 16, "experience": 5},
    target="income"
)
```

## Counterfactual Generation

Methods for generating counterfactual scenarios.

### Mathematical Foundation

Counterfactuals compute:

$$P(Y_{x'} | X=x, Y=y)$$

Using:
1. **Abduction**: $U_Y = f^{-1}(X=x, Z=z, Y=y)$
2. **Action**: Set $X = x'$
3. **Prediction**: $Y_{x'} = f(X=x', Z=z, U_Y)$

### Methods

- `generate_counterfactuals()`: Generate counterfactual scenarios
- `evaluate_counterfactual()`: Evaluate specific counterfactual

### Usage

```python
counterfactuals = agent.generate_counterfactuals(
    intervention={"education": "college"},
    outcome="income"
)
```

## Next Steps

- [Utils API](../utils/overview.md) - Utils API
