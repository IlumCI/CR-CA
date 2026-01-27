# Batch Prediction

CRCAAgent supports batch processing for multiple scenarios simultaneously.

## Overview

Batch prediction allows processing multiple causal queries in parallel, improving efficiency for large-scale analyses.

## Mathematical Foundation

For a batch of $n$ scenarios, we compute:

$$\{E[Y_i | do(X_i=x_i)]\}_{i=1}^n$$

Where each scenario $i$ has intervention $X_i=x_i$ and target $Y_i$.

## Basic Usage

```python
# Batch of scenarios
scenarios = [
    {"education": 12, "experience": 2},
    {"education": 16, "experience": 5},
    {"education": 20, "experience": 10}
]

results = agent.batch_predict(
    scenarios=scenarios,
    target="income"
)
```

## Vectorized Operations

The agent uses vectorized operations for efficiency:

$$\mathbf{Y} = f(\mathbf{X}, \mathbf{Z}, \mathbf{U})$$

Where bold symbols represent vectors/matrices.

## Parallel Processing

Enable parallel processing:

```python
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    parallel_workers=4
)

results = agent.batch_predict(scenarios, target="income")
```

## Performance

Batch processing is significantly faster than sequential processing:

- **Sequential**: $O(n \cdot t)$ where $t$ is time per scenario
- **Batch**: $O(n \cdot t / p)$ where $p$ is number of workers

## Example

```python
# Generate batch of counterfactuals
interventions = [
    {"education": "college"},
    {"education": "masters"},
    {"education": "phd"}
]

counterfactuals = agent.batch_generate_counterfactuals(
    interventions=interventions,
    outcome="income"
)
```

## Next Steps

- [Async Operations](async-operations.md) - Asynchronous batch processing
- [Optimization](optimization.md) - Optimize batch operations
