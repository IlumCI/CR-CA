# Counterfactual Reasoning

Counterfactual reasoning answers "what-if" questions about alternative scenarios.

## Overview

Counterfactuals enable reasoning about what would have happened under different conditions, given what actually occurred.

## Mathematical Foundation

### Counterfactual Query

A counterfactual query is:

$$P(Y_{x'} | X=x, Y=y)$$

Where:
- $Y_{x'}$: Potential outcome under intervention $do(X=x')$
- $X=x, Y=y$: Observed values

### Three-Step Process

1. **Abduction**: Infer exogenous variables from observations
   $$U_Y = f^{-1}(X=x, Z=z, Y=y)$$

2. **Action**: Set $X = x'$

3. **Prediction**: Compute $Y$ using structural equations
   $$Y_{x'} = f(X=x', Z=z, U_Y)$$

## Usage

```python
counterfactuals = agent.generate_counterfactuals(
    intervention={"education": "college"},
    outcome="income",
    observed={"education": "high_school", "income": 30000}
)
```

## Next Steps

- [Causal Discovery](causal-discovery.md) - Learning causal structure
