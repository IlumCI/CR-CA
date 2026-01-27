# Counterfactual Generation

CRCAAgent generates counterfactual scenarios to answer "what-if" questions.

## Overview

Counterfactuals answer questions of the form: "What would have happened if $X$ had been $x'$ instead of $x$, given that we observed $X=x$ and $Y=y$?"

## Mathematical Foundation

A counterfactual query is:

$$P(Y_{x'} | X=x, Y=y)$$

Where:
- $Y_{x'}$: Potential outcome under intervention $do(X=x')$
- $X=x, Y=y$: Observed values

This is computed using the three-step process:

1. **Abduction**: Infer exogenous variables from observations
2. **Action**: Set $X = x'$
3. **Prediction**: Compute $Y$ using structural equations

## Basic Usage

```python
# Generate counterfactuals
counterfactuals = agent.generate_counterfactuals(
    intervention={"education": "college"},
    outcome="income",
    observed={"education": "high_school", "income": 30000}
)
```

## Counterfactual Types

### Simple Counterfactuals

Single intervention:

```python
cf = agent.generate_counterfactuals(
    intervention={"X": x_new},
    outcome="Y"
)
```

### Multiple Interventions

```python
cf = agent.generate_counterfactuals(
    intervention={"X1": x1_new, "X2": x2_new},
    outcome="Y"
)
```

### Conditional Counterfactuals

Given observed values:

```python
cf = agent.generate_counterfactuals(
    intervention={"X": x_new},
    outcome="Y",
    observed={"X": x_old, "Y": y_old}
)
```

## Counterfactual Computation

For a structural equation $Y = f(X, Z, U_Y)$:

1. **Abduction**: $U_Y = f^{-1}(X=x, Z=z, Y=y)$
2. **Action**: Set $X = x'$
3. **Prediction**: $Y_{x'} = f(X=x', Z=z, U_Y)$

## Example

```python
# What if education was college instead of high school?
counterfactuals = agent.generate_counterfactuals(
    intervention={"education": "college"},
    outcome="income",
    observed={"education": "high_school", "income": 30000}
)

for cf in counterfactuals:
    print(f"Expected income: ${cf.expected_outcome:,.2f}")
    print(f"Confidence: {cf.confidence:.2f}")
```

## Next Steps

- [Deterministic Simulation](deterministic-simulation.md) - Understand simulation
- [Batch Prediction](batch-prediction.md) - Process multiple scenarios
