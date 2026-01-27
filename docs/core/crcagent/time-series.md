# Time-Series Analysis

CRCAAgent supports time-series causal analysis including Granger causality and VAR models.

## Overview

Time-series analysis identifies causal relationships in temporal data, accounting for temporal dependencies.

## Mathematical Foundation

### Vector Autoregression (VAR)

A VAR($p$) model for $k$ variables:

$$\mathbf{y}_t = \mathbf{c} + \sum_{i=1}^p \mathbf{\Phi}_i \mathbf{y}_{t-i} + \boldsymbol{\epsilon}_t$$

Where:
- $\mathbf{y}_t$: Vector of variables at time $t$
- $\mathbf{\Phi}_i$: Coefficient matrices
- $\boldsymbol{\epsilon}_t$: Error terms

### Granger Causality

$X$ Granger-causes $Y$ if:

$$E[Y_t | \mathcal{I}_{t-1}] \neq E[Y_t | \mathcal{I}_{t-1} \setminus X_{t-1}]$$

Where $\mathcal{I}_{t-1}$ is the information set at time $t-1$.

## Usage

### VAR Estimation

```python
# Estimate VAR model
var_model = agent.estimate_var(
    data=dataframe,
    variables=["X", "Y", "Z"],
    lags=4
)
```

### Granger Causality Test

```python
# Test if X Granger-causes Y
result = agent.granger_causality_test(
    data=dataframe,
    cause="X",
    effect="Y",
    max_lag=4
)
```

## Next Steps

- [Bayesian Inference](bayesian-inference.md) - Bayesian time-series methods
