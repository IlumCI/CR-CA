# Portfolio Optimization

CRCA-Q uses Conditional Value at Risk (CVaR) optimization for portfolio allocation.

## Overview

Portfolio optimization allocates capital based on expected returns, covariance, and risk constraints.

## Mathematical Foundation

### Optimization Problem

Maximize expected return subject to risk constraints:

$$\max_{\mathbf{w}} \boldsymbol{\mu}' \mathbf{w} - \lambda \mathbf{w}' \boldsymbol{\Sigma} \mathbf{w}$$

Subject to:

$$\sum_{i} |w_i| \leq L$$
$$w_i \geq -L, \quad w_i \leq L$$
$$\sum_{i \in T} |w_i| \leq L_T$$

Where:
- $\boldsymbol{\mu}$: Expected returns vector
- $\boldsymbol{\Sigma}$: Covariance matrix
- $\mathbf{w}$: Portfolio weights
- $\lambda$: Risk aversion parameter
- $L$: Maximum leverage
- $L_T$: Maximum exposure per asset type $T$

### CVaR Optimization

CVaR (Conditional Value at Risk) measures expected loss in worst-case scenarios:

$$\text{CVaR}_\alpha = E[Loss | Loss \geq \text{VaR}_\alpha]$$

Where $\text{VaR}_\alpha$ is the Value at Risk at confidence level $\alpha$.

## Usage

```python
from branches.CRCA-Q import PortfolioOptimizer

optimizer = PortfolioOptimizer()

weights = optimizer.optimize(
    expected_returns=mu,
    covariance=sigma,
    risk_aversion=1.0,
    max_leverage=1.0
)
```

## Covariance Estimation

Covariance is estimated using EWMA:

$$\boldsymbol{\Sigma}_t = (1-\alpha) \boldsymbol{\Sigma}_{t-1} + \alpha \mathbf{r}_t \mathbf{r}_t'$$

With shrinkage for stability:

$$\boldsymbol{\Sigma}_{shrink} = \beta \frac{\text{tr}(\boldsymbol{\Sigma})}{n} \mathbf{I} + (1-\beta) \boldsymbol{\Sigma}$$

## Next Steps

- [Risk Management](risk-management.md) - Risk management framework
