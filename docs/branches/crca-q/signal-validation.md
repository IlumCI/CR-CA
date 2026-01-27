# Signal Validation

Signal validation ensures trading signals are causally valid, not just correlated.

## Overview

Signal validation uses causal criteria to filter out spurious signals that may break down in different market regimes.

## Mathematical Foundation

Signals are validated using:

### Causal Score

$$Score(S) = w_1 \cdot MI(S, Y) + w_2 \cdot RI(S, Y) + w_3 \cdot SC(S)$$

Where:
- $MI$: Mutual information (40% weight)
- $RI$: Regime invariance (40% weight)
- $SC$: Structural consistency (20% weight)

### Regime Invariance

$$RI(S, Y) = 1 - \text{Var}(Corr(S, Y | Regime=r))$$

Causal signals should be stable across regimes.

## Usage

```python
from branches.CRCA-Q import SignalValidator

validator = SignalValidator()
score = validator.compute_causal_score(
    signal_name="momentum",
    signal_values=signals,
    target=returns
)
```

## Next Steps

- [Portfolio Optimization](portfolio-optimization.md) - Portfolio optimization
