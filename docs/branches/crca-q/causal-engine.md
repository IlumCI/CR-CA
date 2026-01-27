# Causal Reasoning Engine

The causal reasoning engine validates trading signals using causal inference.

## Overview

Causal engine builds Structural Causal Models and validates signals for causal validity rather than just correlation.

## Mathematical Foundation

Causal engine builds SCM:

$$M_t = f_M(U_M, Vol_{t-1})$$
$$Vol_t = f_Vol(U_Vol, L_t)$$
$$L_t = f_L(U_L, Volume_t)$$
$$Price_t = f_Price(U_Price, M_t, Vol_t, L_t)$$

Signal validation checks:

$$P(Returns | do(Signal=s)) \neq P(Returns)$$

## Usage

```python
from branches.CRCA-Q import CausalEngine

engine = CausalEngine()
validated = engine.validate_signal(signal, target="returns")
```

## Next Steps

- [Signal Validation](signal-validation.md) - Signal validation
