# Signal Generation

CRCA-Q generates hundreds of signals across multiple categories for trading decisions.

## Signal Categories

### Time-Series Signals

Technical indicators based on price/volume history:

- **Momentum**: $\log(P_t) - \log(P_{t-k})$ for lookback $k$
- **Reversal**: Negative of recent returns
- **MA Distance**: Distance from price to moving average

### Volatility Signals

Measure and predict volatility:

- **Realized Volatility**: $\sqrt{\frac{1}{n}\sum_{i=1}^n (r_i - \bar{r})^2}$
- **GARCH Volatility**: GARCH(1,1) model
- **Vol of Vol**: Volatility of volatility

### Liquidity Signals

Measure market depth:

- **Amihud Illiquidity**: $\frac{|r_t|}{V_t}$ (price impact per volume)
- **VPIN**: Volume-synchronized probability of informed trading
- **Turnover**: $\frac{V_t}{MC_t}$ (volume to market cap ratio)

## Mathematical Foundation

Signals are computed and then validated using causal reasoning. A signal $S$ is causally valid if:

$$P(Returns | do(S=s)) \neq P(Returns)$$

And the relationship is stable across regimes:

$$\text{Var}(P(Returns | do(S=s), Regime=r)) < \epsilon$$

## Signal Validation

Signals are validated using:

1. **Mutual Information**: Information-theoretic relationship
2. **Regime Invariance**: Stability across market regimes
3. **Structural Consistency**: Mapping to SCM structure

## Usage

```python
from branches.CRCA-Q import QuantTradingAgent

agent = QuantTradingAgent()
signals = agent.compute_signals(data)
validated = agent.validate_signals(signals)
```

## Next Steps

- [Portfolio Optimization](portfolio-optimization.md) - Optimize portfolio allocation
