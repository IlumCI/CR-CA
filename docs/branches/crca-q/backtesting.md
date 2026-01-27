# Backtesting Engine

Backtesting engine for evaluating trading strategies.

## Overview

Backtesting engine performs walk-forward analysis to evaluate strategy performance on historical data.

## Mathematical Foundation

### Walk-Forward Analysis

Rolling window approach:

- Train window: $[T-H, T]$
- Test window: $[T, T+h]$
- Step forward: $T \leftarrow T + h$

### Performance Metrics

**Sharpe Ratio**:
$$Sharpe = \frac{E[R] - R_f}{\sigma(R)}$$

**Sortino Ratio**:
$$Sortino = \frac{E[R] - R_f}{\sigma_{down}(R)}$$

**Max Drawdown**:
$$MDD = \max_t \frac{Peak_t - Value_t}{Peak_t}$$

## Usage

```python
from branches.CRCA-Q import BacktestEngine

engine = BacktestEngine()
results = engine.backtest(
    strategy=strategy,
    data=historical_data,
    initial_capital=100000
)
```

## Next Steps

- [Setup](setup.md) - Setup and configuration
