# Risk Management Framework

CRCA-Q implements comprehensive risk management with multiple layers of protection.

## Overview

Risk management includes position limits, leverage constraints, correlation limits, volatility limits, drawdown limits, and circuit breakers.

## Risk Constraints

### Position Size Limits

Maximum position per asset:

$$|w_i| \leq w_{max}$$

Where $w_{max}$ is typically 15-20% of portfolio.

### Leverage Limits

Total leverage constraint:

$$\sum_i |w_i| \leq L_{max}$$

Where $L_{max}$ is maximum leverage (default: 1.0 for no leverage).

### Correlation Limits

Prevent over-concentration:

$$\rho_{i,j} \leq \rho_{max}$$

Where $\rho_{max}$ is typically 0.8.

### Volatility Limits

Portfolio volatility constraint:

$$\sqrt{\mathbf{w}' \boldsymbol{\Sigma} \mathbf{w}} \leq \sigma_{max}$$

Where $\sigma_{max}$ is maximum portfolio volatility (default: 20% annualized).

### Drawdown Limits

Maximum allowed drawdown:

$$\text{DD}_t = \frac{\text{Peak}_t - \text{Value}_t}{\text{Peak}_t} \leq \text{DD}_{max}$$

Where $\text{DD}_{max}$ is typically 15%.

## Circuit Breakers

### Daily Loss Limit

Maximum daily loss:

$$\text{Daily Loss}_t \leq \text{Loss}_{max}$$

Where $\text{Loss}_{max}$ is typically 5% of portfolio.

### Trade Frequency Limit

Maximum trades per day:

$$N_{trades} \leq N_{max}$$

Where $N_{max}$ is typically 50 trades/day.

### Kill Switch

Manual emergency stop via file check.

## Mathematical Foundation

Risk-adjusted returns use Sharpe ratio:

$$\text{Sharpe} = \frac{E[R] - R_f}{\sigma(R)}$$

Where $R_f$ is the risk-free rate.

## Usage

```python
from branches.CRCA-Q import RiskMonitor

monitor = RiskMonitor(
    max_position_size=0.15,
    max_leverage=1.0,
    max_drawdown=0.15
)

is_valid, reason, adjusted_size = monitor.pre_trade_check(
    signal='BUY',
    position_size=0.20,
    current_positions={},
    portfolio_value=10000
)
```

## Next Steps

- [Setup](setup.md) - Setup and configuration
