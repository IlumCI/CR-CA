# CRCA-Q Overview

CRCA-Q is a quantitative trading system that integrates causal reasoning with traditional quantitative finance techniques.

## Executive Summary

CRCA-Q uses **causal inference** to understand *why* market movements occur, enabling more robust predictions that remain valid across different market regimes. Unlike correlation-based systems, CRCA-Q implements Structural Causal Models (SCMs) based on Judea Pearl's framework.

## Key Features

- **Causal Signal Validation**: Validates trading signals using causal reasoning
- **Portfolio Optimization**: CVaR-based portfolio optimization
- **Risk Management**: Multi-layer risk controls and circuit breakers
- **Alternative Data**: Integration with on-chain, social, and news data
- **Backtesting**: Walk-forward analysis and performance metrics

## Mathematical Foundation

CRCA-Q implements Structural Causal Models for market variables:

$$M_t = f_M(U_M, Vol_{t-1})$$
$$Vol_t = f_Vol(U_Vol, L_t)$$
$$L_t = f_L(U_L, Volume_t)$$
$$Price_t = f_Price(U_Price, M_t, Vol_t, L_t)$$

Where:
- $M_t$: Momentum
- $Vol_t$: Volatility
- $L_t$: Liquidity
- $Price_t$: Price

## Philosophy

### Causal vs. Correlational Trading

Traditional systems rely on correlations: "When X happens, Y tends to follow."

CRCA-Q uses causal reasoning: "X *causes* Y through mechanism Z."

This approach is more robust because:
- It explains *why* relationships exist
- It predicts behavior under interventions
- It remains valid across market regimes

## Documentation

- **[Philosophy](philosophy.md)**: Causal vs. correlational trading
- **[Architecture](architecture.md)**: System architecture
- **[Market Data](market-data.md)**: Market data client
- **[Signal Generation](signal-generation.md)**: Signal generation classes
- **[Portfolio Optimization](portfolio-optimization.md)**: Portfolio optimization
- **[Risk Management](risk-management.md)**: Risk management framework
- **[Setup](setup.md)**: Setup and configuration

## Next Steps

- [Philosophy](philosophy.md) - Understand the causal approach
- [Architecture](architecture.md) - System architecture
