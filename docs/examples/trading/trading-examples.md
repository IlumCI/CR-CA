# Trading Examples

Comprehensive examples for CRCA-Q quantitative trading.

## Basic Usage

Basic trading with CRCA-Q:

```python
from branches.CRCA-Q import QuantTradingAgent

agent = QuantTradingAgent(
    account_size=10000,
    dry_run=True
)

result = agent.run("BTC")
print(result)
```

## Mathematical Foundation

CRCA-Q uses causal signals validated through:

$$P(Returns | do(Signal = s)) \neq P(Returns)$$

Where signals are validated for causal validity.

## Advanced Trading

Multi-asset and advanced features:

```python
agent = QuantTradingAgent(
    account_size=100000,
    multi_asset_mode=True,
    asset_rotation_enabled=True
)

# Multi-asset analysis
result = agent.run(["BTC", "ETH", "SOL"])
```

## Next Steps

- [Integration Examples](../integration/integration-examples.md) - Integration examples
