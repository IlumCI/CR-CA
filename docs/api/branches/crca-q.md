# CRCA-Q API

API reference for CRCA-Q quantitative trading.

## Classes

### QuantTradingAgent

Main trading agent.

```python
class QuantTradingAgent:
    def run(self, asset: str) -> Dict[str, Any]:
        """Run trading analysis for asset."""
```

## Mathematical Foundation

Trading decisions use:

$$Signal = f(Predictions, Risk, Constraints)$$

Where predictions come from causal models.

## Next Steps

- [CRCA-SD](crca-sd.md) - Socioeconomic dynamics API
