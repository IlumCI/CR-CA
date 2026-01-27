# Market Data Client

MarketDataClient fetches and normalizes market data from multiple sources.

## Overview

MarketDataClient provides a unified interface for accessing market data from various sources, normalizing it to a consistent schema.

## Data Sources

- **Crypto**: CoinGecko API (free tier)
- **Stocks**: Yahoo Finance (via `yfinance`)
- **FX/Futures**: Yahoo Finance with symbol transformations

## Unified Schema

All data is normalized to:

```python
{
    'date': pd.Timestamp,
    'price': float,      # Close price
    'volume': float,     # Trading volume
    'market_cap': float, # Market capitalization
    'returns': float     # Computed returns
}
```

## Usage

```python
from branches.CRCA-Q import MarketDataClient

client = MarketDataClient()

# Fetch single asset
data = client.fetch_price_data("BTC", days=365)

# Fetch multiple assets
multi_data = client.fetch_multiple_assets(["BTC", "ETH"], days=365)

# Compute covariance
covariance = client.compute_multi_asset_covariance(multi_data)
```

## Mathematical Foundation

Returns are computed as:

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{\Delta P_t}{P_{t-1}}$$

Covariance is estimated using EWMA:

$$\Sigma_t = (1-\alpha) \Sigma_{t-1} + \alpha r_t r_t'$$

Where $\alpha = 0.06$ (decay factor).

## Next Steps

- [Signal Generation](signal-generation.md) - Generate trading signals
