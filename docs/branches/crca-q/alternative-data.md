# Alternative Data Sources

CRCA-Q integrates alternative data sources for enhanced signal generation.

## Overview

Alternative data includes on-chain metrics, social sentiment, news, GitHub activity, and exchange metrics.

## Data Sources

### On-Chain Metrics

Ethereum/blockchain metrics:
- Active addresses growth
- Transaction volume trends
- Network growth rate

### Social Sentiment

- Twitter sentiment (via Twitter API v2)
- Reddit sentiment (via Reddit API)
- Social volume metrics

### News Sentiment

- NewsAPI sentiment scores
- Headline sentiment analysis

## Mathematical Foundation

Alternative data signals are weighted by confidence:

$$S_{alt} = \sum_{i} w_i S_i$$

Where:
- $w_i$: Confidence weight for source $i$
- $S_i$: Signal from source $i$

Confidence weights consider:
- Freshness (40%)
- Reliability (40%)
- Stability (20%)

## Usage

```python
from branches.CRCA-Q import AltDataClient

client = AltDataClient()
alt_data = client.fetch_alternative_data("BTC")
```

## Next Steps

- [Signal Generation](signal-generation.md) - Signal generation
