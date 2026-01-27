# CRCA-Q Setup

Setup and configuration guide for CRCA-Q.

## Installation

```bash
# Install trading dependencies
pip install ccxt web3

# Or install all optional dependencies
pip install crca[trading]
```

## Environment Variables

Create `.env` file:

```bash
# Required for LLM
OPENAI_API_KEY=your_openai_key

# For live trading (Kraken example)
KRAKEN_API_KEY=your_kraken_key
KRAKEN_API_SECRET=your_kraken_secret

# For alternative data
TWITTER_BEARER_TOKEN=your_twitter_token
NEWSAPI_KEY=your_newsapi_key
ETHERSCAN_API_KEY=your_etherscan_key
THEGRAPH_API_KEY=your_thegraph_key
```

## Configuration

```python
TRADING_CONFIG = {
    'account_size': 10000,
    'max_position_size': 0.3,
    'max_position_hard_cap': 0.3,
    'min_trade_value': 5.0,
    'conservative_mode': True,
    'stop_loss_pct': -10.0,
    'stop_gain_pct': 20.0
}
```

## Quick Start

```python
from branches.CRCA-Q import QuantTradingAgent

agent = QuantTradingAgent(
    account_size=10000,
    dry_run=True  # Start in demo mode
)

result = agent.run("BTC")
print(result)
```

## Next Steps

- [Overview](overview.md) - Return to overview
- [Architecture](architecture.md) - System architecture
