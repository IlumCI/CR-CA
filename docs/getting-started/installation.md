# Installation

This guide covers installing CR-CA and its dependencies.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Git for cloning the repository
- (Optional) Docker for containerized deployment

## Quick Install

### Method 1: From PyPI (Recommended)

```bash
pip install crca
```

### Method 2: From Source

Clone the repository:

```bash
git clone https://github.com/IlumCI/CR-CA.git
cd CR-CA
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Method 3: Development Install

For development with editable install:

```bash
git clone https://github.com/IlumCI/CR-CA.git
cd CR-CA
pip install -e ".[dev]"
```

## Optional Dependencies

### Policy Engine

```bash
# Install policy engine dependencies
pip install cvxpy ruptures psutil

# Or install all optional dependencies
pip install crca[policy-engine]
```

### CRCA-SD

CRCA-SD dependencies are included in `requirements.txt`. Additional dependencies may be needed for specific features.

### CRCA-CG

```bash
# Install TUI dependencies (optional)
pip install rich

# Or install all optional dependencies
pip install crca[tui]
```

### CRCA-Q

```bash
# Install trading dependencies
pip install ccxt web3

# Or install all optional dependencies
pip install crca[trading]
```

## Environment Variables

Create a `.env` file in the project root with required API keys:

```bash
# Required for LLM functionality
OPENAI_API_KEY=your_openai_api_key_here

# Optional: For CRCA-Q trading
KRAKEN_API_KEY=your_kraken_key_here
KRAKEN_API_SECRET=your_kraken_secret_here

# Optional: For alternative data sources
TWITTER_BEARER_TOKEN=your_twitter_token_here
ETHERSCAN_API_KEY=your_etherscan_key_here
THEGRAPH_API_KEY=your_thegraph_key_here

# Optional: For MCP servers
MCP_SERVER_URL=http://localhost:8000
MCP_API_KEY=your_mcp_api_key_here
```

## Verification

Verify your installation:

```python
from CRCA import CRCAAgent

# This should work without errors
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    agent_name="test-agent"
)
print("Installation successful!")
```

## Troubleshooting

### Windows: cvxpy Installation Issues

If you encounter issues installing `cvxpy` on Windows, you need Microsoft Visual C++ 14.0 or higher:

1. Download [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install the "Desktop development with C++" workload
3. Retry: `pip install cvxpy`

### Import Errors

If you see import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### API Key Issues

Make sure your `.env` file is in the project root and contains valid API keys. The framework will automatically load environment variables from `.env`.

## Next Steps

- [Quickstart Guide](quickstart.md) - Get started with your first agent
- [Configuration](configuration.md) - Configure your environment
- [First Agent](first-agent.md) - Create your first causal reasoning agent
