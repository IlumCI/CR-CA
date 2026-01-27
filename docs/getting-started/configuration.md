# Configuration

Configure CR-CA for your specific use case. This guide covers environment setup, agent configuration, and advanced settings.

## Environment Configuration

### Required Environment Variables

Create a `.env` file in your project root:

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Model selection (optional, defaults to gpt-4o-mini)
OPENAI_MODEL=gpt-4o-mini
```

### Optional Environment Variables

```bash
# CRCA-Q Trading
KRAKEN_API_KEY=your_kraken_key
KRAKEN_API_SECRET=your_kraken_secret

# Alternative Data Sources
TWITTER_BEARER_TOKEN=your_twitter_token
NEWSAPI_KEY=your_newsapi_key
ETHERSCAN_API_KEY=your_etherscan_key
THEGRAPH_API_KEY=your_thegraph_key

# MCP Servers
MCP_SERVER_URL=http://localhost:8000
MCP_API_KEY=your_mcp_api_key
```

## Agent Configuration

### Basic Configuration

```python
from CRCA import CRCAAgent

agent = CRCAAgent(
    model_name="gpt-4o-mini",
    agent_name="my-agent",
    max_loops=5,
    temperature=0.7
)
```

### Advanced Configuration

```python
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    agent_name="advanced-agent",
    
    # LLM Settings
    max_loops=10,
    temperature=0.7,
    max_tokens=2000,
    
    # Causal Reasoning Settings
    enable_automatic_extraction=True,
    causal_graph_threshold=0.3,
    
    # Simulation Settings
    simulation_steps=100,
    simulation_dt=0.01,
    
    # Optimization Settings
    optimization_method="gradient",
    optimization_max_iter=1000,
    
    # Tools
    use_crca_tools=True,
    use_image_annotation=False
)
```

## Configuration Parameters

### LLM Parameters

- `model_name`: LLM model to use (e.g., "gpt-4o-mini", "gpt-4")
- `max_loops`: Maximum number of reasoning loops
- `temperature`: Sampling temperature (0.0-2.0)
- `max_tokens`: Maximum tokens per response

### Causal Reasoning Parameters

- `enable_automatic_extraction`: Enable automatic variable extraction
- `causal_graph_threshold`: Minimum strength for causal relationships
- `enable_counterfactuals`: Enable counterfactual generation
- `counterfactual_samples`: Number of counterfactual scenarios

### Simulation Parameters

- `simulation_steps`: Number of simulation steps
- `simulation_dt`: Time step size
- `simulation_method`: Simulation method ("euler", "rk4")

### Optimization Parameters

- `optimization_method`: Optimization method ("gradient", "bellman")
- `optimization_max_iter`: Maximum optimization iterations
- `optimization_tolerance`: Convergence tolerance

## Mode Configuration

### LLM Mode (Default)

Uses LLM for causal reasoning and analysis:

```python
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    mode="llm"  # Default
)
```

### Deterministic Mode

Uses mathematical simulation without LLM:

```python
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    mode="deterministic"
)
```

## Policy Engine Configuration

If using the policy engine:

```python
from schemas.policy import DoctrineV1
from utils.ledger import Ledger

# Create doctrine
doctrine = DoctrineV1(
    version="1.0.0",
    objectives=["maximize_efficiency", "minimize_cost"],
    constraints=["budget_limit", "resource_availability"]
)

# Initialize ledger
ledger = Ledger(doctrine=doctrine)

# Use with agent
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    policy_ledger=ledger
)
```

## Image Annotation Configuration

Enable image annotation:

```python
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    use_image_annotation=True,
    image_annotation_config={
        "gpt_model": "gpt-4o-mini",
        "enable_temporal_tracking": False,
        "cache_enabled": True
    }
)
```

## Logging Configuration

Configure logging:

```python
from loguru import logger

# Configure loguru
logger.add(
    "crca.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)

agent = CRCAAgent(
    model_name="gpt-4o-mini",
    verbose=True  # Enable verbose logging
)
```

## Performance Configuration

### Caching

Enable caching for faster repeated queries:

```python
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    cache_enabled=True,
    cache_ttl=3600  # Cache TTL in seconds
)
```

### Parallel Processing

Enable parallel processing for batch operations:

```python
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    parallel_workers=4  # Number of parallel workers
)
```

## Next Steps

- [First Agent](first-agent.md) - Create your first configured agent
- [Core Modules](../core/crcagent/overview.md) - Learn about core functionality
- [Advanced Features](../features/causal-reasoning/overview.md) - Explore advanced features
