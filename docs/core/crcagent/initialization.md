# CRCAAgent Initialization

This guide covers initializing and configuring CRCAAgent instances.

## Basic Initialization

```python
from CRCA import CRCAAgent

agent = CRCAAgent(
    model_name="gpt-4o-mini",
    agent_name="my-agent"
)
```

## Configuration Parameters

### Required Parameters

- `model_name`: LLM model identifier (e.g., "gpt-4o-mini", "gpt-4")
- `agent_name`: Unique identifier for the agent

### Optional Parameters

```python
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    agent_name="configured-agent",
    
    # LLM Settings
    max_loops=5,
    temperature=0.7,
    max_tokens=2000,
    
    # Causal Reasoning
    enable_automatic_extraction=True,
    causal_graph_threshold=0.3,
    
    # Simulation
    simulation_steps=100,
    simulation_dt=0.01,
    
    # Tools
    use_crca_tools=True,
    use_image_annotation=False
)
```

## Mathematical Configuration

The agent's causal reasoning is configured through structural equations. For a variable $Y$ with parents $Pa(Y)$:

$$Y = f(Pa(Y), U_Y)$$

Where $U_Y$ is an exogenous variable. The agent estimates these functions from data or uses LLM-based inference.

## Mode Selection

### LLM Mode (Default)

Uses LLM for causal reasoning:

```python
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    mode="llm"
)
```

### Deterministic Mode

Uses mathematical simulation:

```python
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    mode="deterministic"
)
```

## Advanced Configuration

See [Configuration Guide](../../getting-started/configuration.md) for detailed configuration options.

## Next Steps

- [Automatic Extraction](automatic-extraction.md) - Extract variables automatically
- [Dual-Mode Operation](dual-mode-operation.md) - Understand mode switching
