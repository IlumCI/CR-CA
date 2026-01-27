# Dual-Mode Operation

CRCAAgent supports two operational modes: LLM-based analysis and deterministic simulation.

## Modes

### LLM Mode

Uses large language models for causal reasoning and analysis. Suitable for:
- Natural language understanding
- Complex causal reasoning
- Variable extraction
- Qualitative analysis

### Deterministic Mode

Uses mathematical simulation for precise causal inference. Suitable for:
- Quantitative analysis
- Numerical simulations
- Optimization
- Batch processing

## Mathematical Foundation

### LLM Mode

In LLM mode, the agent uses the LLM to infer causal relationships:

$$P(Y | do(X=x), Z=z) \approx LLM(T, X, Y, Z)$$

Where $T$ is the task description, and the LLM approximates the causal effect.

### Deterministic Mode

In deterministic mode, the agent uses structural equations:

$$Y = f(X, Z, U_Y)$$

Where $f$ is a known or estimated function, and $U_Y$ is an exogenous variable.

## Mode Selection

### Automatic Mode Selection

```python
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    auto_mode_selection=True  # Automatically choose mode
)
```

### Manual Mode Selection

```python
# LLM mode
result_llm = agent.run(task, mode="llm")

# Deterministic mode
result_det = agent.run(task, mode="deterministic")
```

## Hybrid Approach

Combine both modes:

```python
# Use LLM for extraction
agent.run(task, mode="llm")

# Switch to deterministic for simulation
simulation = agent.simulate(variables, mode="deterministic")
```

## Performance Considerations

- **LLM Mode**: Slower but more flexible, handles complex reasoning
- **Deterministic Mode**: Faster, precise, requires known structural equations

## Next Steps

- [Deterministic Simulation](deterministic-simulation.md) - Learn about simulations
- [Causal Graph](causal-graph.md) - Understand graph operations
