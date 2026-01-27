# Hybrid Agent Examples

Comprehensive examples for hybrid agent usage.

## Auto Extraction

Automatic variable extraction with hybrid agent:

```python
from architecture.hybrid.hybrid_agent import HybridAgent

agent = HybridAgent(model_name="gpt-4o-mini")

task = """
Analyze how increasing minimum wage affects employment.
Consider inflation, business costs, and consumer spending.
"""

result = agent.run(task)
# Variables and relationships are automatically extracted
```

## Dictionary Demo

Dictionary extraction demo:

```python
task = "Extract all key-value pairs from the following text..."
result = agent.run(task)
```

## Enhanced Features

Enhanced hybrid agent features:

```python
agent = HybridAgent(
    model_name="gpt-4o-mini",
    enable_consistency_checking=True,
    enable_self_verification=True
)

result = agent.run("Complex multi-step reasoning task")
```

## General Knowledge

General knowledge tasks:

```python
result = agent.run("Explain quantum computing in simple terms")
```

## Next Steps

- [Image Annotation Examples](../image-annotation/image-annotation-examples.md) - Image annotation examples
