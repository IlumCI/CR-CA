# General Agent Examples

Comprehensive examples for GeneralAgent usage.

## Quickstart

Basic usage of GeneralAgent:

```python
from branches.general_agent.general_agent import GeneralAgent

agent = GeneralAgent(
    model_name="gpt-4o-mini",
    enable_multimodal=True
)

response = agent.run("Hello, how can you help me?")
```

## Simple Example

Simple example using GeneralAgent:

```python
agent = GeneralAgent(model_name="gpt-4o-mini")
response = agent.run("Analyze this data")
print(response)
```

## Advanced Usage

Advanced usage with personality and multimodal support:

```python
from branches.general_agent.general_agent import GeneralAgent
from branches.general_agent.personality import Personality

personality = Personality(traits=["analytical", "detailed"])
agent = GeneralAgent(
    model_name="gpt-4o-mini",
    enable_multimodal=True,
    personality=personality
)

response = agent.run("Analyze this image and provide detailed analysis")
```

## Next Steps

- [Hybrid Agent Examples](../hybrid-agent/hybrid-agent-examples.md) - Hybrid agent examples
