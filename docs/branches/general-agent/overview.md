# General Agent Overview

GeneralAgent is a flexible general-purpose agent with personality system and multimodal capabilities.

## Overview

GeneralAgent provides a general-purpose agent framework that can be customized for various tasks with personality traits and multimodal support.

## Key Features

- **Personality System**: Configurable personality traits
- **Multimodal Support**: Text, image, and other modalities
- **Prompt Building**: Dynamic prompt construction
- **Flexible Architecture**: Extensible for various use cases

## Usage

```python
from branches.general_agent.general_agent import GeneralAgent

agent = GeneralAgent(
    model_name="gpt-4o-mini",
    enable_multimodal=True
)

response = agent.run("Analyze this image and text")
```

## Documentation

- **[Personality](personality.md)**: Personality system
- **[Prompt Builder](prompt-builder.md)**: Prompt building utilities

## Next Steps

- [Personality](personality.md) - Personality system
- [Prompt Builder](prompt-builder.md) - Prompt building
