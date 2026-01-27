# Personality System

GeneralAgent includes a configurable personality system.

## Overview

The personality system allows configuring agent behavior, tone, and response style.

## Usage

```python
from branches.general_agent.general_agent import GeneralAgent
from branches.general_agent.personality import Personality

personality = Personality(
    traits=["analytical", "helpful", "concise"]
)

agent = GeneralAgent(
    model_name="gpt-4o-mini",
    personality=personality
)
```

## Personality Traits

Common traits include:
- Analytical
- Creative
- Helpful
- Concise
- Detailed

## Next Steps

- [Prompt Builder](prompt-builder.md) - Prompt building utilities
