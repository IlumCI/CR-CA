# Templates Overview

The `templates/` module provides a framework for creating specialized agents.

## Modules

- **[Base Agent](base-agent.md)**: Base specialized agent template
- **[Feature Mixins](feature-mixins.md)**: Reusable feature mixins
- **[Policy Loop](policy-loop.md)**: Policy loop mixin
- **[Graph Management](graph-management.md)**: Graph management mixin
- **[LLM Integration](llm-integration.md)**: LLM integration mixin

## Creating Custom Agents

Use templates to create specialized agents:

```python
from templates.base_specialized_agent import BaseSpecializedAgent

class MyAgent(BaseSpecializedAgent):
    def __init__(self):
        super().__init__()
        # Add custom functionality
```

## Next Steps

- [Base Agent](base-agent.md) - Base template
- [Feature Mixins](feature-mixins.md) - Reusable features
