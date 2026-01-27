# Feature Mixins

Feature mixins provide reusable functionality that can be mixed into agent classes.

## Overview

Mixins allow adding features to agents without multiple inheritance complexity.

## Available Mixins

- Policy loop functionality
- Graph management
- LLM integration
- Statistical methods

## Usage

```python
from templates.base_specialized_agent import BaseSpecializedAgent
from templates.feature_mixins import PolicyLoopMixin

class MyAgent(BaseSpecializedAgent, PolicyLoopMixin):
    def __init__(self):
        super().__init__()
        # Mixin functionality is now available
```

## Next Steps

- [Policy Loop](policy-loop.md) - Policy loop mixin
- [Graph Management](graph-management.md) - Graph management mixin
