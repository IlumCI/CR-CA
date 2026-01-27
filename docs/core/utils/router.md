# Router

The router module provides request routing and agent selection functionality.

## Overview

Router helps select appropriate agents for specific tasks based on capabilities and availability.

## Usage

```python
from utils.router import Router

router = Router()
agent = router.select_agent(task="Analyze causal relationships")
```

## Features

- Agent capability matching
- Load balancing
- Fallback mechanisms

## Next Steps

- [Conversation](conversation.md) - Conversation management
- [Graph Reasoner](graph-reasoner.md) - Graph reasoning
