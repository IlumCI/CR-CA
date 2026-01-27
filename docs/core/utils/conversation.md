# Conversation Management

The conversation module manages conversation state and history for CR-CA agents.

## Overview

Conversation management tracks dialogue context, maintains history, and manages multi-turn interactions.

## Usage

```python
from utils.conversation import ConversationManager

manager = ConversationManager()
manager.add_message("user", "Analyze X -> Y")
manager.add_message("assistant", result)
```

## Features

- Conversation history tracking
- Context management
- Multi-turn dialogue support

## Next Steps

- [Graph Reasoner](graph-reasoner.md) - Graph reasoning utilities
