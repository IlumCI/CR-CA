# Conversation API

API for conversation management.

## Classes

### ConversationManager

Manages conversation state and history.

```python
class ConversationManager:
    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation."""
    
    def get_history(self) -> List[Message]:
        """Get conversation history."""
```

## Next Steps

- [Graph Reasoner](graph-reasoner.md) - Graph reasoner API
