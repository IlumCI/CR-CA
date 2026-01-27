# General Agent API

API reference for General Agent.

## Classes

### GeneralAgent

General-purpose agent.

```python
class GeneralAgent(BaseSpecializedAgent):
    def __init__(
        self,
        model_name: str,
        enable_multimodal: bool = False,
        personality: Optional[Personality] = None
    ):
        """Initialize general agent."""
```

## Next Steps

- [Image Annotation API](../image-annotation/overview.md) - Image annotation API
