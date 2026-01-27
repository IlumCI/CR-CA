# Schemas Overview

The `schemas/` module provides type definitions and data structures for CR-CA.

## Modules

- **[Annotation](annotation.md)**: Image annotation schemas
- **[Conversation](conversation.md)**: Conversation schemas
- **[Hybrid](hybrid.md)**: Hybrid agent schemas
- **[Reasoning](reasoning.md)**: Reasoning schemas
- **[Policy](policy.md)**: Policy engine schemas

## Usage

Schemas are Pydantic models that provide type safety and validation:

```python
from schemas.annotation import AnnotationResult

result = AnnotationResult(
    entities=[],
    labels=[],
    relations=[]
)
```

## Next Steps

- [Annotation](annotation.md) - Image annotation schemas
- [Policy](policy.md) - Policy engine schemas
