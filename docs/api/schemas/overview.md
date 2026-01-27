# Schemas API Reference

API reference for data schemas.

## Overview

Schemas define Pydantic models for type safety and validation.

## Modules

- **[Annotation](annotation.md)**: Image annotation schemas
- **[Conversation](conversation.md)**: Conversation schemas
- **[Hybrid](hybrid.md)**: Hybrid agent schemas
- **[Reasoning](reasoning.md)**: Reasoning schemas
- **[Policy](policy.md)**: Policy engine schemas

## Usage

```python
from schemas.annotation import AnnotationResult

result = AnnotationResult(
    entities=[],
    labels=[],
    relations=[]
)
```

## Next Steps

- [Annotation](annotation.md) - Annotation schemas
- [Policy](policy.md) - Policy schemas
