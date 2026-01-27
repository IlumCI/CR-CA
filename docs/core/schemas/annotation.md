# Annotation Schemas

Schemas for image annotation system.

## Overview

Annotation schemas define data structures for image annotation results, including entities, labels, and relations.

## Key Schemas

- `AnnotationResult`: Complete annotation result
- `PrimitiveEntity`: Geometric primitive entity
- `SemanticLabel`: Semantic label for entities
- `Relation`: Relationship between entities

## Usage

```python
from schemas.annotation import AnnotationResult, PrimitiveEntity

result = AnnotationResult(
    entities=[PrimitiveEntity(...)],
    labels=[],
    relations=[]
)
```

## Next Steps

- [Conversation](conversation.md) - Conversation schemas
