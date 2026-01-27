# Annotation Schemas

Schemas for image annotation system.

## Classes

### AnnotationResult

Complete annotation result.

```python
class AnnotationResult(BaseModel):
    annotation_graph: AnnotationGraph
    overlay_image: Optional[bytes]
    formal_report: str
    json_output: Dict[str, Any]
    processing_time: float
```

### PrimitiveEntity

Geometric primitive entity.

```python
class PrimitiveEntity(BaseModel):
    entity_id: str
    primitive_type: str
    pixel_coords: List[Tuple[int, int]]
    metadata: Dict[str, Any]
```

## Next Steps

- [Conversation](conversation.md) - Conversation schemas
