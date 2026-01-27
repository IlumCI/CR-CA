# AnnotationEngine API

API reference for AnnotationEngine class.

## Class Definition

```python
class ImageAnnotationEngine:
    """Image annotation engine."""
```

## Methods

### annotate()

Annotate image(s).

```python
def annotate(
    self,
    input: Union[str, np.ndarray, Image.Image, List, Path],
    frame_id: Optional[int] = None,
    output: Optional[str] = None
) -> Union[AnnotationResult, np.ndarray, Dict[str, Any], str, List]:
    """Annotate image(s)."""
```

### query()

Query image with natural language.

```python
def query(
    self,
    input: Union[str, np.ndarray, Image.Image, Path],
    query: str,
    frame_id: Optional[int] = None
) -> Dict[str, Any]:
    """Query image."""
```

## Mathematical Foundation

Annotation uses:

$$P(L_i | E_i, I) = LLM(E_i, I)$$

Where $L_i$ is label, $E_i$ is entity, and $I$ is image.

## Next Steps

- [Schemas](schemas.md) - Annotation schemas
