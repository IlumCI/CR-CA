# AnnotationEngine

The AnnotationEngine class provides the core image annotation functionality.

## Overview

AnnotationEngine processes images to extract geometric primitives, generate semantic labels, and enable natural language queries.

## Initialization

```python
from image_annotation import ImageAnnotationEngine
from image_annotation.annotation_engine import AnnotationConfig

# Basic initialization
engine = ImageAnnotationEngine()

# With custom configuration
config = AnnotationConfig(
    gpt_model="gpt-4o-mini",
    enable_temporal_tracking=True,
    auto_detect_type=True,
    output_format="all"
)
engine = ImageAnnotationEngine(config=config)
```

## Main Methods

### annotate()

Annotate image(s) with full automation:

```python
result = engine.annotate(
    input="image.png",
    frame_id=None,
    output="all"
)
```

Returns `AnnotationResult` object with:
- `annotation_graph`: Graph of entities, labels, and relations
- `overlay_image`: Annotated image overlay
- `formal_report`: Textual report
- `json_output`: JSON representation

### query()

Query an image with natural language:

```python
result = engine.query(
    input="image.png",
    query="find all circles",
    frame_id=None
)
```

Returns dictionary with:
- `answer`: Natural language answer
- `entities`: List of relevant entities
- `measurements`: Dict of measurements
- `graph`: AnnotationGraph if needed

## Mathematical Foundation

The annotation process involves:

1. **Geometric Detection**: Extract primitives using computer vision
2. **Semantic Classification**: Assign labels using LLM

For entity $E_i$ in image $I$:

$$P(L_i | E_i, I) = \text{LLM}(E_i, I)$$

Where $L_i$ is the semantic label.

## Next Steps

- [Geometric Primitives](geometric-primitives.md) - Primitive extraction
- [Semantic Labeling](semantic-labeling.md) - Labeling system
