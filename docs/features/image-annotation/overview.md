# Image Annotation Overview

The image annotation system in CR-CA provides automated image analysis with geometric primitive extraction, semantic labeling, and query capabilities.

## Key Features

- **Geometric Primitive Extraction**: Automatically detect circles, lines, contours, and other geometric shapes
- **Semantic Labeling**: Identify and label objects in images using LLM-based analysis
- **Natural Language Queries**: Query images with natural language questions
- **Temporal Tracking**: Track entities across video frames using Kalman filters
- **Batch Processing**: Process multiple images in parallel
- **Integration**: Seamless integration with CRCAAgent and GeneralAgent

## Quick Start

```python
from image_annotation import ImageAnnotationEngine

engine = ImageAnnotationEngine()
result = engine.annotate("path/to/image.png", output="all")

print(f"Found {len(result.annotation_graph.entities)} entities")
print(result.formal_report)
```

## Mathematical Foundation

Image annotation uses computer vision algorithms combined with LLM-based semantic understanding. Geometric primitives are detected using:

- **Circle Detection**: Hough circle transform with parameters $(x_c, y_c, r)$
- **Line Detection**: Hough line transform detecting lines with parameters $(\rho, \theta)$
- **Contour Detection**: Edge detection followed by contour extraction

Semantic labeling uses LLM-based classification to assign labels $L_i$ to entities $E_i$:

$$P(L_i | E_i, I) = LLM(E_i, I)$$

Where $I$ is the image context.

## Integration

- **[Quickstart](quickstart.md)**: Get started quickly
- **[Engine](engine.md)**: AnnotationEngine class details
- **[Geometric Primitives](geometric-primitives.md)**: Primitive extraction
- **[Semantic Labeling](semantic-labeling.md)**: Semantic labeling system
- **[Query Capabilities](query-capabilities.md)**: Natural language queries
- **[Temporal Tracking](temporal-tracking.md)**: Video frame tracking
- **[Integration](integration.md)**: Integration with agents

## Next Steps

- [Quickstart](quickstart.md) - Get started in minutes
- [Engine](engine.md) - Learn about the engine
