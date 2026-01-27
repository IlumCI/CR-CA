# Image Annotation Quickstart

Get started with image annotation in minutes.

## Basic Annotation

```python
from image_annotation import ImageAnnotationEngine

engine = ImageAnnotationEngine()
result = engine.annotate("path/to/image.png", output="all")

print(f"Found {len(result.annotation_graph.entities)} entities")
print(result.formal_report)
```

## With CRCAAgent

```python
from CRCA import CRCAAgent

agent = CRCAAgent(
    model_name="gpt-4o-mini",
    use_image_annotation=True
)

response = agent.run("Analyze the image at path/to/image.png and identify all objects")
```

## Supported Input Types

```python
# File path
result = engine.annotate("image.png")

# URL
result = engine.annotate("https://example.com/image.png")

# NumPy array
import cv2
img = cv2.imread("image.png")
result = engine.annotate(img)

# PIL Image
from PIL import Image
img = Image.open("image.png")
result = engine.annotate(img)

# Batch processing
results = engine.annotate(["img1.png", "img2.png", "img3.png"])
```

## Output Formats

```python
# Annotated overlay image
overlay = engine.annotate("image.png", output="overlay")
cv2.imwrite("annotated.png", overlay)

# JSON data
json_data = engine.annotate("image.png", output="json")

# Formal report
report = engine.annotate("image.png", output="report")

# Complete result object
result = engine.annotate("image.png", output="all")
```

## Next Steps

- [Engine](engine.md) - Learn about AnnotationEngine
- [Geometric Primitives](geometric-primitives.md) - Understand primitive extraction
