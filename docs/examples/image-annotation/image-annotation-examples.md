# Image Annotation Examples

Comprehensive examples for image annotation usage.

## Quickstart

Basic annotation:

```python
from image_annotation import ImageAnnotationEngine

engine = ImageAnnotationEngine()
result = engine.annotate("image.png", output="all")

print(f"Found {len(result.annotation_graph.entities)} entities")
print(result.formal_report)
```

## With CRCAAgent

Integration with CRCAAgent:

```python
from CRCA import CRCAAgent

agent = CRCAAgent(
    model_name="gpt-4o-mini",
    use_image_annotation=True
)

response = agent.run("Analyze the image at path/to/image.png")
```

## Operational Examples

Batch processing and production scenarios:

```python
engine = ImageAnnotationEngine(
    cache_enabled=True,
    parallel_workers=4
)

# Batch process images
results = engine.annotate(["img1.png", "img2.png", "img3.png"])

# Analyze results
for result in results:
    print(f"Entities: {len(result.annotation_graph.entities)}")
```

## Next Steps

- [Trading Examples](../trading/trading-examples.md) - Trading examples
