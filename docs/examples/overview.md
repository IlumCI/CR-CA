# Examples Overview

This section provides code examples and tutorials for using CR-CA.

## Example Categories

- **[Basic Usage](basic-usage.md)**: Basic agent examples
- **[General Agent](general-agent/quickstart.md)**: General agent examples
- **[Hybrid Agent](hybrid-agent/auto-extraction.md)**: Hybrid agent examples
- **[Image Annotation](image-annotation/quickstart.md)**: Image annotation examples
- **[Trading](trading/crca-q-basic.md)**: Trading examples
- **[Integration](integration/crca-sd.md)**: Integration examples

## Quick Examples

### Basic Agent

```python
from CRCA import CRCAAgent

agent = CRCAAgent(model_name="gpt-4o-mini")
result = agent.run("Analyze X -> Y")
```

### Image Annotation

```python
from image_annotation import ImageAnnotationEngine

engine = ImageAnnotationEngine()
result = engine.annotate("image.png")
```

## Next Steps

- [Basic Usage](basic-usage.md) - Start with basics
- [General Agent](general-agent/quickstart.md) - General agent examples
