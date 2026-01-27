# Image Annotation Usage Guide

This guide explains how to use the image annotation system in CR-CA. The image annotation engine provides automated image analysis with geometric primitive extraction, semantic labeling, and query capabilities.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Direct Usage with ImageAnnotationEngine](#direct-usage-with-imageannotationengine)
3. [Integration with CRCAAgent](#integration-with-crcaagent)
4. [Integration with GeneralAgent](#integration-with-generalagent)
5. [Main Methods](#main-methods)
6. [Output Formats](#output-formats)
7. [Advanced Features](#advanced-features)
8. [Examples](#examples)

## Quick Start

### Basic Annotation

```python
from image_annotation import ImageAnnotationEngine

# Initialize the engine
engine = ImageAnnotationEngine()

# Annotate an image (file path, URL, numpy array, or PIL Image)
result = engine.annotate("path/to/image.png", output="all")

# Access the results
print(f"Found {len(result.annotation_graph.entities)} entities")
print(f"Generated {len(result.annotation_graph.labels)} labels")
print(result.formal_report)
```

### With CRCAAgent

```python
from CRCA import CRCAAgent

# Create agent with image annotation enabled
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    use_image_annotation=True
)

# The agent now has access to annotate_image and query_image tools
response = agent.run("Analyze the image at path/to/image.png and identify all objects")
```

## Direct Usage with ImageAnnotationEngine

### Initialization

```python
from image_annotation import ImageAnnotationEngine
from image_annotation.annotation_engine import AnnotationConfig

# Basic initialization with defaults
engine = ImageAnnotationEngine()

# With custom configuration
config = AnnotationConfig(
    gpt_model="gpt-4o-mini",
    enable_temporal_tracking=True,
    auto_detect_type=True,
    output_format="all"
)
engine = ImageAnnotationEngine(config=config)

# With individual parameters
engine = ImageAnnotationEngine(
    gpt_model="gpt-4o-mini",
    enable_temporal_tracking=False,
    use_crca_tools=True,
    cache_enabled=True,
    auto_retry=True
)
```

### Supported Input Types

The `annotate()` method accepts multiple input types:

```python
# File path (string or Path)
result = engine.annotate("image.png")
result = engine.annotate(Path("image.png"))

# URL
result = engine.annotate("https://example.com/image.png")

# NumPy array
import numpy as np
import cv2
img = cv2.imread("image.png")
result = engine.annotate(img)

# PIL Image
from PIL import Image
img = Image.open("image.png")
result = engine.annotate(img)

# Batch processing (list of any above)
results = engine.annotate(["img1.png", "img2.png", "img3.png"])
```

## Integration with CRCAAgent

### Basic Setup

```python
from CRCA import CRCAAgent

agent = CRCAAgent(
    model_name="gpt-4o-mini",
    use_image_annotation=True,  # Enable image annotation tools
    use_crca_tools=True
)
```

### Available Tools

When `use_image_annotation=True`, the agent automatically gets two tools:

1. **`annotate_image`**: Annotate an image and get structured results
2. **`query_image`**: Query an annotated image with natural language

### Using the Tools

```python
# The agent can use these tools automatically in conversations
task = """
Analyze the image at path/to/circuit.png:
1. Identify all components
2. Measure distances between components
3. Find the largest component
"""

response = agent.run(task)
```

### Manual Tool Usage

```python
# Get the tool handlers directly
annotate_tool = agent.tools_list_dictionary.get("annotate_image")
query_tool = agent.tools_list_dictionary.get("query_image")

# Use annotate_image
result = annotate_tool(
    image_path="path/to/image.png",
    output_format="all",
    frame_id=None
)

# Use query_image
answer = query_tool(
    image_path="path/to/image.png",
    query="What is the largest object in this image?",
    frame_id=None
)
```

## Integration with GeneralAgent

### Basic Setup

```python
from branches.general_agent.general_agent import GeneralAgent

agent = GeneralAgent(
    model_name="gpt-4o-mini",
    enable_multimodal=True  # Enables image annotation tools
)
```

The GeneralAgent automatically includes image annotation tools when `enable_multimodal=True`.

## Main Methods

### 1. `annotate()` - Main Annotation Method

```python
def annotate(
    self,
    input: Union[str, np.ndarray, Image.Image, List, Path],
    frame_id: Optional[int] = None,
    output: Optional[str] = None
) -> Union[AnnotationResult, np.ndarray, Dict[str, Any], str, List]:
    """
    Annotate image(s) with full automation.
    
    Args:
        input: Image input (file path, URL, numpy array, PIL Image, or list for batch)
        frame_id: Optional frame ID for temporal tracking
        output: Output format - "overlay", "json", "report", or "all"
    
    Returns:
        Depends on output format:
        - "overlay": numpy array (annotated image)
        - "json": dict (JSON data)
        - "report": str (formal report)
        - "all": AnnotationResult object
        - If input is list: List of above
    """
```

**Examples:**

```python
# Get annotated overlay image
overlay = engine.annotate("image.png", output="overlay")
cv2.imwrite("annotated.png", overlay)

# Get JSON data
json_data = engine.annotate("image.png", output="json")
print(json_data["entities"])

# Get formal report
report = engine.annotate("image.png", output="report")
print(report)

# Get complete result
result = engine.annotate("image.png", output="all")
print(result.annotation_graph.entities)
print(result.formal_report)
print(result.processing_time)
```

### 2. `query()` - Natural Language Querying

```python
def query(
    self,
    input: Union[str, np.ndarray, Image.Image, Path],
    query: str,
    frame_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Query an image with natural language.
    
    Args:
        input: Image input (file path, URL, numpy array, or PIL Image)
        query: Natural language query (e.g., "find all circles", "measure distance")
        frame_id: Optional frame ID for temporal tracking
    
    Returns:
        Dictionary with:
        - "answer": Natural language answer
        - "entities": List of relevant entities
        - "measurements": Dict of measurements (if requested)
        - "graph": AnnotationGraph (if needed)
    """
```

**Examples:**

```python
# Find specific objects
result = engine.query("image.png", "find all circles")
print(result["answer"])
print(result["entities"])

# Measure distances
result = engine.query("image.png", "measure the distance from the border to the largest city")
print(result["measurements"])

# Identify objects
result = engine.query("image.png", "identify all military installations")
print(result["answer"])

# Count objects
result = engine.query("image.png", "how many lines are in this image?")
print(result["answer"])
```

## Output Formats

### AnnotationResult Object

When using `output="all"`, you get a complete `AnnotationResult` object:

```python
result = engine.annotate("image.png", output="all")

# Access annotation graph
graph = result.annotation_graph
entities = graph.entities  # List[PrimitiveEntity]
labels = graph.labels      # List[SemanticLabel]
relations = graph.relations  # List[Relation]
contradictions = graph.contradictions  # List[Contradiction]

# Access overlay image (as bytes, need to decode)
if result.overlay_image:
    overlay = cv2.imdecode(
        np.frombuffer(result.overlay_image, np.uint8),
        cv2.IMREAD_COLOR
    )

# Access formal report
report = result.formal_report

# Access JSON output
json_data = result.json_output

# Access metadata
processing_time = result.processing_time
instability_detected = result.instability_detected
```

### AnnotationGraph Structure

```python
graph = result.annotation_graph

# Get entity by ID
entity = graph.get_entity_by_id("entity-id-123")

# Get labels for an entity
labels = graph.get_labels_for_entity("entity-id-123")

# Get relations for an entity
relations = graph.get_relations_for_entity("entity-id-123")
```

## Advanced Features

### Temporal Tracking

For video sequences or time-series images:

```python
engine = ImageAnnotationEngine(enable_temporal_tracking=True)

# Process frame sequence
for frame_id, image_path in enumerate(frame_paths):
    result = engine.annotate(image_path, frame_id=frame_id)
    # Engine tracks entities across frames using Kalman filters
```

### Batch Processing

```python
# Process multiple images in parallel
image_paths = ["img1.png", "img2.png", "img3.png"]
results = engine.annotate(image_paths, output="all")

# Results is a list of AnnotationResult objects
for i, result in enumerate(results):
    print(f"Image {i}: {len(result.annotation_graph.entities)} entities")
```

### Custom Configuration

```python
from image_annotation.annotation_engine import AnnotationConfig

config = AnnotationConfig(
    gpt_model="gpt-4o-mini",
    enable_temporal_tracking=True,
    auto_detect_type=True,
    auto_tune_params=True,
    cache_enabled=True,
    auto_retry=True,
    max_retries=3,
    output_format="all",
    parallel_workers=4,
    show_progress=True
)

engine = ImageAnnotationEngine(config=config)
```

### Caching

The engine supports automatic caching to avoid re-processing the same images:

```python
engine = ImageAnnotationEngine(cache_enabled=True)
# First call processes the image
result1 = engine.annotate("image.png")
# Second call uses cache (much faster)
result2 = engine.annotate("image.png")
```

## Examples

### Example 1: Circuit Diagram Analysis

```python
from image_annotation import ImageAnnotationEngine

engine = ImageAnnotationEngine()

# Annotate circuit diagram
result = engine.annotate("circuit.png", output="all")

# Find all components
components = [e for e in result.annotation_graph.entities 
              if e.primitive_type in ["circle", "contour"]]

print(f"Found {len(components)} components")

# Get labels
for label in result.annotation_graph.labels:
    print(f"Entity {label.entity_id}: {label.label} (uncertainty: {label.uncertainty})")
```

### Example 2: Tactical Map Analysis

```python
engine = ImageAnnotationEngine()

# Query for specific information
result = engine.query(
    "tactical_map.png",
    "identify all military bases and measure their sizes"
)

print(result["answer"])
for entity in result["entities"]:
    if entity.primitive_type == "circle":
        radius = entity.metadata.get("radius", 0)
        area = 3.14159 * radius * radius
        print(f"Base at {entity.pixel_coords[0]}: area = {area:.2f} pixels²")
```

### Example 3: Integration with CR-CA for Strategic Analysis

```python
from CRCA import CRCAAgent

# Create agent with image annotation
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    use_image_annotation=True,
    use_crca_tools=True
)

# Complex task combining image analysis and causal reasoning
task = """
Analyze the tactical map at path/to/map.png:
1. Use query_image to identify all military installations
2. Use query_image to measure distances between key targets
3. Extract causal variables:
   - Distance from border to capital
   - Number of military bases
   - Road network connectivity
4. Perform causal analysis:
   - What factors affect invasion success?
   - What are the critical chokepoints?
5. Provide strategic recommendations
"""

response = agent.run(task)
print(response)
```

### Example 4: Batch Processing with Progress

```python
import os
from pathlib import Path

engine = ImageAnnotationEngine(
    cache_enabled=True,
    show_progress=True  # Requires tqdm
)

# Get all images in directory
image_dir = Path("images")
image_paths = list(image_dir.glob("*.png"))

# Process all images
results = engine.annotate(image_paths, output="all")

# Analyze results
total_entities = sum(len(r.annotation_graph.entities) for r in results)
print(f"Total entities across all images: {total_entities}")
```

### Example 5: Custom Query Processing

```python
engine = ImageAnnotationEngine()

# Multiple queries on same image
queries = [
    "find all circles",
    "identify the largest structure",
    "measure distances between all circles",
    "count the number of lines"
]

for query_text in queries:
    result = engine.query("image.png", query_text)
    print(f"\nQuery: {query_text}")
    print(f"Answer: {result['answer']}")
    if result.get("measurements"):
        print(f"Measurements: {result['measurements']}")
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure all dependencies are installed
   ```bash
   pip install opencv-python numpy pillow loguru rustworkx
   ```

2. **GPT API Error**: Check your OpenAI API key
   ```python
   import os
   os.environ["OPENAI_API_KEY"] = "your-key-here"
   ```

3. **Memory Issues with Large Images**: The engine automatically downscales, but you can pre-process:
   ```python
   import cv2
   img = cv2.imread("large_image.png")
   img = cv2.resize(img, (1920, 1080))  # Resize before annotation
   result = engine.annotate(img)
   ```

4. **Slow Processing**: Enable caching and use batch processing with parallel workers
   ```python
   engine = ImageAnnotationEngine(
       cache_enabled=True,
       parallel_workers=4  # Adjust based on CPU cores
   )
   ```

## API Reference

For complete API documentation, see:
- `image_annotation/annotation_engine.py` - Main engine class
- `schemas/annotation.py` - Data models
- `tests/test_image_annotation_*.py` - Test examples
