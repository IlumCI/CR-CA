# Image Annotation Features

Comprehensive guide to image annotation features.

## Geometric Primitive Extraction

Extracts geometric primitives from images using computer vision algorithms.

### Mathematical Foundation

**Circle Detection**: Circles are detected using Hough circle transform:

$$(x - x_c)^2 + (y - y_c)^2 = r^2$$

Where $(x_c, y_c)$ is the center and $r$ is the radius.

**Line Detection**: Lines are detected using Hough line transform:

$$\rho = x \cos \theta + y \sin \theta$$

Where $\rho$ is the distance from origin and $\theta$ is the angle.

**Contour Detection**: Contours are extracted using:

$$C = \text{findContours}(\nabla I)$$

Where $\nabla I$ is the image gradient.

### Usage

```python
result = engine.annotate("image.png", output="all")
entities = result.annotation_graph.entities
circles = [e for e in entities if e.primitive_type == "circle"]
```

## Semantic Labeling

Assigns meaningful labels to detected geometric primitives.

### Mathematical Foundation

For entity $E_i$ with geometric properties $G_i$, the semantic label $L_i$ is:

$$L_i = \arg\max_{l} P(l | E_i, G_i, I)$$

Where $I$ is the image context, and the probability is estimated using LLM:

$$P(l | E_i, G_i, I) \approx \text{LLM}(E_i, G_i, I)$$

### Usage

```python
labels = result.annotation_graph.labels
for label in labels:
    print(f"Entity {label.entity_id}: {label.label}")
```

## Query Capabilities

Natural language queries about images.

### Query Types

- **Object Detection**: "find all circles", "identify military bases"
- **Measurement**: "measure distance", "calculate area"
- **Counting**: "how many objects", "count lines"
- **Spatial**: "what is near X", "find objects between A and B"

### Mathematical Foundation

Queries are processed by:

1. Parsing the natural language query $Q$
2. Identifying relevant entities $E_Q \subseteq E$
3. Computing requested measurements $M_Q$
4. Generating answer $A = f(Q, E_Q, M_Q)$

### Usage

```python
result = engine.query("image.png", "find all circles")
print(result["answer"])
```

## Temporal Tracking

Tracks entities across video frames using Kalman filters.

### Mathematical Foundation

Temporal tracking uses Kalman filters for state estimation:

**State Model**:
$$\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \mathbf{B} \mathbf{u}_t + \mathbf{w}_t$$

**Observation Model**:
$$\mathbf{z}_t = \mathbf{H} \mathbf{x}_t + \mathbf{v}_t$$

**Kalman Update**:
$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t (\mathbf{z}_t - \mathbf{H} \hat{\mathbf{x}}_{t|t-1})$$

### Usage

```python
engine = ImageAnnotationEngine(enable_temporal_tracking=True)
for frame_id, image_path in enumerate(frame_paths):
    result = engine.annotate(image_path, frame_id=frame_id)
```

## Next Steps

- [Integration](integration.md) - Integration with agents
