# Integration with Agents

Image annotation integrates seamlessly with CRCAAgent and GeneralAgent.

## CRCAAgent Integration

```python
from CRCA import CRCAAgent

agent = CRCAAgent(
    model_name="gpt-4o-mini",
    use_image_annotation=True
)

# Agent automatically has annotate_image and query_image tools
response = agent.run("Analyze the image at path/to/image.png")
```

## Available Tools

When `use_image_annotation=True`, the agent gets:

1. **`annotate_image`**: Annotate an image and get structured results
2. **`query_image`**: Query an annotated image with natural language

## Manual Tool Usage

```python
annotate_tool = agent.tools_list_dictionary.get("annotate_image")
query_tool = agent.tools_list_dictionary.get("query_image")

# Use annotate_image
result = annotate_tool(
    image_path="path/to/image.png",
    output_format="all"
)

# Use query_image
answer = query_tool(
    image_path="path/to/image.png",
    query="What is the largest object?"
)
```

## GeneralAgent Integration

```python
from branches.general_agent.general_agent import GeneralAgent

agent = GeneralAgent(
    model_name="gpt-4o-mini",
    enable_multimodal=True  # Enables image annotation
)
```

## Combined Analysis

Combine image analysis with causal reasoning:

```python
task = """
Analyze the tactical map at path/to/map.png:
1. Identify all military installations
2. Measure distances between key targets
3. Extract causal variables from the map
4. Perform causal analysis
"""

response = agent.run(task)
```

## Next Steps

- [Overview](overview.md) - Return to overview
- [Policy Engine](../policy-engine/overview.md) - Policy engine features
