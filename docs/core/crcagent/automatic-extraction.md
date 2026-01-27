# Automatic Variable Extraction

CRCAAgent can automatically extract variables and causal relationships from natural language tasks.

## Overview

The automatic extraction system uses LLM-based analysis to identify:
- Variables of interest
- Causal relationships between variables
- Confounders and mediators
- Structural equations

## Mathematical Foundation

Given a natural language task $T$, the extraction process identifies:

1. **Variable Set**: $V = \{V_1, V_2, \ldots, V_n\}$
2. **Causal Graph**: $G = (V, E)$ where $E$ are causal edges
3. **Structural Equations**: $F = \{f_1, f_2, \ldots, f_n\}$

For each variable $V_i$, the system extracts:

$$V_i = f_i(Pa(V_i), U_i)$$

Where $Pa(V_i)$ are the parents identified from the text.

## Usage

### Basic Extraction

```python
from CRCA import CRCAAgent

agent = CRCAAgent(
    model_name="gpt-4o-mini",
    enable_automatic_extraction=True
)

task = """
Analyze how increasing the minimum wage affects employment rates.
Consider inflation, business costs, and consumer spending.
"""

result = agent.run(task)
# Variables and relationships are automatically extracted
```

### View Extracted Variables

```python
# After running a task
graph = agent.causal_graph
variables = graph.get_variables()
print("Extracted variables:", variables)
```

### View Causal Relationships

```python
relationships = graph.get_relationships()
for parent, child, strength in relationships:
    print(f"{parent} -> {child}: {strength:.2f}")
```

## Extraction Process

The extraction process follows these steps:

1. **Text Analysis**: Parse natural language to identify entities
2. **Variable Identification**: Extract variable names and types
3. **Relationship Detection**: Identify causal relationships
4. **Graph Construction**: Build causal DAG
5. **Strength Estimation**: Estimate relationship strengths

## Example

```python
task = """
Study the effect of education on income.
Education affects experience, which in turn affects income.
Geographic location also influences income.
"""

result = agent.run(task)

# Extracted structure:
# education -> experience -> income
# location -> income
```

## Configuration

Control extraction behavior:

```python
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    enable_automatic_extraction=True,
    extraction_confidence_threshold=0.5,
    max_extracted_variables=20
)
```

## Next Steps

- [Causal Graph](causal-graph.md) - Work with extracted graphs
- [Dual-Mode Operation](dual-mode-operation.md) - Combine with deterministic mode
