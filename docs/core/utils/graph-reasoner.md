# Graph Reasoner

The graph reasoner module provides utilities for reasoning about causal graphs.

## Overview

Graph reasoner utilities help analyze causal graphs, find paths, identify confounders, and compute causal effects.

## Mathematical Foundation

Graph reasoning uses graph algorithms to:

1. **Path Finding**: Find all paths between variables
2. **Confounder Identification**: Identify variables that confound relationships
3. **Adjustment Sets**: Find valid adjustment sets for causal identification

## Usage

```python
from utils.graph_reasoner import GraphReasoner

reasoner = GraphReasoner(graph)
paths = reasoner.find_paths("X", "Y")
confounders = reasoner.identify_confounders("X", "Y")
```

## Next Steps

- [Batch Processor](batch-processor.md) - Batch processing
