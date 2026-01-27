# Graph Reasoner API

API for graph reasoning utilities.

## Classes

### GraphReasoner

Provides utilities for reasoning about causal graphs.

```python
class GraphReasoner:
    def find_paths(self, source: str, target: str) -> List[List[str]]:
        """Find all paths from source to target."""
    
    def identify_confounders(self, X: str, Y: str) -> List[str]:
        """Identify confounders for X -> Y."""
```

## Mathematical Foundation

Path finding uses graph algorithms:

$$Paths(X, Y) = \{p : X \to^* Y\}$$

Confounder identification:

$$Z \in Confounders(X, Y) \iff Z \to X \text{ and } Z \to Y$$

## Next Steps

- [Batch Processor](batch-processor.md) - Batch processor API
