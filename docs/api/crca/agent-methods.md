# CRCAAgent Class & Core Methods

Main causal reasoning agent class and core methods.

## Class Definition

```python
class CRCAAgent(Agent):
    """Causal Reasoning with Counterfactual Analysis Agent."""
```

## Initialization

```python
CRCAAgent(
    model_name: str = "gpt-4o-mini",
    agent_name: str = "cr-ca-lite-agent",
    max_loops: Optional[Union[int, str]] = 3,
    variables: Optional[List[str]] = None,
    causal_edges: Optional[List[Tuple[str, str]] = None,
    **kwargs
)
```

## Core Methods

### run()

Execute a causal reasoning task.

```python
def run(self, task: str) -> str:
    """Execute causal reasoning task."""
```

### add_causal_relationship()

Add a causal relationship to the graph.

```python
def add_causal_relationship(
    self,
    parent: str,
    child: str,
    strength: float = 0.0
) -> None:
    """Add causal relationship."""
```

## Mathematical Foundation

The agent implements operations on SCMs:

$$V_i = f_i(Pa(V_i), U_i)$$

Methods implement:
- **run**: $Result = f(Task, SCM)$
- **add_causal_relationship**: $E \leftarrow E \cup \{(X, Y)\}$

## Next Steps

- [Operations](operations.md) - Advanced operations
