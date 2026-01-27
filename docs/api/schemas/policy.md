# Policy Engine Schemas

Schemas for policy engine system.

## Classes

### DoctrineV1

Policy doctrine definition.

```python
class DoctrineV1(BaseModel):
    version: str
    objectives: List[str]
    constraints: List[str]
```

## Mathematical Foundation

Doctrine defines optimization problem:

$$\min_{\mathbf{u}} \sum_{i} w_i J_i(\mathbf{u})$$

Subject to:

$$g_j(\mathbf{u}) \leq 0, \quad j = 1, \ldots, m$$

## Next Steps

- [Branches API](../branches/overview.md) - Branches API
