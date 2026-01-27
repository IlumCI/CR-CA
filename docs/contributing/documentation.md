# Documentation Contribution Guide

Guide for contributing to CR-CA documentation.

## Documentation Structure

Documentation is organized in `docs/` with:
- Getting started guides
- Core module documentation
- Feature documentation
- Examples
- API reference

## Writing Documentation

### Include Mathematical Formulations

All concepts should be explainable mathematically:

```markdown
The causal effect is:

$$E[Y | do(X=x)] = \sum_{z} E[Y | X=x, Z=z] P(Z=z)$$
```

### Code Examples

Include working code examples:

```python
from CRCA import CRCAAgent

agent = CRCAAgent(model_name="gpt-4o-mini")
result = agent.run("Analyze X -> Y")
```

### No Emojis

Do not use emojis in documentation. Use text-based symbols if needed.

## Next Steps

- [Testing](testing.md) - Testing guidelines
