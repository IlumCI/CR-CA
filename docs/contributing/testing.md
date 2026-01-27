# Testing Guidelines

Guidelines for writing and running tests.

## Test Structure

Tests are located in `tests/` directory.

## Writing Tests

```python
import pytest
from CRCA import CRCAAgent

def test_basic_agent():
    agent = CRCAAgent(model_name="gpt-4o-mini")
    result = agent.run("Test task")
    assert result is not None
```

## Running Tests

```bash
pytest tests/
```

## Next Steps

- [CI/CD Setup](../../getting-started/configuration.md) - CI/CD configuration
