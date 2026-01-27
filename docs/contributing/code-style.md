# Code Style Guide

Follow these guidelines when contributing code to CR-CA.

## Type Annotations

Always use type annotations:

```python
def calculate_average(numbers: List[float]) -> float:
    """Calculate average."""
    return sum(numbers) / len(numbers)
```

## Docstrings

Use Google-style docstrings:

```python
def process_data(data: List[int]) -> int:
    """
    Process data and return result.
    
    Args:
        data: List of integers to process.
    
    Returns:
        Processed result as integer.
    
    Raises:
        ValueError: If data is empty.
    """
    if not data:
        raise ValueError("Data cannot be empty")
    return sum(data)
```

## Logging

Use Loguru for logging:

```python
from loguru import logger

logger.info("Processing data")
logger.error(f"Error: {error}")
```

## Mathematical Documentation

Include mathematical formulations for algorithms:

```python
def compute_causal_effect(treatment: str, outcome: str) -> float:
    """
    Compute causal effect E[Y | do(X=x)].
    
    Uses the formula:
    $$E[Y | do(X=x)] = \sum_{z} E[Y | X=x, Z=z] P(Z=z)$$
    
    Where Z are confounders.
    """
    # Implementation
```

## Next Steps

- [Documentation](documentation.md) - Documentation guidelines
- [Testing](testing.md) - Testing guidelines
