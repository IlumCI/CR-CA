# Async Operations

CRCAAgent supports asynchronous operations for concurrent causal analysis.

## Overview

Async operations allow non-blocking causal inference, enabling concurrent processing of multiple tasks.

## Basic Usage

```python
import asyncio
from CRCA import CRCAAgent

agent = CRCAAgent(model_name="gpt-4o-mini")

async def analyze():
    result = await agent.run_async("Analyze X -> Y")
    return result

# Run async
result = asyncio.run(analyze())
```

## Concurrent Analysis

Process multiple tasks concurrently:

```python
async def analyze_multiple():
    tasks = [
        agent.run_async("Analyze education -> income"),
        agent.run_async("Analyze experience -> income"),
        agent.run_async("Analyze location -> income")
    ]
    
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(analyze_multiple())
```

## Async Batch Processing

```python
async def batch_async():
    scenarios = [...]
    results = await agent.batch_predict_async(
        scenarios=scenarios,
        target="income"
    )
    return results
```

## Performance Benefits

Async operations improve throughput:

- **Sequential**: Total time = $\sum_{i=1}^n t_i$
- **Async**: Total time = $\max_{i=1}^n t_i$ (for I/O bound operations)

## Next Steps

- [Batch Prediction](batch-prediction.md) - Batch processing
- [Optimization](optimization.md) - Optimization methods
