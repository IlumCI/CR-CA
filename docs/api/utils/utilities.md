# Utility Functions API

API for smaller utility modules grouped together.

## Formatter

Text formatting and output utilities.

### Functions

- `format_causal_result()`: Format causal analysis results
- `format_graph()`: Format causal graph for display
- `format_counterfactuals()`: Format counterfactual scenarios

### Usage

```python
from utils.formatter import format_causal_result

result = agent.run("Analyze X -> Y")
formatted = format_causal_result(result)
```

## Batch Processor

Batch processing utilities for multiple tasks.

### Usage

```python
from utils.batch_processor import BatchProcessor

processor = BatchProcessor(agent)
results = processor.process_batch(tasks)
```

## Rate Limiter

Rate limiting for API calls.

### Usage

```python
from utils.rate_limiter import RateLimiter

limiter = RateLimiter(max_calls=100, period=60)
limiter.wait_if_needed()
```

## Agent Discovery

Agent discovery and registration.

### Usage

```python
from utils.agent_discovery import AgentDiscovery

discovery = AgentDiscovery()
agents = discovery.discover_agents()
```

## Tool Discovery

Tool discovery and registration.

### Usage

```python
from utils.tool_discovery import ToolDiscovery

discovery = ToolDiscovery()
tools = discovery.discover_tools()
```

## Edit Distance

Edit distance algorithms for text correction.

### Mathematical Foundation

Levenshtein distance:

$$D_{lev}(s_1, s_2) = \min\{Insertions, Deletions, Substitutions\}$$

### Usage

```python
from utils.edit_distance import levenshtein_distance

distance = levenshtein_distance("hello", "hallo")
```

## Next Steps

- [Router](router.md) - Router API
- [Conversation](conversation.md) - Conversation API
