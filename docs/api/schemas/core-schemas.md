# Core Schemas API

API for core data schemas grouped together.

## Conversation Schemas

Schemas for conversation management.

### Classes

- `ConversationState`: Current conversation state
- `Message`: Individual message in conversation
- `Context`: Conversation context

### Usage

```python
from schemas.conversation import ConversationState, Message

state = ConversationState(
    messages=[Message(role="user", content="Hello")]
)
```

## Hybrid Agent Schemas

Schemas for hybrid agent architecture.

### Classes

- `HybridAgentState`: Hybrid agent state
- `ReasoningStep`: Individual reasoning step
- `ConsistencyCheck`: Consistency check result

### Usage

```python
from schemas.hybrid import HybridAgentState

state = HybridAgentState(
    reasoning_steps=[],
    consistency_checks=[]
)
```

## Reasoning Schemas

Schemas for causal reasoning operations and results.

### Classes

- `CausalInferenceResult`: Causal inference result
- `CounterfactualScenario`: Counterfactual scenario
- `CausalEffect`: Causal effect estimate

### Mathematical Foundation

Causal effect:

$$\tau = E[Y | do(X=1)] - E[Y | do(X=0)]$$

With confidence interval:

$$CI_{1-\alpha} = [\tau - z_{\alpha/2} \cdot SE, \tau + z_{\alpha/2} \cdot SE]$$

### Usage

```python
from schemas.reasoning import CausalInferenceResult

result = CausalInferenceResult(
    treatment="X",
    outcome="Y",
    effect=0.5,
    confidence=0.9
)
```

## Next Steps

- [Annotation](annotation.md) - Image annotation schemas
- [Policy](policy.md) - Policy engine schemas
