# Templates Guide

Comprehensive guide to using templates for creating specialized agents.

## Policy Loop Mixin

Policy loop mixin provides policy engine functionality for agents.

### Mathematical Foundation

Policy loop implements Model Predictive Control (MPC):

$$\min_{\mathbf{u}_{t:t+H-1}} \sum_{k=0}^{H-1} \ell(\mathbf{x}_{t+k}, \mathbf{u}_{t+k}) + V(\mathbf{x}_{t+H})$$

Subject to:

$$\mathbf{x}_{t+k+1} = f(\mathbf{x}_{t+k}, \mathbf{u}_{t+k})$$
$$\mathbf{g}(\mathbf{x}_{t+k}, \mathbf{u}_{t+k}) \leq 0$$

### Usage

```python
from templates.base_specialized_agent import BaseSpecializedAgent
from templates.policy_loop import PolicyLoopMixin

class PolicyAgent(BaseSpecializedAgent, PolicyLoopMixin):
    def __init__(self):
        super().__init__()
```

## Graph Management Mixin

Graph management mixin provides causal graph management functionality.

### Usage

```python
from templates.graph_management import GraphManagementMixin

class GraphAgent(BaseSpecializedAgent, GraphManagementMixin):
    def __init__(self):
        super().__init__()
```

## LLM Integration Mixin

LLM integration mixin provides LLM-based reasoning capabilities.

### Usage

```python
from templates.llm_integration import LLMIntegrationMixin

class LLMAgent(BaseSpecializedAgent, LLMIntegrationMixin):
    def __init__(self):
        super().__init__()
```

## Module Registry

Module registry for managing and discovering modules.

### Usage

```python
from templates.module_registry import ModuleRegistry

registry = ModuleRegistry()
registry.register(module)
```

## Next Steps

- [Base Agent](base-agent.md) - Base template
- [Feature Mixins](feature-mixins.md) - Reusable features
