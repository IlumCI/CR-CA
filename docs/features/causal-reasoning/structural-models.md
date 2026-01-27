# Structural Causal Models (SCMs)

Structural Causal Models provide the mathematical foundation for causal reasoning.

## Overview

SCMs represent causal relationships through structural equations that define how variables depend on their causes.

## Mathematical Foundation

### Definition

An SCM is a triple:

$$(U, V, F)$$

Where:
- $U = \{U_1, \ldots, U_n\}$: Exogenous (unobserved) variables
- $V = \{V_1, \ldots, V_m\}$: Endogenous (observed) variables
- $F = \{f_1, \ldots, f_m\}$: Structural equations

### Structural Equations

Each endogenous variable has a structural equation:

$$V_i = f_i(Pa(V_i), U_i)$$

Where $Pa(V_i)$ are the parents of $V_i$ in the causal graph.

### Example

For education and income:

$$Education = f_E(U_E)$$
$$Experience = f_{Exp}(Education, U_{Exp})$$
$$Income = f_I(Education, Experience, U_I)$$

## Usage

```python
from CRCA import CRCAAgent

agent = CRCAAgent(model_name="gpt-4o-mini")

# Build SCM
agent.add_causal_relationship("education", "income", strength=0.6)
agent.add_causal_relationship("experience", "income", strength=0.4)
```

## Next Steps

- [Do-Calculus](do-calculus.md) - Do-calculus implementation
