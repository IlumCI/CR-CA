# Advanced Causal Reasoning Topics

Advanced topics in causal reasoning including discovery, interventions, and confounders.

## Causal Discovery

Causal discovery learns causal structure from observational data.

### Mathematical Foundation

Causal discovery uses various methods:

**Constraint-Based Methods**: Test conditional independencies:

$$X \perp Y | Z \implies \text{No direct edge } X \to Y$$

**Score-Based Methods**: Search for graph $G$ that maximizes:

$$Score(G, D) = \sum_{i=1}^n \log P(V_i | Pa(V_i), \theta_i) - \text{Complexity}(G)$$

**Functional Causal Models**: Assume functional relationships:

$$Y = f(X, U_Y)$$

Where $U_Y$ is independent of $X$.

### Usage

Causal discovery is integrated into automatic extraction:

```python
agent = CRCAAgent(enable_automatic_extraction=True)
result = agent.run("Analyze relationships in data")
```

## Causal Interventions

Interventions represent actions that change the value of a variable.

### Mathematical Foundation

**Intervention vs. Observation**:

**Observation**: $P(Y | X=x)$ - Conditional probability

**Intervention**: $P(Y | do(X=x))$ - Causal effect

The key difference:

$$P(Y | X=x) = \sum_z P(Y | X=x, Z=z) P(Z=z | X=x)$$
$$P(Y | do(X=x)) = \sum_z P(Y | X=x, Z=z) P(Z=z)$$

**Average Causal Effect (ACE)**:

$$ACE = E[Y | do(X=1)] - E[Y | do(X=0)]$$

### Usage

```python
effect = agent.compute_causal_effect(
    treatment="X",
    outcome="Y",
    value_treated=1,
    value_control=0
)
```

## Confounder Identification

Confounders are variables that cause both treatment and outcome.

### Mathematical Foundation

**Confounder Definition**: $Z$ is a confounder for $X \to Y$ if:

$$Z \to X \text{ and } Z \to Y$$

This creates a backdoor path: $X \leftarrow Z \to Y$.

**Backdoor Criterion**: To identify $E[Y | do(X=x)]$, we need to block all backdoor paths. A set $Z$ satisfies the backdoor criterion if:

1. $Z$ blocks all backdoor paths from $X$ to $Y$
2. $Z$ contains no descendants of $X$

Then:

$$E[Y | do(X=x)] = \sum_z E[Y | X=x, Z=z] P(Z=z)$$

### Usage

```python
from utils.graph_reasoner import GraphReasoner

reasoner = GraphReasoner(graph)
confounders = reasoner.identify_confounders("X", "Y")
adjustment_set = reasoner.find_adjustment_set("X", "Y")
```

## Next Steps

- [Overview](overview.md) - Return to overview
