# Data Flow

Data flow through the CR-CA system.

## Causal Analysis Flow

```mermaid
graph LR
    Input[Natural Language Input] --> Extract[Variable Extraction]
    Extract --> Graph[Causal Graph Construction]
    Graph --> SCM[SCM Fitting]
    SCM --> Inference[Causal Inference]
    Inference --> Counterfactual[Counterfactual Generation]
    Counterfactual --> Output[Results]
```

## Mathematical Flow

1. **Input**: Natural language task $T$
2. **Extraction**: Variables $V = \{V_1, \ldots, V_n\}$
3. **Graph**: Causal DAG $G = (V, E)$
4. **SCM**: Structural equations $F = \{f_1, \ldots, f_n\}$
5. **Inference**: Causal effects $E[Y | do(X=x)]$
6. **Counterfactuals**: $P(Y_{x'} | X=x, Y=y)$

## Next Steps

- [Modular Structure](modular-structure.md) - Modular architecture
- [Hybrid Agent](hybrid-agent/overview.md) - Hybrid agent architecture
