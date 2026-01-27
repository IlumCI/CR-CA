# Causal vs. Correlational Trading

CRCA-Q's fundamental philosophy is based on causal reasoning rather than correlation-based patterns.

## The Fundamental Problem

Traditional quantitative trading systems rely heavily on **correlational patterns**: "When X happens, Y tends to follow." This approach has critical limitations:

1. **Regime Dependency**: Correlations break down when market regimes change (e.g., bull vs. bear markets)
2. **Spurious Relationships**: Many correlations are coincidental, not causal
3. **Lack of Interpretability**: It's unclear *why* a signal works
4. **Overfitting Risk**: Complex models can memorize patterns without understanding mechanisms

## The Causal Solution

CRCA-Q implements **causal reasoning** based on Judea Pearl's framework:

- **Structural Causal Models (SCMs)**: Represent market variables as nodes in a DAG
- **Do-Calculus**: Reason about interventions using Pearl's do-operator
- **Counterfactual Reasoning**: Answer "what would have happened if" questions
- **Confounder Identification**: Distinguish direct causal effects from spurious correlations

## Mathematical Foundation

### Correlation vs. Causation

**Correlation**: $P(Y | X) \neq P(Y)$

**Causation**: $P(Y | do(X)) \neq P(Y)$

The key difference is the do-operator, which represents intervention rather than observation.

### Causal Effect

The causal effect of $X$ on $Y$ is:

$$E[Y | do(X=x)] = \sum_{z} E[Y | X=x, Z=z] P(Z=z)$$

Where $Z$ are confounders that need to be adjusted for.

## Example: Volume and Price

**Correlational view**: "High volume correlates with price increases"

**Causal view**: "High volume *causes* increased liquidity, which *causes* reduced price impact, which *causes* more efficient price discovery"

The causal view is more robust because it explains *why* the relationship exists and can predict behavior under interventions.

## Why This Matters

Causal models:
- Explain *why* relationships exist
- Predict behavior under interventions
- Remain valid across different market regimes
- Enable robust counterfactual analysis

## Next Steps

- [Architecture](architecture.md) - System architecture
- [Signal Generation](signal-generation.md) - Causal signal validation
