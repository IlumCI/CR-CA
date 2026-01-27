# Bayesian Inference

CRCAAgent supports Bayesian inference for causal effects with uncertainty quantification.

## Overview

Bayesian inference provides posterior distributions over causal effects, enabling uncertainty quantification through credible intervals.

## Mathematical Foundation

### Posterior Distribution

Given data $D$ and prior $P(\theta)$, the posterior is:

$$P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}$$

### Causal Effect Posterior

For causal effect $\tau = E[Y | do(X=1)] - E[Y | do(X=0)]$:

$$P(\tau | D) = \int P(\tau | \theta) P(\theta | D) d\theta$$

### Credible Intervals

A $(1-\alpha)$ credible interval $[a, b]$ satisfies:

$$P(a \leq \tau \leq b | D) = 1 - \alpha$$

## Usage

### Bayesian Causal Inference

```python
# Compute posterior distribution
posterior = agent.bayesian_causal_inference(
    treatment="X",
    outcome="Y",
    data=dataframe
)

# Get credible interval
ci = posterior.credible_interval(alpha=0.05)
print(f"95% CI: [{ci.lower}, {ci.upper}]")
```

### Bootstrap Sampling

```python
# Bootstrap for uncertainty
samples = agent.bootstrap_causal_effect(
    treatment="X",
    outcome="Y",
    n_samples=1000
)
```

## Next Steps

- [Core Modules Overview](../overview.md) - Return to core modules
- [Utils](../utils/overview.md) - Utility functions
