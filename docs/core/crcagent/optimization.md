# Optimization Methods

CRCAAgent includes optimization methods for finding optimal interventions.

## Overview

Optimization methods find interventions that maximize or minimize target outcomes subject to constraints.

## Mathematical Foundation

### Optimization Problem

Find intervention $x^*$ that maximizes expected outcome:

$$x^* = \arg\max_x E[Y | do(X=x)]$$

Subject to constraints:

$$g_i(x) \leq 0, \quad i = 1, \ldots, m$$
$$h_j(x) = 0, \quad j = 1, \ldots, p$$

### Gradient-Based Optimization

For differentiable objectives:

$$\nabla_x E[Y | do(X=x)] = 0$$

Using gradient descent:

$$x_{t+1} = x_t - \alpha \nabla_x E[Y | do(X=x_t)]$$

Where $\alpha$ is the learning rate.

### Dynamic Programming

For sequential decision problems:

$$V_t(s) = \max_a \left\{ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_{t+1}(s') \right\}$$

## Usage

### Gradient-Based

```python
optimal = agent.optimize_intervention(
    target="income",
    method="gradient",
    constraints={"education": (0, 20)}
)
```

### Dynamic Programming

```python
optimal = agent.optimize_intervention(
    target="income",
    method="bellman",
    horizon=10
)
```

## Next Steps

- [Time-Series](time-series.md) - Time-series analysis
- [Bayesian Inference](bayesian-inference.md) - Bayesian methods
