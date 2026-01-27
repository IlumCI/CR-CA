# Deterministic Simulation

CRCAAgent performs deterministic causal simulations using structural equations.

## Overview

Deterministic simulation uses known or estimated structural equations to compute causal effects mathematically, without relying on LLM inference.

## Mathematical Foundation

Given structural equations:

$$V_i = f_i(Pa(V_i), U_i)$$

For $i = 1, \ldots, n$, we can simulate the system by:

1. Setting exogenous variables $U_i$
2. Computing variables in topological order
3. Propagating values through the graph

## Basic Simulation

```python
# Simulate with specific variable values
result = agent.simulate(
    variables={
        "education": 16,
        "experience": 5
    },
    target="income"
)
```

## Simulation Methods

### Euler Method

For differential equation systems:

$$V_i(t+\Delta t) = V_i(t) + f_i(Pa(V_i(t)), U_i) \cdot \Delta t$$

### Runge-Kutta 4th Order

More accurate for complex systems:

$$k_1 = f(t, V(t))$$
$$k_2 = f(t + \Delta t/2, V(t) + k_1 \Delta t/2)$$
$$k_3 = f(t + \Delta t/2, V(t) + k_2 \Delta t/2)$$
$$k_4 = f(t + \Delta t, V(t) + k_3 \Delta t)$$
$$V(t+\Delta t) = V(t) + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

## Configuration

```python
agent = CRCAAgent(
    model_name="gpt-4o-mini",
    simulation_method="rk4",  # or "euler"
    simulation_steps=100,
    simulation_dt=0.01
)
```

## Time-Series Simulation

Simulate over time:

```python
time_series = agent.simulate_time_series(
    initial_conditions={"X": 0, "Y": 0},
    time_steps=100,
    dt=0.1
)
```

## Next Steps

- [Counterfactuals](counterfactuals.md) - Generate counterfactual scenarios
- [Optimization](optimization.md) - Optimize interventions
