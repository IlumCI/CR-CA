# MPC & Control Systems

Model Predictive Control and policy execution systems.

## Model Predictive Control (MPC)

MPC solves optimization problems over a prediction horizon to determine optimal control actions.

### Mathematical Foundation

MPC solves:

$$\min_{\mathbf{u}_{t:t+H-1}} \sum_{k=0}^{H-1} \ell(\mathbf{x}_{t+k}, \mathbf{u}_{t+k}) + V(\mathbf{x}_{t+H})$$

Subject to:

$$\mathbf{x}_{t+k+1} = f(\mathbf{x}_{t+k}, \mathbf{u}_{t+k})$$
$$\mathbf{g}(\mathbf{x}_{t+k}, \mathbf{u}_{t+k}) \leq 0$$

### Stage Cost

$$\ell(\mathbf{x}, \mathbf{u}) = \|\mathbf{x} - \mathbf{x}_{ref}\|_Q^2 + \|\mathbf{u} - \mathbf{u}_{ref}\|_R^2$$

### Usage

```python
from templates.mpc_planner import MPCPlanner

planner = MPCPlanner(horizon=10)
optimal_controls = planner.solve(current_state, reference_state)
```

## Policy Loop

Policy loop executes policy decisions in a temporal loop.

### Mathematical Foundation

Policy loop implements:

$$\pi_t = \arg\max_{\pi} E\left[\sum_{k=0}^H \gamma^k r_{t+k} | s_t, \pi\right]$$

Where:
- $\pi_t$: Policy at time $t$
- $r_{t+k}$: Reward at time $t+k$
- $\gamma$: Discount factor
- $H$: Horizon

### Usage

```python
from templates.policy_loop import PolicyLoopMixin

class PolicyAgent(BaseAgent, PolicyLoopMixin):
    def __init__(self):
        super().__init__()
```

## Sensors & Actuators

Sensors observe system state, actuators execute actions.

### Mathematical Foundation

**Sensors** observe state:

$$o_t = h(s_t, \nu_t)$$

Where:
- $o_t$: Observation
- $s_t$: True state
- $\nu_t$: Observation noise

**Actuators** execute actions:

$$a_t = \pi(o_t)$$

Where $\pi$ is the policy.

### Usage

```python
observation = sensor.observe()
actuator.execute(action)
```

## Next Steps

- [Monitoring](monitoring.md) - Monitoring and drift detection
