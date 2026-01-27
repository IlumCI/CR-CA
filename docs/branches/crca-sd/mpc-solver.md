# MPC Solver

Model Predictive Control (MPC) solver for CRCA-SD.

## Overview

The MPC solver optimizes control actions over a prediction horizon to achieve policy objectives while satisfying constraints.

## Mathematical Foundation

### MPC Optimization Problem

$$\min_{\mathbf{u}_{t:t+H-1}} \sum_{k=0}^{H-1} \ell(\mathbf{x}_{t+k}, \mathbf{u}_{t+k}) + V(\mathbf{x}_{t+H})$$

Subject to:

$$\mathbf{x}_{t+k+1} = f(\mathbf{x}_{t+k}, \mathbf{u}_{t+k}, \mathbf{w}_{t+k})$$
$$\mathbf{g}(\mathbf{x}_{t+k}, \mathbf{u}_{t+k}) \leq 0$$
$$\mathbf{h}(\mathbf{x}_{t+k}, \mathbf{u}_{t+k}) = 0$$

Where:
- $H$: Prediction horizon
- $\ell$: Stage cost function
- $V$: Terminal cost function
- $f$: System dynamics
- $\mathbf{g}, \mathbf{h}$: Inequality and equality constraints

### Stage Cost

The stage cost typically includes:

$$\ell(\mathbf{x}, \mathbf{u}) = \|\mathbf{x} - \mathbf{x}_{ref}\|_Q^2 + \|\mathbf{u} - \mathbf{u}_{ref}\|_R^2$$

Where $Q$ and $R$ are weighting matrices.

## Usage

```python
from branches.crca_sd.crca_sd_mpc import MPCSolver

solver = MPCSolver(
    horizon=10,
    state_weights=Q,
    control_weights=R
)

# Solve MPC problem
optimal_controls = solver.solve(
    current_state=state,
    reference_state=ref_state,
    constraints=constraints
)
```

## State Estimation

The solver includes state estimation for unobserved variables:

$$\hat{\mathbf{x}}_t = E[\mathbf{x}_t | \mathbf{z}_{0:t}]$$

Where $\mathbf{z}_{0:t}$ are observations up to time $t$.

## Next Steps

- [Governance](governance.md) - Governance systems
