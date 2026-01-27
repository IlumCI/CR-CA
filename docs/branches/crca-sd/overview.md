# CRCA-SD Overview

CRCA-SD (CRCA for Socioeconomic Dynamics) is a constrained, stochastic, multi-objective model-predictive control system for socioeconomic dynamics and logistics optimization.

## Overview

CRCA-SD integrates CRCA (Causal Reasoning and Counterfactual Analysis) for:
- Causal scenario generation
- Causal policy validation
- Understanding why policies work

## Key Features

- **State Vector**: 15 core socioeconomic variables
- **Control Vector**: Budget shares and allocations
- **Dynamics Model**: Macro system dynamics equations
- **MPC Solver**: Model Predictive Control optimization
- **Governance Systems**: Policy execution and monitoring
- **Real-Time Control**: Real-time data acquisition and control

## Mathematical Foundation

CRCA-SD models socioeconomic systems using state-space representation:

$$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t)$$

Where:
- $\mathbf{x}_t$: State vector (15 variables)
- $\mathbf{u}_t$: Control vector (budget allocations)
- $\mathbf{w}_t$: Exogenous shocks

The MPC optimization problem is:

$$\min_{\mathbf{u}_{t:t+H-1}} \sum_{k=0}^{H-1} \ell(\mathbf{x}_{t+k}, \mathbf{u}_{t+k}) + V(\mathbf{x}_{t+H})$$

Subject to:

$$\mathbf{x}_{t+k+1} = f(\mathbf{x}_{t+k}, \mathbf{u}_{t+k}, \mathbf{w}_{t+k})$$
$$\mathbf{g}(\mathbf{x}_{t+k}, \mathbf{u}_{t+k}) \leq 0$$

## State Variables

The state vector contains 15 variables:
- **Macro**: Population (P), Labor (L), Unemployment (U), Wage (W), Stability (S)
- **Human Capital**: Literacy (ℓ), Education Capacity (Ecap), Healthcare Capacity (Hcap)
- **Capital**: Capital Stock (K), Infrastructure Health (I), Transport Capacity (Tcap)
- **Stocks**: Energy (E_stock), Food (F_stock), Materials (M_stock), Ecological Damage (C)
- **Output**: GDP Proxy (Y)

## Documentation

- **[Core](core.md)**: Core CRCA-SD functionality
- **[MPC Solver](mpc-solver.md)**: Model Predictive Control implementation
- **[Governance](governance.md)**: Governance systems

## Next Steps

- [Core](core.md) - Core functionality
- [MPC Solver](mpc-solver.md) - MPC optimization
