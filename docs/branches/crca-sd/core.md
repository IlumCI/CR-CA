# CRCA-SD Core

Core components of CRCA-SD including state vectors, control vectors, and dynamics models.

## State Vector

The state vector represents the socioeconomic system state with 15 variables:

$$\mathbf{x}_t = [P_t, L_t, U_t, W_t, S_t, \ell_t, E_{cap,t}, H_{cap,t}, K_t, I_t, T_{cap,t}, E_{stock,t}, F_{stock,t}, M_{stock,t}, C_t, Y_t]'$$

### Variable Definitions

- $P_t$: Population
- $L_t$: Labor force
- $U_t$: Unemployment rate $[0, 1]$
- $W_t$: Average wage proxy
- $S_t$: Social stability proxy $[0, 1]$
- $\ell_t$: Literacy rate $[0, 1]$
- $E_{cap,t}$: Education capacity
- $H_{cap,t}$: Healthcare capacity
- $K_t$: Productive capital stock
- $I_t$: Infrastructure health index $[0, 1]$
- $T_{cap,t}$: Transport capacity
- $E_{stock,t}$: Stored energy
- $F_{stock,t}$: Food stock
- $M_{stock,t}$: Critical materials stock
- $C_t$: Ecological damage (irreversible)
- $Y_t$: GDP proxy

## Control Vector

The control vector represents budget allocations:

$$\mathbf{u}_t = [b_E, b_F, b_H, b_I, b_K, b_RD, b_welfare, b_security]'$$

Where each $b_i$ is a budget share such that:

$$\sum_i b_i = 1, \quad b_i \geq 0$$

## Dynamics Model

The dynamics model implements discrete-time evolution:

$$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t)$$

### Key Dynamics

**Population Growth**:
$$P_{t+1} = P_t (1 + \beta - \delta + \mu)$$

Where $\beta$ is birth rate, $\delta$ is death rate, and $\mu$ is migration.

**Capital Accumulation**:
$$K_{t+1} = (1 - \delta_K) K_t + \kappa_K b_K B_t$$

Where $\delta_K$ is depreciation and $\kappa_K$ is investment efficiency.

**Output Production** (Cobb-Douglas with gating):
$$Y_{t+1} = A_t K_t^\alpha L_t^{1-\alpha} g_E h_T$$

Where $g_E$ and $h_T$ are energy and transport gates.

## Usage

```python
from branches.crca_sd.crca_sd_core import StateVector, ControlVector, DynamicsModel

# Initialize state
state = StateVector(P=1000000, L=500000, U=0.05)

# Define control
control = ControlVector(b_E=0.1, b_F=0.2, b_H=0.15)

# Create dynamics model
dynamics = DynamicsModel()

# Step forward
next_state = dynamics.step(state, control)
```

## Next Steps

- [MPC Solver](mpc-solver.md) - MPC optimization
- [Governance](governance.md) - Governance systems
