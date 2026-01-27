# CRCA-SD API

API reference for CRCA-SD socioeconomic dynamics.

## Classes

### StateVector

Socioeconomic state vector.

```python
class StateVector:
    P: float  # Population
    L: float  # Labor
    U: float  # Unemployment
    # ... 15 variables total
```

### DynamicsModel

System dynamics model.

```python
class DynamicsModel:
    def step(self, x_t: StateVector, u_t: ControlVector) -> StateVector:
        """Step forward one time period."""
```

## Mathematical Foundation

Dynamics follow:

$$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t)$$

## Next Steps

- [CRCA-CG](crca-cg.md) - Corporate governance API
