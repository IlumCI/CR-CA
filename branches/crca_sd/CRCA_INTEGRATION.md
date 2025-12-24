# CRCA Integration in crca_sd Core Module

## Answer: **Now it does!**

Previously, the `crca_sd` core module did **NOT** use CRCA - it only used:
- Random scenario generation (Gaussian/Student-t)
- MPC optimization
- System dynamics simulation

**Now CRCA is integrated into the core module.**

## What Changed

### 1. **ScenarioGenerator** Now Supports CRCA

The `ScenarioGenerator` class now accepts an optional `crca_agent` parameter:

```python
from crca_sd import get_crca_agent
from crca_sd.crca_sd_mpc import ScenarioGenerator

# Get CRCAAgent (if available)
crca = get_crca_agent(variables=['Y', 'U', 'S', 'P', 'L', 'W', 'K', 'I'])

# Create scenario generator with CRCA
gen = ScenarioGenerator(crca_agent=crca)

# Generate causal scenarios (not just random)
scenarios = gen.generate_causal_scenarios(
    n_scenarios=10,
    horizon=12,
    current_state=state_vector,
    target_variables=['Y', 'U', 'S']  # Focus on GDP, unemployment, stability
)
```

### 2. **New Method: `generate_causal_scenarios()`**

Instead of just random noise, this method:
- Uses CRCA's counterfactual analysis
- Generates scenarios based on causal relationships
- Falls back to Gaussian if CRCA not available

### 3. **Module-Level CRCA Access**

The `crca_sd` module now exports:

```python
from crca_sd import get_crca_agent, CRCA_AVAILABLE

if CRCA_AVAILABLE:
    crca = get_crca_agent(variables=['Y', 'U', 'S'])
```

## How It Works

### Before (No CRCA):
```python
# Random scenarios
scenarios = gen.generate_gaussian(10, 12)
# → Just noise, no causal understanding
```

### Now (With CRCA):
```python
# Causal scenarios
scenarios = gen.generate_causal_scenarios(10, 12, current_state)
# → Based on causal relationships and counterfactuals
```

## Benefits

1. **Causal Understanding**: Scenarios are based on causal relationships, not just correlations
2. **Counterfactual Analysis**: Explores "what-if" scenarios using causal reasoning
3. **Better Policy Optimization**: MPC uses causally-informed scenarios
4. **Backward Compatible**: Falls back to Gaussian if CRCA not available

## Usage Example

```python
from crca_sd import get_crca_agent
from crca_sd.crca_sd_mpc import ScenarioGenerator, MPCSolver
from crca_sd.crca_sd_core import StateVector, DynamicsModel

# Initialize with CRCA
crca = get_crca_agent(
    variables=['P', 'L', 'U', 'W', 'S', 'Y', 'K', 'I'],
    agent_name='crca-sd-core'
)

# Build causal graph
crca.add_causal_relationship('K', 'Y', strength=0.7)  # Capital → GDP
crca.add_causal_relationship('L', 'Y', strength=0.5)  # Labor → GDP
crca.add_causal_relationship('Y', 'W', strength=0.4)   # GDP → Wages

# Create scenario generator with CRCA
gen = ScenarioGenerator(crca_agent=crca)

# Generate causal scenarios
current_state = StateVector(Y=1e9, U=0.07, S=0.65)
scenarios = gen.generate_causal_scenarios(10, 12, current_state)

# Use in MPC
solver = MPCSolver(dynamics, checker, objective_computer)
policy, info = solver.solve(current_state, scenarios)
```

## Fallback Behavior

If CRCA is not available:
- `get_crca_agent()` returns `None`
- `generate_causal_scenarios()` falls back to `generate_gaussian()`
- System continues to work normally (just without causal reasoning)

## Summary

**Before**: `crca_sd` did NOT use CRCA  
**Now**: `crca_sd` **DOES** use CRCA for causal scenario generation

The core module now truly integrates causal reasoning and counterfactual analysis, not just optimization.

