# Integration Examples

Examples of integrating CR-CA with other systems.

## CRCA-SD Integration

Integration with CRCA-SD for socioeconomic dynamics:

```python
from branches.crca_sd.crca_sd_core import StateVector, DynamicsModel
from CRCA import CRCAAgent

crca = CRCAAgent(model_name="gpt-4o-mini")
state = StateVector(P=1000000)
dynamics = DynamicsModel()

result = crca.run("Analyze policy impact on GDP")
```

## CRCA-CG Integration

Integration with CRCA-CG for corporate governance:

```python
from branches.crca_cg.corposwarm import CorporateSwarm
from CRCA import CRCAAgent

crca = CRCAAgent(model_name="gpt-4o-mini")
swarm = CorporateSwarm()

result = crca.run("Analyze governance decision impact")
```

## Palantir Integration

Integration with Palantir systems:

```python
from CRCA import CRCAAgent

agent = CRCAAgent(model_name="gpt-4o-mini")
result = agent.run("Analyze data from Palantir system")
```

## Data Broker Integration

Integration with data broker systems:

```python
from CRCA import CRCAAgent

agent = CRCAAgent(model_name="gpt-4o-mini")
result = agent.run("Fetch and analyze data from broker")
```

## Next Steps

- [Architecture](../architecture/overview.md) - System architecture
