# CRCAAgent Overview

CRCAAgent is the core causal reasoning agent in the CR-CA framework. It combines structural causal models (SCMs) with large language model (LLM) integration to enable sophisticated causal inference and counterfactual analysis.

## Key Features

- **Automatic Variable Extraction**: Extract variables and causal relationships from natural language
- **Dual-Mode Operation**: Switch between LLM-based analysis and deterministic simulation
- **Causal Graph Management**: Build and manage causal directed acyclic graphs (DAGs)
- **Deterministic Simulation**: Perform mathematical causal simulations
- **Counterfactual Generation**: Generate "what-if" scenarios
- **Batch Prediction**: Process multiple scenarios simultaneously
- **Async Operations**: Concurrent analysis with async/await support

## Mathematical Foundation

CRCAAgent implements Structural Causal Models (SCMs) as defined by Judea Pearl. An SCM is a triple:

$$(U, V, F)$$

Where:
- $U$: Exogenous (unobserved) variables
- $V$: Endogenous (observed) variables
- $F$: Structural equations defining causal relationships

Each structural equation has the form:

$$V_i = f_i(Pa(V_i), U_i)$$

Where $Pa(V_i)$ are the parents of $V_i$ in the causal graph.

## Basic Usage

```python
from CRCA import CRCAAgent

agent = CRCAAgent(
    model_name="gpt-4o-mini",
    agent_name="my-agent"
)

result = agent.run("Analyze the causal relationship between X and Y")
```

## Core Components

- **[Initialization](initialization.md)**: Agent setup and configuration
- **[Automatic Extraction](automatic-extraction.md)**: Variable extraction from natural language
- **[Dual-Mode Operation](dual-mode-operation.md)**: LLM vs deterministic modes
- **[Causal Graph](causal-graph.md)**: Causal DAG construction and management
- **[Deterministic Simulation](deterministic-simulation.md)**: Mathematical simulation
- **[Counterfactuals](counterfactuals.md)**: Counterfactual scenario generation
- **[Batch Prediction](batch-prediction.md)**: Batch processing capabilities
- **[Async Operations](async-operations.md)**: Async/await support
- **[Optimization](optimization.md)**: Optimization methods
- **[Time-Series](time-series.md)**: Time-series analysis
- **[Bayesian Inference](bayesian-inference.md)**: Bayesian inference with credible intervals

## Next Steps

- [Initialization](initialization.md) - Configure your agent
- [Automatic Extraction](automatic-extraction.md) - Extract variables from text
- [Causal Graph](causal-graph.md) - Work with causal graphs
