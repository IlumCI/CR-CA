# CR-CA: Causal Reasoning and Counterfactual Analysis Framework

**Version**: v1.4.0  
**Repository**: [GitHub](https://github.com/IlumCI/CR-CA)  
**License**: Apache-2.0

CR-CA is a comprehensive framework for causal reasoning and counterfactual analysis, combining structural causal models (SCMs) with large language model (LLM) integration. The framework enables sophisticated causal inference, automatic variable extraction from natural language, deterministic causal simulation, and comprehensive counterfactual scenario generation.

## Quick Start

```bash
pip install crca
```

```python
from CRCA import CRCAAgent

agent = CRCAAgent(
    model_name="gpt-4o-mini",
    agent_name="my-causal-agent"
)

result = agent.run("Analyze the causal relationship between education and income")
```

## Key Features

### Core Capabilities

- **Automatic Variable Extraction**: Extract variables and causal relationships from natural language tasks
- **Dual-Mode Operation**: Switch between LLM-based analysis and deterministic simulation
- **Structural Causal Models**: Build and manage causal directed acyclic graphs (DAGs)
- **Deterministic Simulation**: Perform causal simulations with mathematical precision
- **Counterfactual Generation**: Generate "what-if" scenarios for decision support

### Advanced Analysis

- **Batch Prediction**: Process multiple scenarios simultaneously
- **Async Operations**: Concurrent analysis with async/await support
- **Optimization Methods**: Gradient-based and dynamic programming optimization
- **Time-Series Analysis**: Granger causality, VAR estimation, and temporal modeling
- **Bayesian Inference**: Posterior distributions with credible intervals

### Specialized Branches

- **CRCA-Q**: Quantitative trading with causal signals and portfolio optimization
- **CRCA-SD**: Socioeconomic dynamics with MPC solver and governance systems
- **CRCA-CG**: Corporate governance with CorporateSwarm and ESG scoring
- **General Agent**: Flexible general-purpose agent with personality system

## Mathematical Foundation

CR-CA is built on rigorous mathematical foundations:

### Structural Causal Models

A Structural Causal Model (SCM) is defined as a triple $(U, V, F)$ where:

- $U$: Exogenous (unobserved) variables
- $V$: Endogenous (observed) variables  
- $F$: Structural equations defining causal relationships

### Do-Calculus

The causal effect of intervention $do(X=x)$ on outcome $Y$ is:

$$E[Y | do(X=x)] = \sum_{z} E[Y | X=x, Z=z] P(Z=z)$$

Where $Z$ represents confounders that need to be adjusted for.

### Counterfactual Reasoning

Counterfactual queries answer "what would have happened if" questions:

$$P(Y_{x'} | X=x, Y=y)$$

This represents the probability of outcome $Y$ under intervention $X=x'$, given that we observed $X=x$ and $Y=y$.

## Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation and setup guides
- **[Core Modules](core/crcagent/overview.md)**: CRCAAgent and core functionality
- **[Branches](branches/crca-q/overview.md)**: Specialized agent implementations
- **[Features](features/image-annotation/overview.md)**: Image annotation, policy engine, hybrid agent
- **[Examples](examples/overview.md)**: Code examples and tutorials
- **[Architecture](architecture/overview.md)**: System design and principles
- **[API Reference](api/crca/overview.md)**: Complete API documentation

## Installation

```bash
# From PyPI
pip install crca

# From source
git clone https://github.com/IlumCI/CR-CA.git
cd CR-CA
pip install -r requirements.txt
```

## Quick Links

- [Installation Guide](getting-started/installation.md)
- [Quickstart Tutorial](getting-started/quickstart.md)
- [API Reference](api/crca/overview.md)
- [Examples](examples/overview.md)
- [Contributing](contributing/overview.md)

## Research & Theory

CR-CA implements concepts from:

- **Judea Pearl's Causal Hierarchy**: Association, intervention, and counterfactual reasoning
- **Structural Causal Models**: Mathematical framework for causal inference
- **Do-Calculus**: Rules for identifying causal effects from observational data
- **Counterfactual Analysis**: Reasoning about alternative scenarios

For detailed mathematical foundations, see the [Causal Reasoning](features/causal-reasoning/overview.md) section.
