# CRCAAgent API Reference

API reference for CRCAAgent and related classes.

## Overview

This section provides API documentation for CRCAAgent. Documentation is hybrid: auto-generated where docstrings exist, manually created where they are missing.

## Core Classes

- **[CRCAAgent](agent.md)**: Main agent class
- **[Methods](methods.md)**: Core methods
- **[Extraction](extraction.md)**: Variable extraction methods
- **[Simulation](simulation.md)**: Simulation methods
- **[Counterfactuals](counterfactuals.md)**: Counterfactual methods

## Mathematical Foundation

All methods implement operations on Structural Causal Models:

$$(U, V, F)$$

Where operations include:
- Variable extraction: $V = Extract(T)$
- Graph construction: $G = Build(V, E)$
- Causal inference: $E[Y | do(X=x)]$
- Counterfactuals: $P(Y_{x'} | X=x, Y=y)$

## Next Steps

- [CRCAAgent](agent.md) - Agent class
- [Methods](methods.md) - Core methods
