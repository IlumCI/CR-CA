# Causal Graph Architecture

Architecture for causal graph representation and manipulation.

## Overview

Causal graphs are represented as directed acyclic graphs (DAGs) with nodes for variables and edges for causal relationships.

## Mathematical Foundation

A causal graph $G = (V, E)$ where:
- $V$: Set of variables
- $E$: Set of causal edges $(X \to Y)$

The graph must satisfy acyclicity:

$$\nexists \text{ path } X \to \cdots \to X$$

## Graph Operations

### Path Finding

Find all paths from $X$ to $Y$:

$$Paths(X, Y) = \{p : X \to^* Y\}$$

### Confounder Identification

Identify confounders $Z$ for relationship $X \to Y$:

$$Z \in Confounders(X, Y) \iff Z \to X \text{ and } Z \to Y$$

### Adjustment Sets

Find valid adjustment set for $X \to Y$:

$$Z \text{ valid } \iff Z \text{ blocks all backdoor paths}$$

## Next Steps

- [LLM Integration](llm-integration.md) - LLM integration architecture
