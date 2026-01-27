# Causal Reasoning Overview

Causal reasoning framework based on Judea Pearl's Structural Causal Models.

## Overview

Causal reasoning enables understanding of cause-and-effect relationships, going beyond correlation to identify true causal mechanisms.

## Key Concepts

- **Structural Causal Models (SCMs)**: Mathematical framework for causality
- **Do-Calculus**: Rules for causal inference
- **Counterfactuals**: "What-if" reasoning
- **Causal Discovery**: Learning causal structure from data
- **Interventions**: Reasoning about actions

## Mathematical Foundation

### Structural Causal Models

An SCM is a triple $(U, V, F)$ where:
- $U$: Exogenous variables
- $V$: Endogenous variables
- $F$: Structural equations

Each variable has:

$$V_i = f_i(Pa(V_i), U_i)$$

Where $Pa(V_i)$ are parents of $V_i$.

### Do-Calculus

The do-operator represents intervention:

$$P(Y | do(X=x)) = \sum_{z} P(Y | X=x, Z=z) P(Z=z)$$

## Documentation

- **[Structural Models](structural-models.md)**: SCMs
- **[Do-Calculus](do-calculus.md)**: Do-calculus implementation
- **[Counterfactuals](counterfactuals.md)**: Counterfactual reasoning

## Next Steps

- [Structural Models](structural-models.md) - SCMs
- [Do-Calculus](do-calculus.md) - Do-calculus
