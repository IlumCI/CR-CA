# Do-Calculus

Do-calculus provides rules for identifying causal effects from observational data.

## Overview

Do-calculus, developed by Judea Pearl, provides three rules for manipulating probability expressions involving the do-operator.

## Mathematical Foundation

### Do-Operator

The do-operator $do(X=x)$ represents intervention, setting $X$ to value $x$:

$$P(Y | do(X=x)) \neq P(Y | X=x)$$

The key difference: observation vs. intervention.

### Do-Calculus Rules

**Rule 1** (Insertion/deletion of observations):
$$P(Y | do(X), Z, W) = P(Y | do(X), W)$$

If $Z$ is irrelevant to $Y$ given $X$ and $W$ in the graph.

**Rule 2** (Action/observation exchange):
$$P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W)$$

If $Z$ is irrelevant to $Y$ given $X$ and $W$ in the modified graph.

**Rule 3** (Insertion/deletion of actions):
$$P(Y | do(X), do(Z), W) = P(Y | do(X), W)$$

If $Z$ is irrelevant to $Y$ given $X$ and $W$ in the modified graph.

## Causal Effect Identification

Using do-calculus, we can identify:

$$E[Y | do(X=x)] = \sum_{z} E[Y | X=x, Z=z] P(Z=z)$$

Where $Z$ is a valid adjustment set.

## Usage

CR-CA automatically applies do-calculus rules when computing causal effects.

## Next Steps

- [Counterfactuals](counterfactuals.md) - Counterfactual reasoning
