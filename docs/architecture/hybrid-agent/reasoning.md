# Reasoning Architecture

The hybrid agent's reasoning architecture.

## Overview

Reasoning architecture combines multiple reasoning modes for robust inference.

## Mathematical Foundation

Reasoning combines:

$$R_{final} = \arg\max_{r} \sum_{i} w_i P(r | M_i)$$

Where:
- $M_i$: Different reasoning modes
- $w_i$: Weights for each mode
- $P(r | M_i)$: Probability of result $r$ under mode $M_i$

## Next Steps

- [Causal Graphs](../causal-graphs.md) - Causal graph architecture
