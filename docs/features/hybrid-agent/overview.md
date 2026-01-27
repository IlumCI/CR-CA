# Hybrid Agent Overview

The hybrid agent architecture combines multiple reasoning components for robust causal analysis.

## Overview

Hybrid agent integrates consistency engine, conversation manager, explanation generator, and other components for enhanced reasoning.

## Key Components

- **Consistency Engine**: Ensures logical consistency
- **Conversation Manager**: Manages dialogue
- **Explanation Generator**: Generates explanations
- **Task Decomposer**: Decomposes complex tasks
- **Self Verifier**: Verifies reasoning steps

## Mathematical Foundation

Hybrid agent combines reasoning modes:

$$R_{hybrid} = \alpha R_{LLM} + \beta R_{deterministic} + \gamma R_{symbolic}$$

Where $\alpha + \beta + \gamma = 1$ and each $R_i$ is a reasoning component.

## Documentation

- **[Consistency Engine](consistency-engine.md)**: Consistency mechanisms
- **[Conversation Manager](conversation-manager.md)**: Dialogue management
- **[Task Decomposer](task-decomposer.md)**: Task decomposition

## Next Steps

- [Consistency Engine](consistency-engine.md) - Consistency mechanisms
- [Conversation Manager](conversation-manager.md) - Dialogue management
