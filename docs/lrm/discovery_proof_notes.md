---
title: Discovery Proof Notes
---

# Discovery Proof Notes (Assumptions + Correctness)

## PC algorithm
- **Assumptions**: causal sufficiency, faithfulness, correct CI test.
- **Output**: CPDAG (equivalence class).
- **Correctness**: asymptotic correctness under assumptions (Spirtes et al.).

## FCI algorithm
- **Assumptions**: faithfulness, correct CI test, allows latent confounding.
- **Output**: PAG (partial ancestral graph).
- **Correctness**: relies on ancestral graph Markov properties (Richardson & Spirtes).

## GES
- **Assumptions**: correct scoring criterion (e.g., BIC), large sample.
- **Output**: CPDAG of highest score.
- **Correctness**: asymptotic consistency for score-equivalent metrics.

## PCMCI / PCMCI+
- **Assumptions**: stationarity, lag sufficiency, correct CI test.
- **Output**: lagged causal graph hypothesis.
- **Correctness**: consistency depends on correct lag window and CI testing.
