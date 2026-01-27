## v1.0 Release Notes (draft)

### Core Features
- In-house identification with proof objects and conservative scope labeling.
- Real discovery backends: causal-learn (tabular) and tigramite (time-series).
- Linear-Gaussian SCM counterfactuals with partial-abduction mode (optional).
- Estimation/refutation gated by identification results.
- Schema exports + generated documentation + benchmark harness outputs.

### Non-Authoritative LLM Layer
- DraftSpec generation only.
- Tool orchestration is gated by LockedSpec and refusal rules.

### Known Limitations
- ID algorithm is conservative for latent-confounded cases.
- Discovery outputs are hypotheses and require assumptions.

