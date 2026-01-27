## CR-CA v1.0 Stability Policy

This document defines what **stable** means for the v1.0 release.

### Stable Guarantees
- **Structured outputs only**: all core results are JSON/Pydantic structures.
- **Refusal-first**: numeric causal outputs are refused unless prerequisites are met.
- **SCM-required counterfactuals**: counterfactuals require explicit structural equations.
- **Identification gating**: estimation requires an `IdentificationResult`.

### Known Limits (by design)
- **Identification is conservative**: latent-confounded cases often return non-identifiable.
- **Discovery is hypothesis-only**: outputs are not causal truth, only hypotheses under assumptions.
- **LLM layer is non-authoritative**: drafts and orchestration only; no numeric causal claims.

### Compatibility
- Schema contracts are versioned via the exported JSON schemas in `schemas_export/crca_core`.
- Changes that alter semantics must increment schema versions before release.

