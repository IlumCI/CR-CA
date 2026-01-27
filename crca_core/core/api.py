"""Public API functions for the H1 `crca_core`.

These functions provide the stable, refusal-first entry points that other
layers (including LLM tooling) should call.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from crca_core.core.estimate import EstimatorConfig, estimate_effect_dowhy
from crca_core.identify import identify_effect
from crca_core.core.intervention_design import (
    FeasibilityConstraints,
    TargetQuery,
    design_intervention,
)
from crca_core.models.provenance import ProvenanceManifest
from crca_core.models.refusal import RefusalChecklistItem, RefusalReasonCode, RefusalResult
from crca_core.models.result import CounterfactualResult
from crca_core.scm import LinearGaussianSCM
from crca_core.models.spec import DraftSpec, LockedSpec
from crca_core.timeseries.pcmci import PCMCIConfig, discover_timeseries_pcmci
from crca_core.discovery.tabular import TabularDiscoveryConfig, discover_tabular
from utils.canonical import stable_hash


def simulate_counterfactual(
    *,
    locked_spec: LockedSpec,
    factual_observation: Dict[str, float],
    intervention: Dict[str, float],
    allow_partial_observation: bool = False,
) -> CounterfactualResult | RefusalResult:
    """Simulate a counterfactual under an explicit SCM (required).

    Refuses if `locked_spec.scm` is missing.
    """

    if locked_spec.scm is None:
        return RefusalResult(
            message="Counterfactuals require an explicit SCMSpec (structural equations + noise model).",
            reason_codes=[RefusalReasonCode.NO_SCM_FOR_COUNTERFACTUAL],
            checklist=[
                RefusalChecklistItem(
                    item="Provide SCMSpec",
                    rationale="A DAG alone does not define counterfactual semantics; SCM is required.",
                )
            ],
            suggested_next_steps=[
                "Attach a SCMSpec (e.g., linear_gaussian) to the spec, then re-lock and retry."
            ],
        )

    scm = LinearGaussianSCM.from_spec(locked_spec.scm)
    try:
        u = scm.abduce_noise(factual_observation, allow_partial=allow_partial_observation)
    except ValueError as exc:
        return RefusalResult(
            message=str(exc),
            reason_codes=[RefusalReasonCode.INPUT_INVALID],
            checklist=[
                RefusalChecklistItem(
                    item="Provide complete factual observation",
                    rationale="Counterfactuals require abduction for all endogenous variables in v1.0 unless partial mode is enabled.",
                )
            ],
            suggested_next_steps=[
                "Provide all endogenous variables or set allow_partial_observation=True (partial mode)."
            ],
        )
    cf = scm.predict(u, interventions=intervention)

    prov = ProvenanceManifest.minimal(
        spec_hash=stable_hash(
            {
                "spec_hash": locked_spec.spec_hash,
                "module": "simulate_counterfactual",
                "intervention": intervention,
                "factual_keys": sorted(list(factual_observation.keys())),
            }
        )
    )

    return CounterfactualResult(
        provenance=prov,
        assumptions=[
            "SCM structure and parameters are correct (strong assumption).",
            "Factual observation includes all endogenous variables for abduction in v1.0 unless partial mode is enabled.",
        ],
        limitations=[
            "v0.1 counterfactuals require a fully observed system (no missing variables).",
            "Only linear-Gaussian SCMs are supported in v0.1.",
        ],
        counterfactual={"factual": dict(factual_observation), "do": dict(intervention), "result": cf},
    )


__all__ = [
    # Core lifecycle
    "DraftSpec",
    "LockedSpec",
    # Identification
    "identify_effect",
    # Discovery
    "TabularDiscoveryConfig",
    "discover_tabular",
    "PCMCIConfig",
    "discover_timeseries_pcmci",
    # Design
    "TargetQuery",
    "FeasibilityConstraints",
    "design_intervention",
    # Counterfactuals
    "simulate_counterfactual",
    # Estimation
    "EstimatorConfig",
    "estimate_effect_dowhy",
]

