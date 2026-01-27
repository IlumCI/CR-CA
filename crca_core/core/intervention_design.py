"""Intervention/experiment design (v0.1: graphical, non-probabilistic).

This module is intentionally conservative:
- It does not invent numeric information gain.
- It produces structured candidate designs with explicit prerequisites and rationale.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from crca_core.models.provenance import ProvenanceManifest
from crca_core.models.result import InterventionDesignResult
from utils.canonical import stable_hash


class TargetQuery(BaseModel):
    query_type: Literal["identify_effect", "reduce_uncertainty_edge"] = "identify_effect"
    treatment: Optional[str] = None
    outcome: Optional[str] = None
    edge_source: Optional[str] = None
    edge_target: Optional[str] = None


class FeasibilityConstraints(BaseModel):
    manipulable_variables: List[str] = Field(default_factory=list)
    observable_variables: List[str] = Field(default_factory=list)
    costs: Dict[str, Any] = Field(default_factory=dict)
    ethics_notes: Optional[str] = None


class DesignCandidate(BaseModel):
    design_type: str
    mechanism: str
    prerequisites: List[str] = Field(default_factory=list)
    feasibility_inputs_needed: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


def _parents_of(graph_edges, node: str) -> List[str]:
    res: List[str] = []
    for e in graph_edges:
        if e.target == node:
            res.append(e.source)
    return res


def design_intervention(
    *,
    locked_spec: Any,
    target_query: TargetQuery,
    constraints: Optional[FeasibilityConstraints] = None,
) -> InterventionDesignResult:
    """Propose interventions/measurements to support identifiability or hypothesis discrimination.

    v0.1 is graphical and deliberately non-numeric. It does not claim identifiability;
    it produces designs and checklists that *could* help under explicit assumptions.
    """

    constraints = constraints or FeasibilityConstraints()

    # Provenance: we hash only the spec hash + query + constraints (no raw data).
    spec_hash = getattr(locked_spec, "spec_hash", "unknown")
    prov = ProvenanceManifest.minimal(
        spec_hash=stable_hash(
            {
                "spec_hash": spec_hash,
                "target_query": target_query.model_dump(),
                "constraints": constraints.model_dump(),
                "module": "intervention_design_v0.1",
            }
        )
    )

    designs: List[Dict[str, Any]] = []

    if target_query.query_type == "identify_effect":
        X = target_query.treatment
        Y = target_query.outcome
        if not X or not Y:
            return InterventionDesignResult(
                provenance=prov,
                assumptions=[],
                limitations=["Missing treatment/outcome in target_query."],
                designs=[],
            )

        # 1) Randomize treatment if feasible
        if X in constraints.manipulable_variables:
            designs.append(
                DesignCandidate(
                    design_type="randomize_treatment",
                    mechanism=f"Randomize {X} to break backdoor confounding when estimating effect on {Y}.",
                    prerequisites=[
                        "Well-defined intervention on treatment (consistency/SUTVA).",
                        "No interference between units (SUTVA).",
                        "Feasible randomization protocol and compliance monitoring.",
                    ],
                    feasibility_inputs_needed=["sample_size", "randomization_unit", "ethical_constraints"],
                ).model_dump()
            )

        # 2) Measure candidate confounders (parents of treatment in the current draft DAG)
        parents_x = _parents_of(locked_spec.graph.edges, X)
        if parents_x:
            designs.append(
                DesignCandidate(
                    design_type="measure_confounder_candidates",
                    mechanism=(
                        f"Measure candidate confounders {parents_x} because they are modeled as direct causes of {X}; "
                        f"conditioning/adjusting may help estimate effect of {X} on {Y} under exchangeability."
                    ),
                    prerequisites=[
                        "Candidate confounders are measured without severe error or are modeled as proxies.",
                        "Exchangeability holds conditional on measured covariates (assumption).",
                        "Positivity/overlap holds in the collected data.",
                    ],
                    feasibility_inputs_needed=["measurement_instrument_quality", "data_collection_costs"],
                    notes="This does not guarantee identifiability; it is a measurement suggestion grounded in the current graph hypothesis.",
                ).model_dump()
            )

        # 3) Instrument design if user provided candidate instruments
        if getattr(locked_spec.roles, "instruments", []):
            Zs = list(getattr(locked_spec.roles, "instruments", []))
            designs.append(
                DesignCandidate(
                    design_type="instrument_design",
                    mechanism=f"Collect/create instrument(s) {Zs} to identify effect of {X} on {Y} under IV assumptions.",
                    prerequisites=[
                        "Relevance: Z affects X.",
                        "Exclusion: Z affects Y only through X.",
                        "Independence: Z is independent of unmeasured causes of Y.",
                    ],
                    feasibility_inputs_needed=["instrument_source", "exclusion_justification"],
                ).model_dump()
            )

    elif target_query.query_type == "reduce_uncertainty_edge":
        s = target_query.edge_source
        t = target_query.edge_target
        if not s or not t:
            return InterventionDesignResult(
                provenance=prov,
                assumptions=[],
                limitations=["Missing edge_source/edge_target in target_query."],
                designs=[],
            )

        if s in constraints.manipulable_variables:
            designs.append(
                DesignCandidate(
                    design_type="perturb_source",
                    mechanism=f"Intervene on {s} (do({s}=...)) and observe downstream changes in {t} to test the edge hypothesis {s}→{t}.",
                    prerequisites=[
                        "Well-defined intervention on source variable.",
                        "No simultaneous changes to other upstream causes (or they are measured/controlled).",
                    ],
                    feasibility_inputs_needed=["intervention_range", "measurement_frequency", "time_horizon"],
                ).model_dump()
            )

    return InterventionDesignResult(
        provenance=prov,
        assumptions=[],
        limitations=[
            "v0.1 design is graphical and non-probabilistic; it does not compute numeric information gain.",
            "Suggestions depend on the correctness of the locked causal graph/spec assumptions.",
        ],
        designs=designs,
    )

