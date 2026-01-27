"""Identification entry points for crca_core (in-house)."""

from __future__ import annotations

from typing import Dict, List, Optional

from crca_core.identify.backdoor import find_backdoor_adjustment_set
from crca_core.identify.frontdoor import find_frontdoor_mediator
from crca_core.identify.graph import CausalGraph
from crca_core.identify.id_algorithm import id_algorithm
from crca_core.identify.iv import find_instrument
from crca_core.models.provenance import ProvenanceManifest
from crca_core.models.refusal import RefusalChecklistItem, RefusalReasonCode, RefusalResult
from crca_core.models.result import IdentificationResult
from crca_core.models.spec import LockedSpec
from utils.canonical import stable_hash


def identify_effect(
    *,
    locked_spec: LockedSpec,
    treatment: str,
    outcome: str,
) -> IdentificationResult | RefusalResult:
    """Identify an effect using in-house methods (backdoor/frontdoor/IV/ID)."""
    if not treatment or not outcome:
        return RefusalResult(
            message="Treatment and outcome must be provided.",
            reason_codes=[RefusalReasonCode.INPUT_INVALID],
            checklist=[
                RefusalChecklistItem(item="Provide treatment", rationale="Required to define the estimand."),
                RefusalChecklistItem(item="Provide outcome", rationale="Required to define the estimand."),
            ],
            suggested_next_steps=["Pass treatment='X', outcome='Y'."],
        )

    graph = CausalGraph.from_spec(locked_spec.graph)
    prov = ProvenanceManifest.minimal(
        spec_hash=stable_hash(
            {
                "spec_hash": locked_spec.spec_hash,
                "treatment": treatment,
                "outcome": outcome,
                "module": "identify_effect",
            }
        )
    )

    # 1) Backdoor
    z = find_backdoor_adjustment_set(graph, treatment, outcome)
    if z is not None:
        expr = f"sum_{{z}} P({outcome}|{treatment},z) P(z)"
        return IdentificationResult(
            provenance=prov,
            method="backdoor",
            scope="partial",
            confidence="medium",
            estimand_expression=expr,
            assumptions_used=[
                "Backdoor criterion holds with the returned adjustment set.",
                "No unmeasured confounding conditional on Z.",
                "Positivity/overlap for adjustment set.",
            ],
            witnesses={"adjustment_set": sorted(list(z))},
            proof={
                "type": "do-calculus",
                "steps": [
                    "In G_{X̄}, Z d-separates X and Y (backdoor).",
                    "Apply Rule 2 to replace do(X) with observe(X) given Z.",
                ],
            },
            limitations=["Identification assumes all confounding is captured by Z."],
        )

    # 2) Frontdoor
    mediator = find_frontdoor_mediator(
        graph, treatment, outcome, mediators=locked_spec.roles.mediators
    )
    if mediator is not None:
        expr = (
            f"sum_m P(m|{treatment}) sum_{treatment} P({outcome}|m,{treatment}) P({treatment})"
        )
        return IdentificationResult(
            provenance=prov,
            method="frontdoor",
            scope="partial",
            confidence="medium",
            estimand_expression=expr,
            assumptions_used=[
                "Frontdoor criterion holds for mediator M.",
                "No unmeasured confounding between X and M.",
                "All backdoor paths from M to Y are blocked by X.",
            ],
            witnesses={"mediator": mediator},
            proof={
                "type": "do-calculus",
                "steps": [
                    "Use Rule 3 to exchange do(X) with observe(X) for M→Y component.",
                    "Use Rule 2 to exchange do(X) with observe(X) for X→M component.",
                ],
            },
            limitations=["Frontdoor validity depends on strong mediator assumptions."],
        )

    # 3) Instrumental variable
    instrument = find_instrument(
        graph, treatment, outcome, instruments=locked_spec.roles.instruments
    )
    if instrument is not None:
        expr = "IV estimand (see instrument assumptions)"
        return IdentificationResult(
            provenance=prov,
            method="iv",
            scope="partial",
            confidence="low",
            estimand_expression=expr,
            assumptions_used=[
                "Relevance: Z affects X.",
                "Exclusion: Z affects Y only through X.",
                "Independence: Z independent of unmeasured causes of Y.",
            ],
            witnesses={"instrument": instrument},
            proof={
                "type": "linear-IV",
                "steps": [
                    "Assume linear SCM with exclusion and independence.",
                    "Derive β = Cov(Z,Y)/Cov(Z,X).",
                ],
            },
            limitations=["IV estimand expression is left symbolic; estimator must implement IV."],
        )

    # 4) In-house ID algorithm (conservative)
    id_expr = id_algorithm(graph, treatment, outcome)
    if id_expr is not None:
        method, expr = id_expr
        return IdentificationResult(
            provenance=prov,
            method=method,
            scope="conservative",
            confidence="low",
            estimand_expression=expr,
            assumptions_used=["Causal graph is correct; no latent confounding beyond declared."],
            witnesses={},
            proof={
                "type": "id-algorithm",
                "steps": [
                    "Apply ID recursion on C-components.",
                    "Return g-formula when no bidirected edges.",
                ],
            },
            limitations=[
                "ID algorithm is conservative: may return non-identifiable for some identifiable cases with latent confounding."
            ],
        )

    return RefusalResult(
        message="Effect not identifiable under current graph/assumptions.",
        reason_codes=[RefusalReasonCode.NOT_IDENTIFIABLE],
        checklist=[
            RefusalChecklistItem(
                item="Revise causal model or add interventions/measurements",
                rationale="Identification failed with backdoor/frontdoor/IV/ID checks.",
            )
        ],
        suggested_next_steps=["Use design_intervention() to propose identifying experiments."],
    )


__all__ = ["identify_effect"]

