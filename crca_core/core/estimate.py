"""Identification → estimation → refutation wrapper (DoWhy).

This is supporting infrastructure for causal R&D. It is gated behind:
- LockedSpec
- explicit treatment/outcome
- DoWhy identification success

Refuters do not prove causality; they are diagnostics and must be surfaced.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from crca_core.models.provenance import ProvenanceManifest
from crca_core.models.refusal import RefusalChecklistItem, RefusalReasonCode, RefusalResult
from crca_core.models.result import EstimateResult, IdentificationResult
from crca_core.models.spec import LockedSpec
from utils.canonical import stable_hash


class EstimatorConfig(BaseModel):
    method_name: str = Field(default="backdoor.linear_regression")
    test_significance: bool = True
    confidence_intervals: bool = True
    refuters: List[str] = Field(
        default_factory=lambda: ["placebo_treatment_refuter", "random_common_cause", "subset_refuter"]
    )


def _ensure_networkx_compat() -> None:
    """Patch NetworkX API differences required by DoWhy.

    DoWhy versions in the wild have referenced `networkx.algorithms.d_separated`,
    while NetworkX >=3.6 provides `networkx.algorithms.d_separation.is_d_separator`.
    """
    try:
        import networkx as nx  # type: ignore
        import networkx.algorithms.d_separation as ds  # type: ignore

        if not hasattr(nx.algorithms, "d_separated") and hasattr(ds, "is_d_separator"):
            setattr(nx.algorithms, "d_separated", ds.is_d_separator)
    except Exception:
        # If networkx isn't available, DoWhy will fail later anyway.
        return


def _graph_to_dot(spec: LockedSpec) -> str:
    # Minimal DOT string compatible with DoWhy's graph parser.
    edges = spec.graph.edges
    lines = ["digraph {"]
    for n in spec.graph.nodes:
        lines.append(f'  "{n.name}";')
    for e in edges:
        lines.append(f'  "{e.source}" -> "{e.target}";')
    lines.append("}")
    return "\n".join(lines)


def estimate_effect_dowhy(
    *,
    data: Any,
    locked_spec: LockedSpec,
    treatment: str,
    outcome: str,
    identification_result: IdentificationResult | None = None,
    config: Optional[EstimatorConfig] = None,
) -> EstimateResult | RefusalResult:
    """Run DoWhy identify→estimate→refute and return a structured result."""

    cfg = config or EstimatorConfig()

    # Basic gating
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

    if identification_result is None:
        return RefusalResult(
            message="Estimation requires an IdentificationResult.",
            reason_codes=[RefusalReasonCode.INPUT_INVALID],
            checklist=[
                RefusalChecklistItem(
                    item="Run identify_effect() first",
                    rationale="Estimation is gated behind validated identifiability.",
                )
            ],
            suggested_next_steps=["Call identify_effect() and pass its result here."],
        )

    if identification_result.method == "not_identifiable":
        return RefusalResult(
            message="Cannot estimate: identification failed.",
            reason_codes=[RefusalReasonCode.NOT_IDENTIFIABLE],
            checklist=[
                RefusalChecklistItem(
                    item="Revise causal model or collect additional data/interventions",
                    rationale="Identification result indicates non-identifiability.",
                )
            ],
            suggested_next_steps=["Use design_intervention() to propose identifying experiments."],
        )

    dot = _graph_to_dot(locked_spec)
    prov = ProvenanceManifest.minimal(
        spec_hash=stable_hash(
            {
                "spec_hash": locked_spec.spec_hash,
                "treatment": treatment,
                "outcome": outcome,
                "method": cfg.model_dump(),
                "graph": dot,
                "module": "dowhy_pipeline",
            }
        ),
        algorithm_config=cfg.model_dump(),
    )

    try:
        import pandas as pd  # type: ignore

        if not isinstance(data, pd.DataFrame):
            return RefusalResult(
                message="DoWhy estimation requires a pandas DataFrame.",
                reason_codes=[RefusalReasonCode.INPUT_INVALID],
                checklist=[RefusalChecklistItem(item="Provide pandas DataFrame", rationale="DoWhy expects tabular data.")],
                suggested_next_steps=["Convert your data to pandas.DataFrame and retry."],
            )
    except Exception as e:  # pragma: no cover
        return RefusalResult(
            message=f"pandas is required for estimation: {e}",
            reason_codes=[RefusalReasonCode.UNSUPPORTED_OPERATION],
            checklist=[RefusalChecklistItem(item="Install pandas", rationale="Required dependency for tabular estimation.")],
            suggested_next_steps=["pip install pandas"],
        )

    try:
        _ensure_networkx_compat()
        from dowhy import CausalModel  # type: ignore
    except Exception as e:
        return RefusalResult(
            message=f"DoWhy not available: {e}",
            reason_codes=[RefusalReasonCode.UNSUPPORTED_OPERATION],
            checklist=[RefusalChecklistItem(item="Install dowhy", rationale="Required for this estimation pipeline.")],
            suggested_next_steps=["pip install dowhy"],
        )

    model = CausalModel(data=data, treatment=treatment, outcome=outcome, graph=dot)
    identified_estimand = model.identify_effect()

    # If identification fails, DoWhy usually still returns an object; we gate on its string.
    if identified_estimand is None:
        return RefusalResult(
            message="Causal effect not identifiable under the provided graph/assumptions.",
            reason_codes=[RefusalReasonCode.NOT_IDENTIFIABLE],
            checklist=[
                RefusalChecklistItem(
                    item="Revise the causal model or collect additional data/interventions",
                    rationale="Effect identification failed.",
                )
            ],
            suggested_next_steps=["Use design_intervention() to propose identifying experiments."],
        )

    estimate = model.estimate_effect(
        identified_estimand,
        method_name=cfg.method_name,
        test_significance=cfg.test_significance,
        confidence_intervals=cfg.confidence_intervals,
    )

    refutations: Dict[str, Any] = {}
    for refuter in cfg.refuters:
        try:
            ref = model.refute_estimate(identified_estimand, estimate, method_name=refuter)
            refutations[refuter] = str(ref)
        except Exception as e:
            refutations[refuter] = {"error": str(e)}

    return EstimateResult(
        provenance=prov,
        assumptions=[
            "Causal graph is correctly specified (strong assumption).",
            "Estimator assumptions depend on the chosen method (see DoWhy).",
        ],
        limitations=[
            "Refutation tests are diagnostics; passing does not prove causality.",
            "Estimation quality depends on overlap/positivity and measurement quality.",
        ],
        estimate={
            "value": float(getattr(estimate, "value", float("nan"))),
            "method_name": cfg.method_name,
            "estimand": str(identified_estimand),
            "raw_estimate": str(estimate),
        },
        refutations=refutations,
        artifacts={"identification_result": identification_result.model_dump()},
    )

