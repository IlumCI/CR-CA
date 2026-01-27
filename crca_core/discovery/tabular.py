"""Tabular causal discovery (hypothesis generation).

Design goals:
- Wrap established implementations when available (preferred).
- If required backends are missing, return a structured Refusal (never ad-hoc heuristics).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from crca_core.models.provenance import ProvenanceManifest
from crca_core.models.refusal import RefusalChecklistItem, RefusalReasonCode, RefusalResult
from crca_core.models.result import DiscoveryHypothesisResult
from utils.canonical import stable_hash


class TabularDiscoveryConfig(BaseModel):
    algorithm: Literal["pc", "fci", "ges"] = "pc"
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    bootstrap_samples: int = Field(default=0, ge=0)
    ci_test: Literal["fisherz", "gsq", "chisq"] = "fisherz"
    stable: bool = True
    min_samples: int = Field(default=200, ge=20)
    notes: Optional[str] = None


def _backend_available() -> bool:
    try:
        import importlib.util

        # causal-learn installs as `causallearn`
        return importlib.util.find_spec("causallearn") is not None
    except Exception:
        return False


def discover_tabular(
    data: Any,
    discovery_config: Optional[TabularDiscoveryConfig] = None,
    assumptions: Optional[List[str]] = None,
) -> DiscoveryHypothesisResult | RefusalResult:
    """Run tabular causal discovery and return a hypothesis object.

    Notes:
    - This is hypothesis generation only.
    - If `causal-learn` is not installed, we refuse and provide an actionable checklist.
    """

    cfg = discovery_config or TabularDiscoveryConfig()
    assumptions = assumptions or []

    # Compute a lightweight data hash for provenance (schema-level only).
    # We intentionally do not hash raw data values here.
    schema_sig = {}
    try:
        import pandas as pd  # type: ignore

        if isinstance(data, pd.DataFrame):
            schema_sig = {c: str(t) for c, t in data.dtypes.items()}
        else:
            schema_sig = {"type": str(type(data))}
    except Exception:
        schema_sig = {"type": str(type(data))}

    spec_hash = stable_hash({"discovery": "tabular", "config": cfg.model_dump(), "schema": schema_sig})
    prov = ProvenanceManifest.minimal(
        spec_hash=spec_hash,
        data_hash=stable_hash(schema_sig),
        algorithm_config=cfg.model_dump(),
    )

    if not _backend_available():
        return RefusalResult(
            message="Tabular causal discovery backend not available.",
            reason_codes=[RefusalReasonCode.UNSUPPORTED_OPERATION],
            checklist=[
                RefusalChecklistItem(
                    item="Install causal-learn",
                    rationale="Tabular discovery is wrap-first; we refuse rather than run unvalidated heuristics.",
                )
            ],
            suggested_next_steps=["pip install causal-learn"],
        )

    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        from causallearn.search.ConstraintBased.PC import pc  # type: ignore
        from causallearn.search.ConstraintBased.FCI import fci  # type: ignore
        from causallearn.search.ScoreBased.GES import ges  # type: ignore
    except Exception as e:
        return RefusalResult(
            message=f"Failed to import causal-learn: {e}",
            reason_codes=[RefusalReasonCode.UNSUPPORTED_OPERATION],
            checklist=[
                RefusalChecklistItem(
                    item="Install causal-learn",
                    rationale="Tabular discovery requires causal-learn backend.",
                )
            ],
            suggested_next_steps=["pip install causal-learn"],
        )

    if not isinstance(data, pd.DataFrame):
        return RefusalResult(
            message="Tabular discovery requires pandas DataFrame input.",
            reason_codes=[RefusalReasonCode.INPUT_INVALID],
            checklist=[
                RefusalChecklistItem(
                    item="Provide pandas DataFrame",
                    rationale="Causal-learn expects tabular numpy/pandas data.",
                )
            ],
            suggested_next_steps=["Convert your data to pandas.DataFrame and retry."],
        )

    columns = list(data.columns)
    values = data.to_numpy(dtype=float)
    if values.shape[0] < cfg.min_samples:
        return RefusalResult(
            message="Insufficient samples for reliable discovery.",
            reason_codes=[RefusalReasonCode.INPUT_INVALID],
            checklist=[
                RefusalChecklistItem(
                    item="Increase sample size",
                    rationale=f"Need at least {cfg.min_samples} rows for stable discovery.",
                )
            ],
            suggested_next_steps=["Collect more samples or lower min_samples (not recommended)."],
        )

    def _run_once() -> Dict[str, Any]:
        if cfg.algorithm == "pc":
            cg = pc(values, alpha=cfg.alpha, indep_test=cfg.ci_test, stable=cfg.stable)
            graph_obj = getattr(cg, "G", cg)
        elif cfg.algorithm == "fci":
            res = fci(values, alpha=cfg.alpha, indep_test=cfg.ci_test)
            graph_obj = res[0] if isinstance(res, (list, tuple)) else res
            graph_obj = getattr(graph_obj, "G", graph_obj)
        else:
            res = ges(values)
            graph_obj = res.get("G") if isinstance(res, dict) else res
        mat = getattr(graph_obj, "graph", None)
        if mat is None:
            return {"graph_type": "unknown", "raw": str(graph_obj)}
        return {
            "graph_type": "causal_learn_matrix",
            "adjacency": np.asarray(mat).tolist(),
            "columns": columns,
        }

    graph_hypothesis = _run_once()
    stability_report: Dict[str, Any] = {"bootstrap_samples": cfg.bootstrap_samples}

    if cfg.bootstrap_samples > 0:
        edge_counts = None
        for _ in range(cfg.bootstrap_samples):
            idx = np.random.randint(0, values.shape[0], size=values.shape[0])
            boot_values = values[idx]
            if cfg.algorithm == "pc":
                cg = pc(boot_values, alpha=cfg.alpha, indep_test=cfg.ci_test, stable=cfg.stable)
                graph_obj = getattr(cg, "G", cg)
            elif cfg.algorithm == "fci":
                res = fci(boot_values, alpha=cfg.alpha, indep_test=cfg.ci_test)
                graph_obj = res[0] if isinstance(res, (list, tuple)) else res
                graph_obj = getattr(graph_obj, "G", graph_obj)
            else:
                res = ges(boot_values)
                graph_obj = res.get("G") if isinstance(res, dict) else res
            mat = getattr(graph_obj, "graph", None)
            if mat is None:
                continue
            mat = np.asarray(mat)
            if edge_counts is None:
                edge_counts = np.zeros_like(mat, dtype=float)
            edge_counts += (mat != 0).astype(float)
        if edge_counts is not None and cfg.bootstrap_samples > 0:
            stability_report["edge_frequency"] = (edge_counts / float(cfg.bootstrap_samples)).tolist()

    return DiscoveryHypothesisResult(
        provenance=prov,
        assumptions=assumptions,
        limitations=[
            "Discovery outputs are hypotheses under assumptions (e.g., faithfulness, causal sufficiency/latent handling).",
            "Returned graph structure depends on CI test assumptions and sample size.",
        ],
        graph_hypothesis=graph_hypothesis,
        stability_report=stability_report,
    )

