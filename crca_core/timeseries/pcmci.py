"""Time-series causal discovery via Tigramite PCMCI/PCMCI+ (wrap-first).

If Tigramite is not installed, this module returns a structured refusal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from crca_core.models.provenance import ProvenanceManifest
from crca_core.models.refusal import RefusalChecklistItem, RefusalReasonCode, RefusalResult
from crca_core.models.result import DiscoveryHypothesisResult
from utils.canonical import stable_hash


class PCMCIConfig(BaseModel):
    max_lag: int = Field(default=5, ge=1)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    variant: Literal["pcmci", "pcmci_plus"] = "pcmci"
    time_index_column: Optional[str] = None
    assume_sorted: bool = False
    min_samples: int = Field(default=200, ge=20)
    notes: Optional[str] = None


def _tigramite_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("tigramite") is not None
    except Exception:
        return False


def discover_timeseries_pcmci(
    data: Any,
    config: Optional[PCMCIConfig] = None,
    assumptions: Optional[List[str]] = None,
) -> DiscoveryHypothesisResult | RefusalResult:
    cfg = config or PCMCIConfig()
    assumptions = assumptions or []

    # Schema-only signature for provenance (no raw data hashing by default).
    schema_sig: Dict[str, Any] = {}
    try:
        import pandas as pd  # type: ignore

        if isinstance(data, pd.DataFrame):
            schema_sig = {c: str(t) for c, t in data.dtypes.items()}
        else:
            schema_sig = {"type": str(type(data))}
    except Exception:
        schema_sig = {"type": str(type(data))}

    spec_hash = stable_hash({"discovery": "pcmci", "config": cfg.model_dump(), "schema": schema_sig})
    prov = ProvenanceManifest.minimal(
        spec_hash=spec_hash,
        data_hash=stable_hash(schema_sig),
        algorithm_config=cfg.model_dump(),
    )

    if not _tigramite_available():
        return RefusalResult(
            message="Time-series causal discovery backend (tigramite) not available.",
            reason_codes=[RefusalReasonCode.UNSUPPORTED_OPERATION],
            checklist=[
                RefusalChecklistItem(
                    item="Install tigramite",
                    rationale="PCMCI/PCMCI+ discovery is wrap-first; we refuse rather than run unvalidated heuristics.",
                )
            ],
            suggested_next_steps=["pip install tigramite"],
        )

    try:
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
        from tigramite import data_processing as pp  # type: ignore
        from tigramite.pcmci import PCMCI  # type: ignore
        from tigramite.independence_tests import ParCorr  # type: ignore
    except Exception as e:
        return RefusalResult(
            message=f"Failed to import tigramite: {e}",
            reason_codes=[RefusalReasonCode.UNSUPPORTED_OPERATION],
            checklist=[
                RefusalChecklistItem(
                    item="Install tigramite",
                    rationale="PCMCI/PCMCI+ discovery requires tigramite backend.",
                )
            ],
            suggested_next_steps=["pip install tigramite"],
        )

    if not isinstance(data, pd.DataFrame):
        return RefusalResult(
            message="PCMCI requires pandas DataFrame input.",
            reason_codes=[RefusalReasonCode.INPUT_INVALID],
            checklist=[
                RefusalChecklistItem(
                    item="Provide pandas DataFrame",
                    rationale="Tigramite expects tabular time-indexed data.",
                )
            ],
            suggested_next_steps=["Convert your data to pandas.DataFrame and retry."],
        )

    df = data.copy()
    if cfg.time_index_column:
        if cfg.time_index_column not in df.columns:
            return RefusalResult(
                message="time_index_column not found in data.",
                reason_codes=[RefusalReasonCode.TIME_INDEX_INVALID],
                checklist=[
                    RefusalChecklistItem(
                        item="Provide valid time index column",
                        rationale="PCMCI requires a consistent time index.",
                    )
                ],
                suggested_next_steps=["Set PCMCIConfig.time_index_column to a valid column."],
            )
        df = df.sort_values(cfg.time_index_column)
    elif not cfg.assume_sorted:
        # If no column specified, require index monotonicity
        try:
            if not df.index.is_monotonic_increasing:
                return RefusalResult(
                    message="DataFrame index is not monotonic; provide a time index column or sort data.",
                    reason_codes=[RefusalReasonCode.TIME_INDEX_INVALID],
                    checklist=[
                        RefusalChecklistItem(
                            item="Provide time index or sorted data",
                            rationale="PCMCI requires time-ordered samples.",
                        )
                    ],
                    suggested_next_steps=["Sort by time or set PCMCIConfig.time_index_column."],
                )
        except Exception:
            pass

    values = df.to_numpy(dtype=float)
    if values.shape[0] < cfg.min_samples:
        return RefusalResult(
            message="Insufficient samples for PCMCI discovery.",
            reason_codes=[RefusalReasonCode.INPUT_INVALID],
            checklist=[
                RefusalChecklistItem(
                    item="Increase sample size",
                    rationale=f"Need at least {cfg.min_samples} rows for stable lagged discovery.",
                )
            ],
            suggested_next_steps=["Collect more samples or lower min_samples (not recommended)."],
        )
    dataframe = pp.DataFrame(values)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())

    if cfg.variant == "pcmci_plus":
        results = pcmci.run_pcmciplus(tau_max=cfg.max_lag, pc_alpha=cfg.alpha)
    else:
        results = pcmci.run_pcmci(tau_max=cfg.max_lag, pc_alpha=cfg.alpha)

    graph_hypothesis = {
        "graph_type": "pcmci",
        "method": cfg.variant,
        "max_lag": cfg.max_lag,
        "graph": results.get("graph"),
        "val_matrix": results.get("val_matrix"),
    }

    return DiscoveryHypothesisResult(
        provenance=prov,
        assumptions=assumptions,
        limitations=[
            "Time-series discovery is hypothesis generation under explicit assumptions (stationarity, lag sufficiency, no hidden confounding, etc.).",
            "Returned graph depends on CI test assumptions and lag selection.",
        ],
        graph_hypothesis=graph_hypothesis,
        stability_report={"alpha": cfg.alpha},
    )

