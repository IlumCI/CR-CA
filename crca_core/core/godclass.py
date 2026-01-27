"""CausalCore GodClass: consolidated core API facade."""

from __future__ import annotations

from typing import Any, Dict, Optional

from crca_core.core.api import (
    EstimatorConfig,
    FeasibilityConstraints,
    PCMCIConfig,
    TabularDiscoveryConfig,
    TargetQuery,
    design_intervention,
    discover_tabular,
    discover_timeseries_pcmci,
    identify_effect,
    estimate_effect_dowhy,
    simulate_counterfactual,
)
from crca_core.models.result import AnyResult, IdentificationResult
from crca_core.models.spec import LockedSpec


class CausalCoreGod:
    """Single class exposing all core causal operations."""

    def __init__(self) -> None:
        self.last_identification: Optional[IdentificationResult] = None

    def identify(self, *, locked_spec: LockedSpec, treatment: str, outcome: str) -> AnyResult:
        res = identify_effect(locked_spec=locked_spec, treatment=treatment, outcome=outcome)
        if isinstance(res, IdentificationResult):
            self.last_identification = res
        return res

    def estimate(self, *, data: Any, locked_spec: LockedSpec, treatment: str, outcome: str) -> AnyResult:
        return estimate_effect_dowhy(
            data=data,
            locked_spec=locked_spec,
            treatment=treatment,
            outcome=outcome,
            identification_result=self.last_identification,
            config=EstimatorConfig(),
        )

    def counterfactual(
        self,
        *,
        locked_spec: LockedSpec,
        factual_observation: Dict[str, float],
        intervention: Dict[str, float],
        allow_partial_observation: bool = False,
    ) -> AnyResult:
        return simulate_counterfactual(
            locked_spec=locked_spec,
            factual_observation=factual_observation,
            intervention=intervention,
            allow_partial_observation=allow_partial_observation,
        )

    def design_intervention(self, *, locked_spec: LockedSpec, target_query: TargetQuery) -> AnyResult:
        return design_intervention(
            locked_spec=locked_spec,
            target_query=target_query,
            constraints=FeasibilityConstraints(),
        )

    def discover_tabular(self, *, data: Any) -> AnyResult:
        return discover_tabular(data, TabularDiscoveryConfig())

    def discover_timeseries(self, *, data: Any) -> AnyResult:
        return discover_timeseries_pcmci(data, PCMCIConfig())
