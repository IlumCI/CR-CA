"""crca_core: research-grade causal science core (H1).

This package is intentionally strict:
- No numeric causal outputs without a LockedSpec and identifiability checks.
- Counterfactuals require an explicit SCM (structural equations + noise model).
- All outputs are structured (Pydantic) and carry provenance.
"""

from crca_core.models.spec import DraftSpec, LockedSpec
from crca_core.models.result import (
    BaseResult,
    RefusalResult,
    ValidationReport,
)
from crca_core.core.lifecycle import lock_spec
from crca_core.models.validation import validate_spec
from crca_core.core.estimate import estimate_effect_dowhy, EstimatorConfig
from crca_core.core.api import simulate_counterfactual, identify_effect
from crca_core.core.godclass import CausalCoreGod

__all__ = [
    "DraftSpec",
    "LockedSpec",
    "BaseResult",
    "RefusalResult",
    "ValidationReport",
    "lock_spec",
    "validate_spec",
    "EstimatorConfig",
    "estimate_effect_dowhy",
    "simulate_counterfactual",
    "identify_effect",
    "CausalCoreGod",
]

