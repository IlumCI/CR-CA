import numpy as np
import pandas as pd

from crca_core import DraftSpec, EstimatorConfig, estimate_effect_dowhy, lock_spec
from crca_core.identify import identify_effect
from crca_core.models.spec import CausalGraphSpec, EdgeSpec, NodeSpec, RoleSpec
from crca_core.models.result import EstimateResult


def test_estimate_effect_dowhy_runs_on_simple_linear_model() -> None:
    # Generate data: Y = 3*X + noise
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(0, 1, size=n)
    Y = 3.0 * X + rng.normal(0, 1, size=n)
    df = pd.DataFrame({"X": X, "Y": Y})

    draft = DraftSpec(
        graph=CausalGraphSpec(nodes=[NodeSpec(name="X"), NodeSpec(name="Y")], edges=[EdgeSpec(source="X", target="Y")]),
        roles=RoleSpec(treatments=["X"], outcomes=["Y"]),
    )
    locked = lock_spec(draft, approvals=["human"])

    ident = identify_effect(locked_spec=locked, treatment="X", outcome="Y")
    res = estimate_effect_dowhy(
        data=df,
        locked_spec=locked,
        treatment="X",
        outcome="Y",
        identification_result=ident,
        config=EstimatorConfig(method_name="backdoor.linear_regression"),
    )
    assert isinstance(res, EstimateResult)
    assert "value" in res.estimate
    assert np.isfinite(res.estimate["value"])

