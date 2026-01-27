import pandas as pd

from crca_core.discovery import TabularDiscoveryConfig, discover_tabular
from crca_core.models.refusal import RefusalResult, RefusalReasonCode
from crca_core.models.result import DiscoveryHypothesisResult


def test_discover_tabular_refuses_when_backend_missing() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    res = discover_tabular(df, TabularDiscoveryConfig(algorithm="pc"))
    if isinstance(res, RefusalResult):
        assert RefusalReasonCode.UNSUPPORTED_OPERATION in res.reason_codes
    else:
        assert isinstance(res, DiscoveryHypothesisResult)

