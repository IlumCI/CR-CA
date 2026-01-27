import pandas as pd

from crca_core.models.refusal import RefusalReasonCode, RefusalResult
from crca_core.models.result import DiscoveryHypothesisResult
from crca_core.timeseries import PCMCIConfig, discover_timeseries_pcmci


def test_discover_timeseries_pcmci_refuses_when_backend_missing() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    res = discover_timeseries_pcmci(df, PCMCIConfig(max_lag=2))
    if isinstance(res, RefusalResult):
        assert RefusalReasonCode.UNSUPPORTED_OPERATION in res.reason_codes
    else:
        assert isinstance(res, DiscoveryHypothesisResult)

