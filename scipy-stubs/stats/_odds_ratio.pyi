from scipy._typing import Untyped
from scipy.optimize import brentq as brentq
from scipy.special import ndtri as ndtri
from ._common import ConfidenceInterval as ConfidenceInterval
from ._discrete_distns import nchypergeom_fisher as nchypergeom_fisher

class OddsRatioResult:
    statistic: Untyped
    def __init__(self, _table, _kind, statistic) -> None: ...
    def confidence_interval(self, confidence_level: float = 0.95, alternative: str = "two-sided") -> Untyped: ...

def odds_ratio(table, *, kind: str = "conditional") -> Untyped: ...
