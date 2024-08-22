from ._common import ConfidenceInterval as ConfidenceInterval
from ._discrete_distns import binom as binom
from scipy._typing import Untyped
from scipy.optimize import brentq as brentq
from scipy.special import ndtri as ndtri

class BinomTestResult:
    k: Untyped
    n: Untyped
    alternative: Untyped
    statistic: Untyped
    pvalue: Untyped
    proportion_estimate: Untyped
    def __init__(self, k, n, alternative, statistic, pvalue) -> None: ...
    def proportion_ci(self, confidence_level: float = 0.95, method: str = "exact") -> Untyped: ...

def binomtest(k, n, p: float = 0.5, alternative: str = "two-sided") -> Untyped: ...
