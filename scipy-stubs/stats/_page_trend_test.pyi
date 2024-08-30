from dataclasses import dataclass

from scipy._typing import Untyped
from ._continuous_distns import norm as norm

@dataclass
class PageTrendTestResult:
    statistic: float
    pvalue: float
    method: str

def page_trend_test(data, ranked: bool = False, predicted_ranks: Untyped | None = None, method: str = "auto") -> Untyped: ...

class _PageL:
    all_pmfs: Untyped
    def __init__(self) -> None: ...
    k: Untyped
    def set_k(self, k): ...
    def sf(self, l, n) -> Untyped: ...
    def p_l_k_1(self) -> Untyped: ...
    def pmf(self, l, n) -> Untyped: ...
