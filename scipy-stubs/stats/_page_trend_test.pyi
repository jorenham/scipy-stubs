from dataclasses import dataclass

from scipy._typing import Untyped

@dataclass
class PageTrendTestResult:
    statistic: float
    pvalue: float
    method: str

class _PageL:
    all_pmfs: Untyped
    def __init__(self) -> None: ...
    k: Untyped
    def set_k(self, k: Untyped) -> None: ...
    def sf(self, l: Untyped, n: Untyped) -> Untyped: ...
    def p_l_k_1(self) -> Untyped: ...
    def pmf(self, l: Untyped, n: Untyped) -> Untyped: ...

def page_trend_test(
    data: Untyped,
    ranked: bool = False,
    predicted_ranks: Untyped | None = None,
    method: str = "auto",
) -> Untyped: ...
