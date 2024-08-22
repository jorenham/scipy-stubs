from typing import NamedTuple

from scipy._typing import Untyped

class BinnedStatisticResult(NamedTuple):
    statistic: Untyped
    bin_edges: Untyped
    binnumber: Untyped

def binned_statistic(x, values, statistic: str = "mean", bins: int = 10, range: Untyped | None = None) -> Untyped: ...

class BinnedStatistic2dResult(NamedTuple):
    statistic: Untyped
    x_edge: Untyped
    y_edge: Untyped
    binnumber: Untyped

def binned_statistic_2d(
    x, y, values, statistic: str = "mean", bins: int = 10, range: Untyped | None = None, expand_binnumbers: bool = False
) -> Untyped: ...

class BinnedStatisticddResult(NamedTuple):
    statistic: Untyped
    bin_edges: Untyped
    binnumber: Untyped

def binned_statistic_dd(
    sample,
    values,
    statistic: str = "mean",
    bins: int = 10,
    range: Untyped | None = None,
    expand_binnumbers: bool = False,
    binned_statistic_result: Untyped | None = None,
) -> Untyped: ...
