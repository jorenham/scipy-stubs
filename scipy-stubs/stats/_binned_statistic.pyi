from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = ["binned_statistic", "binned_statistic_2d", "binned_statistic_dd"]

_Statistic: TypeAlias = Literal["mean", "std", "median", "count", "sum", "min", "max"]

class BinnedStatisticResult(NamedTuple):
    statistic: onp.Array1D[np.inexact[Any]]
    bin_edges: onp.Array1D[np.float64]
    binnumber: onp.Array1D[np.intp]

def binned_statistic(
    x: npt.ArrayLike,
    values: npt.ArrayLike,
    statistic: _Statistic | Callable[[onp.ArrayND[np.float64]], np.float64 | float] = "mean",
    bins: int = 10,
    range: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
) -> BinnedStatisticResult: ...

class BinnedStatistic2dResult(NamedTuple):
    statistic: onp.Array2D[np.inexact[Any]]
    x_edge: onp.Array1D[np.float64]
    y_edge: onp.Array1D[np.float64]
    binnumber: onp.Array1D[np.intp]

def binned_statistic_2d(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    values: npt.ArrayLike,
    statistic: _Statistic | Callable[[onp.ArrayND[np.float64]], np.float64 | float] = "mean",
    bins: npt.ArrayLike = 10,
    range: tuple[int, int] | None = None,
    expand_binnumbers: bool = False,
) -> BinnedStatistic2dResult: ...

class BinnedStatisticddResult(NamedTuple):
    statistic: onp.ArrayND[np.inexact[Any]]
    bin_edges: list[onp.Array1D[np.float64]]
    binnumber: onp.Array1D[np.intp] | onp.Array2D[np.intp]

def binned_statistic_dd(
    sample: npt.ArrayLike,
    values: npt.ArrayLike,
    statistic: _Statistic | Callable[[onp.ArrayND[np.float64]], np.float64 | float] = "mean",
    bins: npt.ArrayLike = 10,
    range: tuple[int, int] | None = None,
    expand_binnumbers: bool = False,
    binned_statistic_result: BinnedStatisticddResult | None = None,
) -> BinnedStatisticddResult: ...
