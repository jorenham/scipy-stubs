from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy._typing import Untyped

__all__ = ["binned_statistic", "binned_statistic_2d", "binned_statistic_dd"]

_Array_n_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.intp]]
_Array_n_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.intp]]
_Array_f8_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
_Array_uifc_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.number[npt.NBitBase]]]
_Array_uifc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.number[npt.NBitBase]]]
_Array_uifc_nd: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.number[npt.NBitBase]]]

_Statistic: TypeAlias = Literal["mean", "std", "median", "count", "sum", "min", "max"]

class BinnedStatisticResult(NamedTuple):
    statistic: _Array_uifc_1d
    bin_edges: _Array_f8_1d
    binnumber: _Array_n_1d

def binned_statistic(
    x: npt.ArrayLike,
    values: npt.ArrayLike,
    statistic: _Statistic | Callable[[npt.NDArray[np.float64]], np.float64 | float] = "mean",
    bins: int = 10,
    range: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
) -> BinnedStatisticResult: ...

class BinnedStatistic2dResult(NamedTuple):
    statistic: _Array_uifc_2d
    x_edge: _Array_f8_1d
    y_edge: _Array_f8_1d
    binnumber: _Array_n_1d

def binned_statistic_2d(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    values: npt.ArrayLike,
    statistic: _Statistic | Callable[[npt.NDArray[np.float64]], np.float64 | float] = "mean",
    bins: npt.ArrayLike = 10,
    range: tuple[int, int] | None = None,
    expand_binnumbers: bool = False,
) -> BinnedStatistic2dResult: ...

class BinnedStatisticddResult(NamedTuple):
    statistic: _Array_uifc_nd
    bin_edges: list[_Array_f8_1d]
    binnumber: _Array_n_1d | _Array_n_2d

def binned_statistic_dd(
    sample: npt.ArrayLike,
    values: npt.ArrayLike,
    statistic: _Statistic | Callable[[npt.NDArray[np.float64]], np.float64 | float] = "mean",
    bins: npt.ArrayLike = 10,
    range: tuple[int, int] | None = None,
    expand_binnumbers: bool = False,
    binned_statistic_result: BinnedStatisticddResult | None = None,
) -> Untyped: ...
