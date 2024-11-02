from collections.abc import Callable
from typing import Any, Concatenate, Literal, TypeAlias, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt
import optype as op
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt_co
from scipy._typing import AnyInt, AnyReal

__all__ = ["argrelextrema", "argrelmax", "argrelmin", "find_peaks", "find_peaks_cwt", "peak_prominences", "peak_widths"]

_SCT = TypeVar("_SCT", bound=np.generic)

_Array_n: TypeAlias = npt.NDArray[np.intp]
_Array_n_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.intp]]
_Array_f8: TypeAlias = npt.NDArray[np.float64]
_Mode: TypeAlias = Literal["clip", "wrap"]

_ProminencesResult: TypeAlias = tuple[_Array_f8, _Array_n, _Array_n]
_WidthsResult: TypeAlias = tuple[_Array_f8, _Array_f8, _Array_f8, _Array_f8]
_WaveletOutput: TypeAlias = op.CanGetitem[slice, npt.ArrayLike]
_WaveletFunction: TypeAlias = Callable[Concatenate[int | np.int_ | np.float64, AnyReal, ...], _WaveletOutput]

class _FindPeaksResultsDict(TypedDict, total=False):
    peak_heights: _Array_f8
    left_thresholds: _Array_f8
    right_thresholds: _Array_f8
    prominences: _Array_f8
    left_bases: _Array_n
    right_bases: _Array_n
    width_heights: _Array_f8
    left_ips: _Array_f8
    right_ips: _Array_f8
    plateau_sizes: _Array_n
    left_edges: _Array_n
    right_edges: _Array_n

def argrelmin(
    data: npt.NDArray[np.generic],
    axis: op.CanIndex = 0,
    order: AnyInt = 1,
    mode: _Mode = "clip",
) -> tuple[_Array_n, ...]: ...
def argrelmax(
    data: npt.NDArray[np.generic],
    axis: op.CanIndex = 0,
    order: AnyInt = 1,
    mode: _Mode = "clip",
) -> tuple[_Array_n, ...]: ...
def argrelextrema(
    data: npt.NDArray[np.generic],
    comparator: Callable[[npt.NDArray[_SCT], npt.NDArray[_SCT]], npt.NDArray[np.bool_]],
    axis: op.CanIndex = 0,
    order: AnyInt = 1,
    mode: _Mode = "clip",
) -> tuple[_Array_n, ...]: ...
def peak_prominences(
    x: npt.ArrayLike,
    peaks: _ArrayLikeInt_co,
    wlen: AnyReal | None = None,
) -> _ProminencesResult: ...
def peak_widths(
    x: npt.ArrayLike,
    peaks: _ArrayLikeInt_co,
    rel_height: AnyReal = 0.5,
    prominence_data: _ProminencesResult | None = None,
    wlen: AnyReal | None = None,
) -> _WidthsResult: ...
def find_peaks(
    x: npt.ArrayLike,
    height: _ArrayLikeFloat_co | tuple[AnyReal | None, AnyReal | None] | None = None,
    threshold: _ArrayLikeFloat_co | tuple[AnyReal | None, AnyReal | None] | None = None,
    distance: AnyReal | None = None,
    prominence: _ArrayLikeFloat_co | tuple[AnyReal | None, AnyReal | None] | None = None,
    width: _ArrayLikeFloat_co | tuple[AnyReal | None, AnyReal | None] | None = None,
    wlen: AnyReal | None = None,
    rel_height: AnyReal = 0.5,
    plateau_size: _ArrayLikeInt_co | tuple[AnyInt | None, AnyInt | None] | None = None,
) -> tuple[_Array_n, _FindPeaksResultsDict]: ...
def find_peaks_cwt(
    vector: npt.NDArray[np.generic],
    widths: _ArrayLikeFloat_co,
    wavelet: _WaveletFunction | None = None,
    max_distances: npt.NDArray[np.floating[Any] | np.integer[Any]] | None = None,
    gap_thresh: AnyReal | None = None,
    min_length: AnyInt | None = None,
    min_snr: AnyReal = 1,
    noise_perc: AnyReal = 10,
    window_size: AnyInt | None = None,
) -> _Array_n_1d: ...
