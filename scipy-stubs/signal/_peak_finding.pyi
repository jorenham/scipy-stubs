from collections.abc import Callable, Sequence
from typing import Concatenate, Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
import optype as op
from numpy._typing import _ArrayLikeInt_co
from scipy._typing import AnyInt, AnyReal

__all__ = ["argrelextrema", "argrelmax", "argrelmin", "find_peaks", "find_peaks_cwt", "peak_prominences", "peak_widths"]

_Array_n: TypeAlias = npt.NDArray[np.intp]
_Array_f8: TypeAlias = npt.NDArray[np.float64]
_Mode: TypeAlias = Literal["clip", "wrap"]

_ProminencesResult: TypeAlias = tuple[_Array_f8, _Array_n, _Array_n]
_WidthsResult: TypeAlias = tuple[_Array_f8, _Array_f8, _Array_f8, _Array_f8]

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
    order: int = 1,
    mode: _Mode = "clip",
) -> tuple[_Array_n, ...]: ...
def argrelmax(
    data: npt.NDArray[np.generic],
    axis: op.CanIndex = 0,
    order: int = 1,
    mode: _Mode = "clip",
) -> tuple[_Array_n, ...]: ...
def argrelextrema(
    data: npt.NDArray[np.generic],
    comparator: Callable[[npt.NDArray[np.generic], npt.NDArray[np.generic]], npt.NDArray[np.bool_]],
    axis: op.CanIndex = 0,
    order: int = 1,
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
    height: AnyReal | _Array_f8 | tuple[AnyReal | None, AnyReal | None] | None = None,
    threshold: AnyReal | _Array_f8 | tuple[AnyReal | None, AnyReal | None] | None = None,
    distance: AnyReal | None = None,
    prominence: AnyReal | _Array_f8 | tuple[AnyReal | None, AnyReal | None] | None = None,
    width: AnyReal | _Array_f8 | tuple[AnyReal | None, AnyReal | None] | None = None,
    wlen: AnyReal | None = None,
    rel_height: AnyReal = 0.5,
    plateau_size: AnyInt | _Array_n | tuple[AnyInt | None, AnyInt | None] | None = None,
) -> tuple[_Array_n, _FindPeaksResultsDict]: ...
def find_peaks_cwt(
    vector: npt.ArrayLike,
    widths: npt.ArrayLike,
    wavelet: Callable[Concatenate[int, float, ...], _Array_f8] | None = None,
    max_distances: Sequence[int] | None = None,
    gap_thresh: AnyInt | None = None,
    min_length: AnyInt | None = None,
    min_snr: AnyReal = 1,
    noise_perc: AnyReal = 10,
    window_size: AnyInt | None = None,
) -> npt.NDArray[np.intp]: ...
