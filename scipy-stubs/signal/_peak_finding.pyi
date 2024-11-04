from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Literal, TypeAlias, TypedDict, TypeVar, type_check_only

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onpt
from numpy._typing import _ArrayLikeBool_co, _ArrayLikeFloat_co, _ArrayLikeInt_co, _ArrayLikeNumber_co
from scipy._typing import AnyInt, AnyReal

__all__ = ["argrelextrema", "argrelmax", "argrelmin", "find_peaks", "find_peaks_cwt", "peak_prominences", "peak_widths"]

_SCT = TypeVar("_SCT", bound=np.generic)

_Array_n: TypeAlias = npt.NDArray[np.intp]
_Array_n_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.intp]]
_Array_f8: TypeAlias = npt.NDArray[np.float64]
_Array_f8_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
_Mode: TypeAlias = Literal["clip", "wrap"]

_ArgRel: TypeAlias = tuple[_Array_n, ...]
_PeakProminences: TypeAlias = tuple[_Array_f8, _Array_n, _Array_n]
_PeakWidths: TypeAlias = tuple[_Array_f8, _Array_f8, _Array_f8, _Array_f8]
# TODO(jorenham): Narrow down the parameter types (because contravariant)
_WaveletFunc: TypeAlias = (
    Callable[Concatenate[int, float, ...], _ArrayLikeNumber_co]
    | Callable[Concatenate[np.intp, np.float64, ...], _ArrayLikeNumber_co]
)

@type_check_only
class _FindPeaksResultsDict(TypedDict, total=False):
    plateau_sizes: _Array_n_1d
    left_edges: _Array_n_1d
    right_edges: _Array_n_1d

    peak_heights: _Array_f8_1d

    left_thresholds: _Array_f8_1d
    right_thresholds: _Array_f8_1d

    prominences: _Array_f8_1d
    left_bases: _Array_n_1d
    right_bases: _Array_n_1d

    widths: _Array_f8_1d
    width_heights: _Array_f8_1d
    left_ips: _Array_f8_1d
    right_ips: _Array_f8_1d

###

def argrelmin(data: onpt.Array, axis: op.CanIndex = 0, order: AnyInt = 1, mode: _Mode = "clip") -> _ArgRel: ...
def argrelmax(data: onpt.Array, axis: op.CanIndex = 0, order: AnyInt = 1, mode: _Mode = "clip") -> _ArgRel: ...
def argrelextrema(
    data: onpt.Array,
    comparator: Callable[[npt.NDArray[_SCT], npt.NDArray[_SCT]], _ArrayLikeBool_co],
    axis: op.CanIndex = 0,
    order: AnyInt = 1,
    mode: _Mode = "clip",
) -> _ArgRel: ...

#
def peak_prominences(x: npt.ArrayLike, peaks: _ArrayLikeInt_co, wlen: AnyReal | None = None) -> _PeakProminences: ...
def peak_widths(
    x: npt.ArrayLike,
    peaks: _ArrayLikeInt_co,
    rel_height: AnyReal = 0.5,
    prominence_data: _PeakProminences | None = None,
    wlen: AnyReal | None = None,
) -> _PeakWidths: ...

#
def find_peaks(
    x: npt.ArrayLike,
    height: _ArrayLikeFloat_co | Sequence[AnyReal | None] | None = None,
    threshold: _ArrayLikeFloat_co | Sequence[AnyReal | None] | None = None,
    distance: AnyReal | None = None,
    prominence: _ArrayLikeFloat_co | Sequence[AnyReal | None] | None = None,
    width: _ArrayLikeFloat_co | Sequence[AnyReal | None] | None = None,
    wlen: AnyReal | None = None,
    rel_height: AnyReal = 0.5,
    plateau_size: _ArrayLikeInt_co | Sequence[AnyInt | None] | None = None,
) -> tuple[_Array_n_1d, _FindPeaksResultsDict]: ...
def find_peaks_cwt(
    vector: onpt.Array,
    widths: _ArrayLikeFloat_co,
    wavelet: _WaveletFunc | None = None,
    max_distances: npt.NDArray[np.floating[Any] | np.integer[Any]] | None = None,
    gap_thresh: AnyReal | None = None,
    min_length: AnyInt | None = None,
    min_snr: AnyReal = 1,
    noise_perc: AnyReal = 10,
    window_size: AnyInt | None = None,
) -> _Array_n_1d: ...
