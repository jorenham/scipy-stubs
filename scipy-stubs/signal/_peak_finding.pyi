from collections.abc import Callable, Sequence
from typing import Concatenate, Literal, TypeAlias, TypedDict
from typing_extensions import NotRequired

import numpy as np
import numpy.typing as npt

__all__ = ["argrelextrema", "argrelmax", "argrelmin", "find_peaks", "find_peaks_cwt", "peak_prominences", "peak_widths"]

_Mode: TypeAlias = Literal["clip", "wrap"]
_ProminencesResult: TypeAlias = tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.intp]]
_WidthsResult: TypeAlias = tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]

class _FindPeaksResultsDict(TypedDict):
    peak_heights: NotRequired[npt.NDArray[np.float64]]
    left_thresholds: NotRequired[npt.NDArray[np.float64]]
    right_thresholds: NotRequired[npt.NDArray[np.float64]]
    prominences: NotRequired[npt.NDArray[np.float64]]
    left_bases: NotRequired[npt.NDArray[np.intp]]
    right_bases: NotRequired[npt.NDArray[np.intp]]
    width_heights: NotRequired[npt.NDArray[np.float64]]
    left_ips: NotRequired[npt.NDArray[np.float64]]
    right_ips: NotRequired[npt.NDArray[np.float64]]
    plateau_sizes: NotRequired[npt.NDArray[np.intp]]
    left_edges: NotRequired[npt.NDArray[np.intp]]
    right_edges: NotRequired[npt.NDArray[np.intp]]

def argrelmin(
    data: npt.NDArray[np.generic], axis: int = 0, order: int = 1, mode: _Mode = "clip"
) -> tuple[npt.NDArray[np.intp], ...]: ...
def argrelmax(
    data: npt.NDArray[np.generic], axis: int = 0, order: int = 1, mode: _Mode = "clip"
) -> tuple[npt.NDArray[np.intp], ...]: ...
def argrelextrema(
    data: npt.NDArray[np.generic],
    comparator: Callable[[npt.NDArray[np.generic], npt.NDArray[np.generic]], npt.NDArray[np.bool_]],
    axis: int = 0,
    order: int = 1,
    mode: _Mode = "clip",
) -> tuple[npt.NDArray[np.intp], ...]: ...
def peak_prominences(x: npt.ArrayLike, peaks: npt.ArrayLike, wlen: int | None = None) -> _ProminencesResult: ...
def peak_widths(
    x: npt.ArrayLike,
    peaks: npt.ArrayLike,
    rel_height: float = 0.5,
    prominence_data: _ProminencesResult | None = None,
    wlen: int | None = None,
) -> _WidthsResult: ...
def find_peaks(
    x: npt.ArrayLike,
    height: float | npt.NDArray[np.float64] | tuple[float | None, float | None] | None = None,
    threshold: float | npt.NDArray[np.float64] | tuple[float | None, float | None] | None = None,
    distance: np.float64 | None = None,
    prominence: float | npt.NDArray[np.float64] | tuple[float | None, float | None] | None = None,
    width: float | npt.NDArray[np.float64] | tuple[float | None, float | None] | None = None,
    wlen: int | None = None,
    rel_height: float = 0.5,
    plateau_size: int | npt.NDArray[np.intp] | tuple[int | None, int | None] | None = None,
) -> tuple[npt.NDArray[np.intp], _FindPeaksResultsDict]: ...
def find_peaks_cwt(
    vector: npt.ArrayLike,
    widths: npt.ArrayLike,
    wavelet: Callable[Concatenate[int, float, ...], npt.NDArray[np.float64]] | None = None,
    max_distances: Sequence[int] | None = None,
    gap_thresh: float | None = None,
    min_length: int | None = None,
    min_snr: int = 1,
    noise_perc: int = 10,
    window_size: int | None = None,
) -> npt.NDArray[np.intp]: ...
