from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy._typing import Untyped

__all__ = ["argrelextrema", "argrelmax", "argrelmin", "find_peaks", "find_peaks_cwt", "peak_prominences", "peak_widths"]

_Mode: TypeAlias = Literal["clip", "wrap"]
_ProminencesResult: TypeAlias = tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.intp]]
_WidthsResult: TypeAlias = tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]

def argrelmin(
    data: npt.NDArray[np.generic], axis: int = 0, order: int = 1, mode: _Mode = "clip"
) -> tuple[npt.NDArray[np.intp], ...]: ...
def argrelmax(data: Untyped, axis: int = 0, order: int = 1, mode: _Mode = "clip") -> tuple[npt.NDArray[np.intp], ...]: ...
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
    x: Untyped,
    height: Untyped | None = None,
    threshold: Untyped | None = None,
    distance: Untyped | None = None,
    prominence: Untyped | None = None,
    width: Untyped | None = None,
    wlen: Untyped | None = None,
    rel_height: float = 0.5,
    plateau_size: Untyped | None = None,
) -> Untyped: ...
def find_peaks_cwt(
    vector: Untyped,
    widths: Untyped,
    wavelet: Untyped | None = None,
    max_distances: Untyped | None = None,
    gap_thresh: Untyped | None = None,
    min_length: Untyped | None = None,
    min_snr: int = 1,
    noise_perc: int = 10,
    window_size: Untyped | None = None,
) -> Untyped: ...
