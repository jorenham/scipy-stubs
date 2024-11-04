from collections.abc import Callable
from typing import Literal, TypeAlias, overload
from typing_extensions import Unpack

import numpy as np
import numpy.typing as npt
import optype as op
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeNumber_co
from scipy._typing import AnyInt, AnyReal
from .windows._windows import _Window, _WindowNeedsParams

__all__ = ["check_COLA", "check_NOLA", "coherence", "csd", "istft", "lombscargle", "periodogram", "spectrogram", "stft", "welch"]

_Array_f8: TypeAlias = npt.NDArray[np.float64]
_Array_f8_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
_ArrayFloat: TypeAlias = npt.NDArray[np.float32 | np.float64 | np.longdouble]
_ArrayComplex: TypeAlias = npt.NDArray[np.complex64 | np.complex128 | np.clongdouble]

_GetWindowArgument: TypeAlias = _Window | tuple[_Window | _WindowNeedsParams, Unpack[tuple[object, ...]]]
_WindowLike: TypeAlias = _GetWindowArgument | _ArrayLikeFloat_co
_Detrend: TypeAlias = Literal["literal", "constant", False] | Callable[[npt.NDArray[np.generic]], npt.NDArray[np.generic]]
_Scaling: TypeAlias = Literal["density", "spectrum"]
_LegacyScaling: TypeAlias = Literal["psd", "spectrum"]
_Average: TypeAlias = Literal["mean", "median"]
_Boundary: TypeAlias = Literal["even", "odd", "constant", "zeros"] | None

def lombscargle(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    freqs: _ArrayLikeFloat_co,
    precenter: op.CanBool = False,
    normalize: op.CanBool = False,
) -> _Array_f8_1d: ...
def periodogram(
    x: _ArrayLikeNumber_co,
    fs: AnyReal = 1.0,
    window: _WindowLike | None = "boxcar",
    nfft: AnyInt | None = None,
    detrend: _Detrend = "constant",
    return_onesided: op.CanBool = True,
    scaling: _Scaling = "density",
    axis: op.CanIndex = -1,
) -> tuple[_Array_f8, _ArrayFloat]: ...
def welch(
    x: _ArrayLikeNumber_co,
    fs: AnyReal = 1.0,
    window: _WindowLike = "hann",
    nperseg: AnyInt | None = None,
    noverlap: AnyInt | None = None,
    nfft: AnyInt | None = None,
    detrend: _Detrend = "constant",
    return_onesided: op.CanBool = True,
    scaling: _Scaling = "density",
    axis: op.CanIndex = -1,
    average: _Average = "mean",
) -> tuple[_Array_f8, _ArrayFloat]: ...
def csd(
    x: _ArrayLikeNumber_co,
    y: _ArrayLikeNumber_co,
    fs: AnyReal = 1.0,
    window: _WindowLike = "hann",
    nperseg: AnyInt | None = None,
    noverlap: AnyInt | None = None,
    nfft: AnyInt | None = None,
    detrend: _Detrend = "constant",
    return_onesided: op.CanBool = True,
    scaling: _Scaling = "density",
    axis: op.CanIndex = -1,
    average: _Average = "mean",
) -> tuple[_Array_f8, _ArrayComplex]: ...

#
@overload
# non-complex mode (positional and keyword)
def spectrogram(
    x: _ArrayLikeNumber_co,
    fs: AnyReal = 1.0,
    window: _WindowLike = ("tukey", 0.25),
    nperseg: AnyInt | None = None,
    noverlap: AnyInt | None = None,
    nfft: AnyInt | None = None,
    detrend: _Detrend = "constant",
    return_onesided: op.CanBool = True,
    scaling: _Scaling = "density",
    axis: op.CanIndex = -1,
    mode: Literal["psd", "magnitude", "angle", "phase"] = "psd",
) -> tuple[_Array_f8, _Array_f8, _ArrayFloat]: ...
@overload
# complex mode (positional)
def spectrogram(
    x: _ArrayLikeNumber_co,
    fs: AnyReal,
    window: _WindowLike,
    nperseg: AnyInt | None,
    noverlap: AnyInt | None,
    nfft: AnyInt | None,
    detrend: _Detrend,
    return_onesided: op.CanBool,
    scaling: _Scaling,
    axis: op.CanIndex,
    mode: Literal["complex"],
) -> tuple[_Array_f8, _Array_f8, _ArrayComplex]: ...
@overload
# complex mode (keyword)
def spectrogram(
    x: _ArrayLikeNumber_co,
    fs: AnyReal = 1.0,
    window: _WindowLike = ("tukey", 0.25),
    nperseg: AnyInt | None = None,
    noverlap: AnyInt | None = None,
    nfft: AnyInt | None = None,
    detrend: _Detrend = "constant",
    return_onesided: op.CanBool = True,
    scaling: _Scaling = "density",
    axis: op.CanIndex = -1,
    *,
    mode: Literal["complex"],
) -> tuple[_Array_f8, _Array_f8, _ArrayComplex]: ...

#
def check_COLA(
    window: _WindowLike,
    nperseg: AnyInt,
    noverlap: AnyInt,
    tol: AnyReal = 1e-10,
) -> np.bool_: ...
def check_NOLA(
    window: _WindowLike,
    nperseg: AnyInt,
    noverlap: AnyInt,
    tol: AnyReal = 1e-10,
) -> np.bool_: ...
def stft(
    x: _ArrayLikeNumber_co,
    fs: AnyReal = 1.0,
    window: _WindowLike = "hann",
    nperseg: AnyInt = 256,
    noverlap: AnyInt | None = None,
    nfft: AnyInt | None = None,
    detrend: _Detrend = False,
    return_onesided: op.CanBool = True,
    boundary: _Boundary = "zeros",
    padded: op.CanBool = True,
    axis: op.CanIndex = -1,
    scaling: _LegacyScaling = "spectrum",
) -> tuple[_Array_f8, _Array_f8, _ArrayComplex]: ...

#
@overload
# input_onesided is `True`
def istft(
    Zxx: _ArrayLikeNumber_co,
    fs: AnyReal = 1.0,
    window: _WindowLike = "hann",
    nperseg: AnyInt | None = None,
    noverlap: AnyInt | None = None,
    nfft: AnyInt | None = None,
    input_onesided: Literal[True, 1] = True,
    boundary: op.CanBool = True,
    time_axis: op.CanIndex = -1,
    freq_axis: op.CanIndex = -2,
    scaling: _LegacyScaling = "spectrum",
) -> tuple[_Array_f8, _ArrayFloat]: ...
@overload
# input_onesided is `False` (positional)
def istft(
    Zxx: _ArrayLikeNumber_co,
    fs: AnyReal,
    window: _WindowLike,
    nperseg: AnyInt | None,
    noverlap: AnyInt | None,
    nfft: AnyInt | None,
    input_onesided: Literal[False, 0],
    boundary: op.CanBool = True,
    time_axis: op.CanIndex = -1,
    freq_axis: op.CanIndex = -2,
    scaling: _LegacyScaling = "spectrum",
) -> tuple[_Array_f8, _ArrayComplex]: ...
@overload
# input_onesided is `False` (keyword)
def istft(
    Zxx: _ArrayLikeNumber_co,
    fs: AnyReal = 1.0,
    window: _WindowLike = "hann",
    nperseg: AnyInt | None = None,
    noverlap: AnyInt | None = None,
    nfft: AnyInt | None = None,
    *,
    input_onesided: Literal[False, 0],
    boundary: op.CanBool = True,
    time_axis: op.CanIndex = -1,
    freq_axis: op.CanIndex = -2,
    scaling: _LegacyScaling = "spectrum",
) -> tuple[_Array_f8, _ArrayComplex]: ...

#
def coherence(
    x: _ArrayLikeNumber_co,
    y: _ArrayLikeNumber_co,
    fs: AnyReal = 1.0,
    window: _WindowLike = "hann",
    nperseg: AnyInt | None = None,
    noverlap: AnyInt | None = None,
    nfft: AnyInt | None = None,
    detrend: _Detrend = "constant",
    axis: op.CanIndex = -1,
) -> tuple[_Array_f8, _ArrayFloat]: ...
