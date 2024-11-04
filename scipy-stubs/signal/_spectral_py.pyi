from collections.abc import Callable
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import Unpack

import numpy as np
import numpy.typing as npt
import optype as op
from numpy._typing import _ArrayLikeComplex_co, _ArrayLikeFloat_co
from scipy._typing import AnyInt, AnyReal
from scipy.signal.windows._windows import _Window, _WindowNeedsParams

__all__ = ["check_COLA", "check_NOLA", "coherence", "csd", "istft", "lombscargle", "periodogram", "spectrogram", "stft", "welch"]

_Array_f8: TypeAlias = npt.NDArray[np.float64]
_Array_f8_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
_ArrayReal: TypeAlias = npt.NDArray[np.floating[Any]]
_ArrayComplex: TypeAlias = npt.NDArray[np.complexfloating[Any, Any]]

_GetWindowArgument: TypeAlias = _Window | tuple[_Window | _WindowNeedsParams, Unpack[tuple[object, ...]]]
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
    x: _ArrayLikeComplex_co,
    fs: AnyReal = 1.0,
    window: _GetWindowArgument | _ArrayLikeFloat_co | None = "boxcar",
    nfft: AnyInt | None = None,
    detrend: _Detrend = "constant",
    return_onesided: op.CanBool = True,
    scaling: _Scaling = "density",
    axis: op.CanIndex = -1,
) -> tuple[_Array_f8, _ArrayReal]: ...
def welch(
    x: _ArrayLikeComplex_co,
    fs: AnyReal = 1.0,
    window: _GetWindowArgument | _ArrayLikeFloat_co = "hann",
    nperseg: AnyInt | None = None,
    noverlap: AnyInt | None = None,
    nfft: AnyInt | None = None,
    detrend: _Detrend = "constant",
    return_onesided: op.CanBool = True,
    scaling: _Scaling = "density",
    axis: op.CanIndex = -1,
    average: _Average = "mean",
) -> tuple[_Array_f8, _ArrayReal]: ...
def csd(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    fs: AnyReal = 1.0,
    window: _GetWindowArgument | _ArrayLikeFloat_co = "hann",
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
def spectrogram(
    x: npt.NDArray[np.generic],
    fs: AnyReal = 1.0,
    window: _GetWindowArgument | _ArrayLikeFloat_co = ("tukey", 0.25),
    nperseg: AnyInt | None = None,
    noverlap: AnyInt | None = None,
    nfft: AnyInt | None = None,
    detrend: _Detrend = "constant",
    return_onesided: op.CanBool = True,
    scaling: _Scaling = "density",
    axis: op.CanIndex = -1,
    mode: Literal["psd", "magnitude", "angle", "phase"] = "psd",
) -> tuple[_Array_f8, _Array_f8, _ArrayReal]: ...

# complex mode (positional)
@overload
def spectrogram(
    x: npt.NDArray[np.generic],
    fs: AnyReal,
    window: _GetWindowArgument | _ArrayLikeFloat_co,
    nperseg: AnyInt | None,
    noverlap: AnyInt | None,
    nfft: AnyInt | None,
    detrend: _Detrend,
    return_onesided: op.CanBool,
    scaling: _Scaling,
    axis: op.CanIndex,
    mode: Literal["complex"],
) -> tuple[_Array_f8, _Array_f8, _ArrayComplex]: ...

# complex mode (keyword)

@overload
def spectrogram(
    x: npt.NDArray[np.generic],
    fs: AnyReal = 1.0,
    window: _GetWindowArgument | _ArrayLikeFloat_co = ("tukey", 0.25),
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
    window: _GetWindowArgument | _ArrayLikeFloat_co,
    nperseg: AnyInt,
    noverlap: AnyInt,
    tol: AnyReal = 1e-10,
) -> bool: ...
def check_NOLA(
    window: _GetWindowArgument | _ArrayLikeFloat_co,
    nperseg: AnyInt,
    noverlap: AnyInt,
    tol: AnyReal = 1e-10,
) -> bool: ...
def stft(
    x: _ArrayLikeComplex_co,
    fs: AnyReal = 1.0,
    window: _GetWindowArgument | _ArrayLikeFloat_co = "hann",
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

# input_onesided is `True`
@overload
def istft(
    Zxx: _ArrayLikeComplex_co,
    fs: AnyReal = 1.0,
    window: _GetWindowArgument | _ArrayLikeFloat_co = "hann",
    nperseg: AnyInt | None = None,
    noverlap: AnyInt | None = None,
    nfft: AnyInt | None = None,
    input_onesided: Literal[True] = True,
    boundary: op.CanBool = True,
    time_axis: op.CanIndex = -1,
    freq_axis: op.CanIndex = -2,
    scaling: _LegacyScaling = "spectrum",
) -> tuple[_Array_f8, _ArrayReal]: ...

# input_onesided is `False` (positional)
@overload
def istft(
    Zxx: _ArrayLikeComplex_co,
    fs: AnyReal,
    window: _GetWindowArgument | _ArrayLikeFloat_co,
    nperseg: AnyInt | None,
    noverlap: AnyInt | None,
    nfft: AnyInt | None,
    input_onesided: Literal[False],
    boundary: op.CanBool = True,
    time_axis: op.CanIndex = -1,
    freq_axis: op.CanIndex = -2,
    scaling: _LegacyScaling = "spectrum",
) -> tuple[_Array_f8, _ArrayComplex]: ...

# input_onesided is `False` (keyword)
@overload
def istft(
    Zxx: _ArrayLikeComplex_co,
    fs: AnyReal = 1.0,
    window: _GetWindowArgument | _ArrayLikeFloat_co = "hann",
    nperseg: AnyInt | None = None,
    noverlap: AnyInt | None = None,
    nfft: AnyInt | None = None,
    *,
    input_onesided: Literal[False],
    boundary: op.CanBool = True,
    time_axis: op.CanIndex = -1,
    freq_axis: op.CanIndex = -2,
    scaling: _LegacyScaling = "spectrum",
) -> tuple[_Array_f8, _ArrayComplex]: ...

#
def coherence(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    fs: AnyReal = 1.0,
    window: _GetWindowArgument | _ArrayLikeFloat_co = "hann",
    nperseg: AnyInt | None = None,
    noverlap: AnyInt | None = None,
    nfft: AnyInt | None = None,
    detrend: _Detrend = "constant",
    axis: op.CanIndex = -1,
) -> tuple[_Array_f8, _ArrayReal]: ...
