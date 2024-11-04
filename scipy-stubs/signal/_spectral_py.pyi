from collections.abc import Callable
from typing import Any, Literal, TypeAlias
from typing_extensions import Unpack

import numpy as np
import numpy.typing as npt
import optype as op
from numpy._typing import _ArrayLikeComplex_co, _ArrayLikeFloat_co
from scipy._typing import AnyInt, AnyReal, Untyped, UntypedCallable
from scipy.signal.windows._windows import _Window, _WindowNeedsParams

__all__ = ["check_COLA", "check_NOLA", "coherence", "csd", "istft", "lombscargle", "periodogram", "spectrogram", "stft", "welch"]

_GetWindowArgument: TypeAlias = _Window | tuple[_Window | _WindowNeedsParams, Unpack[tuple[object, ...]]]
_Detrend: TypeAlias = Literal["literal", "constant", False] | Callable[[npt.NDArray[np.generic]], npt.NDArray[np.generic]]
_Scaling: TypeAlias = Literal["density", "spectrum"]
_Average: TypeAlias = Literal["mean", "median"]

def lombscargle(x: Untyped, y: Untyped, freqs: Untyped, precenter: bool = False, normalize: bool = False) -> Untyped: ...
def periodogram(
    x: Untyped,
    fs: float = 1.0,
    window: str = "boxcar",
    nfft: int | None = None,
    detrend: str | Literal[False] | UntypedCallable = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
) -> Untyped: ...
def welch(
    x: Untyped,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | Literal[False] | UntypedCallable = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
    average: str = "mean",
) -> Untyped: ...
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
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complexfloating[Any, Any]]]: ...
def spectrogram(
    x: Untyped,
    fs: float = 1.0,
    window: Untyped = ("tukey", 0.25),
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | Literal[False] | UntypedCallable = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
    mode: str = "psd",
) -> Untyped: ...
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
    x: Untyped,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: bool = False,
    return_onesided: bool = True,
    boundary: str = "zeros",
    padded: bool = True,
    axis: int = -1,
    scaling: str = "spectrum",
) -> Untyped: ...
def istft(
    Zxx: Untyped,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    input_onesided: bool = True,
    boundary: bool = True,
    time_axis: int = -1,
    freq_axis: int = -2,
    scaling: str = "spectrum",
) -> Untyped: ...
def coherence(
    x: Untyped,
    y: Untyped,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | Literal[False] | UntypedCallable = "constant",
    axis: int = -1,
) -> Untyped: ...
