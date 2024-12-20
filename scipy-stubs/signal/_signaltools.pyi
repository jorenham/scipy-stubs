from typing import Any, Literal, TypeAlias, overload

import numpy as np
import optype.numpy as onp
from scipy._typing import CorrelateMode, Untyped, UntypedArray
from ._ltisys import dlti

__all__ = [
    "choose_conv_method",
    "convolve",
    "convolve2d",
    "correlate",
    "correlate2d",
    "correlation_lags",
    "decimate",
    "deconvolve",
    "detrend",
    "envelope",
    "fftconvolve",
    "filtfilt",
    "hilbert",
    "hilbert2",
    "invres",
    "invresz",
    "lfilter",
    "lfilter_zi",
    "lfiltic",
    "medfilt",
    "medfilt2d",
    "oaconvolve",
    "order_filter",
    "resample",
    "resample_poly",
    "residue",
    "residuez",
    "sosfilt",
    "sosfilt_zi",
    "sosfiltfilt",
    "unique_roots",
    "vectorstrength",
    "wiener",
]

_CorrelateMethod: TypeAlias = Literal["auto", "direct", "fft"]
_BoundaryConditions: TypeAlias = Literal["fill", "wrap", "symm"]
_ResidueType: TypeAlias = Literal["avg", "min", "max"]
_RootType: TypeAlias = Literal[_ResidueType, "maximum", "avg", "mean"]
_Domain: TypeAlias = Literal["time", "freq"]
_TrendType: TypeAlias = Literal["linear", "constant"]
_PadType: TypeAlias = Literal["constant", "line", "mean", "median", "maximum", "minimum", "symmetric", "reflect", "edge", "wrap"]
_FiltFiltPadType: TypeAlias = Literal["odd", "even", "constant"]
_ResidualKind: TypeAlias = Literal["lowpass", "all"]
_FilterType: TypeAlias = Literal["iir", "fir"] | dlti

###

# TODO(jorenham): deprecate `longdouble` and `object_` input
def choose_conv_method(in1: Untyped, in2: Untyped, mode: CorrelateMode = "full", measure: bool = False) -> Untyped: ...
def correlate(in1: Untyped, in2: Untyped, mode: CorrelateMode = "full", method: _CorrelateMethod = "auto") -> Untyped: ...
def convolve(in1: Untyped, in2: Untyped, mode: CorrelateMode = "full", method: _CorrelateMethod = "auto") -> Untyped: ...
def lfilter(b: Untyped, a: Untyped, x: Untyped, axis: int = -1, zi: Untyped | None = None) -> Untyped: ...
def sosfilt(sos: Untyped, x: Untyped, axis: int = -1, zi: Untyped | None = None) -> Untyped: ...

#
def correlation_lags(in1_len: Untyped, in2_len: Untyped, mode: CorrelateMode = "full") -> Untyped: ...
def fftconvolve(in1: Untyped, in2: Untyped, mode: CorrelateMode = "full", axes: Untyped | None = None) -> Untyped: ...
def oaconvolve(in1: Untyped, in2: Untyped, mode: CorrelateMode = "full", axes: Untyped | None = None) -> Untyped: ...

#
def order_filter(a: Untyped, domain: Untyped, rank: Untyped) -> Untyped: ...

#
def medfilt(volume: Untyped, kernel_size: Untyped | None = None) -> Untyped: ...
def medfilt2d(input: Untyped, kernel_size: int = 3) -> Untyped: ...

#
def wiener(im: Untyped, mysize: Untyped | None = None, noise: Untyped | None = None) -> Untyped: ...

#
def convolve2d(
    in1: Untyped,
    in2: Untyped,
    mode: CorrelateMode = "full",
    boundary: _BoundaryConditions = "fill",
    fillvalue: onp.ToComplex = 0,
) -> Untyped: ...
def correlate2d(
    in1: Untyped,
    in2: Untyped,
    mode: CorrelateMode = "full",
    boundary: _BoundaryConditions = "fill",
    fillvalue: onp.ToComplex = 0,
) -> Untyped: ...

#
def lfiltic(b: Untyped, a: Untyped, y: Untyped, x: Untyped | None = None) -> Untyped: ...

#
def deconvolve(signal: Untyped, divisor: Untyped) -> Untyped: ...

#
def hilbert(x: Untyped, N: Untyped | None = None, axis: int = -1) -> Untyped: ...
def hilbert2(x: Untyped, N: Untyped | None = None) -> Untyped: ...

#
def unique_roots(p: Untyped, tol: float = 0.001, rtype: _RootType = "min") -> Untyped: ...

#
def residue(b: Untyped, a: Untyped, tol: float = 0.001, rtype: _ResidueType = "avg") -> Untyped: ...
def residuez(b: Untyped, a: Untyped, tol: float = 0.001, rtype: _ResidueType = "avg") -> Untyped: ...

#
def invres(r: Untyped, p: Untyped, k: Untyped, tol: float = 0.001, rtype: _ResidueType = "avg") -> Untyped: ...
def invresz(r: Untyped, p: Untyped, k: Untyped, tol: float = 0.001, rtype: _ResidueType = "avg") -> Untyped: ...

#
def resample(
    x: Untyped,
    num: Untyped,
    t: Untyped | None = None,
    axis: int = 0,
    window: Untyped | None = None,
    domain: _Domain = "time",
) -> Untyped: ...
def resample_poly(
    x: Untyped,
    up: Untyped,
    down: Untyped,
    axis: int = 0,
    window: Untyped = ("kaiser", 5.0),
    padtype: _PadType = "constant",
    cval: Untyped | None = None,
) -> Untyped: ...

#
def vectorstrength(events: Untyped, period: Untyped) -> Untyped: ...

#
def detrend(
    data: UntypedArray,
    axis: int = -1,
    type: _TrendType = "linear",
    bp: onp.ToJustInt | onp.ToJustIntND = 0,
    overwrite_data: bool = False,
) -> UntypedArray: ...

#
def lfilter_zi(b: Untyped, a: Untyped) -> Untyped: ...
def sosfilt_zi(sos: Untyped) -> Untyped: ...

#
def filtfilt(
    b: Untyped,
    a: Untyped,
    x: Untyped,
    axis: int = -1,
    padtype: _FiltFiltPadType = "odd",
    padlen: int | None = None,
    method: Literal["pad", "gust"] = "pad",
    irlen: int | None = None,
) -> Untyped: ...

#
def sosfiltfilt(
    sos: Untyped,
    x: Untyped,
    axis: int = -1,
    padtype: _FiltFiltPadType = "odd",
    padlen: int | None = None,
) -> Untyped: ...

#
def decimate(
    x: Untyped,
    q: int,
    n: int | None = None,
    ftype: _FilterType = "iir",
    axis: int = -1,
    zero_phase: bool = True,
) -> Untyped: ...

#

@overload
def envelope(
    z: onp.ArrayND[np.floating[Any]],
    bp_in: tuple[int | None, int | None] = (1, None),
    *,
    n_out: int | None = None,
    squared: bool = False,
    residual: _ResidualKind | None = "lowpass",
    axis: int = -1,
) -> onp.ArrayND[np.floating[Any]]: ...
@overload
def envelope(
    z: onp.ArrayND[np.inexact[Any]],
    bp_in: tuple[int | None, int | None] = (1, None),
    *,
    n_out: int | None = None,
    squared: bool = False,
    residual: _ResidualKind | None = "lowpass",
    axis: int = -1,
) -> onp.ArrayND[np.inexact[Any]]: ...
