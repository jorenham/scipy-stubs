from typing import Literal
from typing_extensions import deprecated

import optype.numpy as onp
from scipy._typing import Untyped, UntypedArray

__all__ = [
    "choose_conv_method",
    "cmplx_sort",
    "convolve",
    "convolve2d",
    "correlate",
    "correlate2d",
    "correlation_lags",
    "decimate",
    "deconvolve",
    "detrend",
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

def correlate(in1: Untyped, in2: Untyped, mode: str = "full", method: str = "auto") -> Untyped: ...
def correlation_lags(in1_len: Untyped, in2_len: Untyped, mode: str = "full") -> Untyped: ...
def fftconvolve(in1: Untyped, in2: Untyped, mode: str = "full", axes: Untyped | None = None) -> Untyped: ...
def oaconvolve(in1: Untyped, in2: Untyped, mode: str = "full", axes: Untyped | None = None) -> Untyped: ...
def choose_conv_method(in1: Untyped, in2: Untyped, mode: str = "full", measure: bool = False) -> Untyped: ...
def convolve(in1: Untyped, in2: Untyped, mode: str = "full", method: str = "auto") -> Untyped: ...
def order_filter(a: Untyped, domain: Untyped, rank: Untyped) -> Untyped: ...
def medfilt(volume: Untyped, kernel_size: Untyped | None = None) -> Untyped: ...
def wiener(im: Untyped, mysize: Untyped | None = None, noise: Untyped | None = None) -> Untyped: ...
def convolve2d(in1: Untyped, in2: Untyped, mode: str = "full", boundary: str = "fill", fillvalue: int = 0) -> Untyped: ...
def correlate2d(in1: Untyped, in2: Untyped, mode: str = "full", boundary: str = "fill", fillvalue: int = 0) -> Untyped: ...
def medfilt2d(input: Untyped, kernel_size: int = 3) -> Untyped: ...
def lfilter(b: Untyped, a: Untyped, x: Untyped, axis: int = -1, zi: Untyped | None = None) -> Untyped: ...
def lfiltic(b: Untyped, a: Untyped, y: Untyped, x: Untyped | None = None) -> Untyped: ...
def deconvolve(signal: Untyped, divisor: Untyped) -> Untyped: ...
def hilbert(x: Untyped, N: Untyped | None = None, axis: int = -1) -> Untyped: ...
def hilbert2(x: Untyped, N: Untyped | None = None) -> Untyped: ...
def unique_roots(p: Untyped, tol: float = 0.001, rtype: str = "min") -> Untyped: ...
def invres(r: Untyped, p: Untyped, k: Untyped, tol: float = 0.001, rtype: str = "avg") -> Untyped: ...
def residue(b: Untyped, a: Untyped, tol: float = 0.001, rtype: str = "avg") -> Untyped: ...
def residuez(b: Untyped, a: Untyped, tol: float = 0.001, rtype: str = "avg") -> Untyped: ...
def invresz(r: Untyped, p: Untyped, k: Untyped, tol: float = 0.001, rtype: str = "avg") -> Untyped: ...
def resample(
    x: Untyped,
    num: Untyped,
    t: Untyped | None = None,
    axis: int = 0,
    window: Untyped | None = None,
    domain: str = "time",
) -> Untyped: ...
def resample_poly(
    x: Untyped,
    up: Untyped,
    down: Untyped,
    axis: int = 0,
    window: Untyped = ("kaiser", 5.0),
    padtype: str = "constant",
    cval: Untyped | None = None,
) -> Untyped: ...
def vectorstrength(events: Untyped, period: Untyped) -> Untyped: ...
def detrend(
    data: UntypedArray,
    axis: int = -1,
    type: Literal["linear", "constant"] = "linear",
    bp: onp.ToJustInt | onp.ToJustIntND = 0,
    overwrite_data: bool = False,
) -> UntypedArray: ...
def lfilter_zi(b: Untyped, a: Untyped) -> Untyped: ...
def sosfilt_zi(sos: Untyped) -> Untyped: ...
def filtfilt(
    b: Untyped,
    a: Untyped,
    x: Untyped,
    axis: int = -1,
    padtype: str = "odd",
    padlen: Untyped | None = None,
    method: str = "pad",
    irlen: Untyped | None = None,
) -> Untyped: ...
def sosfilt(sos: Untyped, x: Untyped, axis: int = -1, zi: Untyped | None = None) -> Untyped: ...
def sosfiltfilt(sos: Untyped, x: Untyped, axis: int = -1, padtype: str = "odd", padlen: Untyped | None = None) -> Untyped: ...
def decimate(
    x: Untyped,
    q: Untyped,
    n: Untyped | None = None,
    ftype: str = "iir",
    axis: int = -1,
    zero_phase: bool = True,
) -> Untyped: ...
@deprecated("Will be removed in SciPy 1.5")
def cmplx_sort(p: Untyped) -> tuple[Untyped, Untyped]: ...
