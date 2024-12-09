from collections.abc import Callable
from typing import Literal, TypeAlias, overload

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import Untyped

__all__ = [
    "BadCoefficients",
    "band_stop_obj",
    "bessel",
    "besselap",
    "bilinear",
    "bilinear_zpk",
    "buttap",
    "butter",
    "buttord",
    "cheb1ap",
    "cheb1ord",
    "cheb2ap",
    "cheb2ord",
    "cheby1",
    "cheby2",
    "ellip",
    "ellipap",
    "ellipord",
    "findfreqs",
    "freqs",
    "freqs_zpk",
    "freqz",
    "freqz_zpk",
    "gammatone",
    "group_delay",
    "iircomb",
    "iirdesign",
    "iirfilter",
    "iirnotch",
    "iirpeak",
    "lp2bp",
    "lp2bp_zpk",
    "lp2bs",
    "lp2bs_zpk",
    "lp2hp",
    "lp2hp_zpk",
    "lp2lp",
    "lp2lp_zpk",
    "normalize",
    "sos2tf",
    "sos2zpk",
    "sosfreqz",
    "tf2sos",
    "tf2zpk",
    "zpk2sos",
    "zpk2tf",
]

_Float1D: TypeAlias = onp.Array1D[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float64]
_Complex1D: TypeAlias = onp.Array1D[np.complex128]
_ComplexND: TypeAlias = onp.ArrayND[np.complex128]
_Inexact1D: TypeAlias = onp.Array1D[np.float64 | np.complex128]
_InexactND: TypeAlias = onp.ArrayND[np.float64 | np.complex128]

###

class BadCoefficients(UserWarning): ...

#
def findfreqs(num: onp.ToComplex1D, den: onp.ToComplex1D, N: op.CanIndex, kind: Literal["ba", "zp"] = "ba") -> _Float1D: ...

#
@overload  # worN: real
def freqs(
    b: onp.ToComplex1D,
    a: onp.ToComplex1D,
    worN: op.CanIndex | onp.ToFloat1D = 200,
    plot: Callable[[_Float1D, _Complex1D], object] | None = None,
) -> tuple[_Float1D, _Complex1D]: ...
@overload  # worN: complex
def freqs(
    b: onp.ToComplex1D,
    a: onp.ToComplex1D,
    worN: onp.ToComplex1D,
    plot: Callable[[_Float1D, _Complex1D], object] | Callable[[_Complex1D, _Complex1D], object] | None = None,
) -> tuple[_Inexact1D, _Complex1D]: ...

#
@overload  # worN: real
def freqs_zpk(
    z: onp.ToComplex1D,
    p: onp.ToComplex1D,
    k: onp.ToComplex1D,
    worN: op.CanIndex | onp.ToFloat1D = 200,
) -> tuple[_Float1D, _Complex1D]: ...
@overload  # worN: complex
def freqs_zpk(
    z: onp.ToComplex1D,
    p: onp.ToComplex1D,
    k: onp.ToComplex1D,
    worN: onp.ToComplex1D,
) -> tuple[_Inexact1D, _Complex1D]: ...

#
@overload  # worN: real
def freqz(
    b: onp.ToComplex | onp.ToComplexND,
    a: onp.ToComplex | onp.ToComplexND = 1,
    worN: op.CanIndex | onp.ToFloat1D = 512,
    whole: op.CanBool = False,
    plot: Callable[[_FloatND, _ComplexND], object] | None = None,
    fs: onp.ToFloat = ...,  # 2 * pi
    include_nyquist: bool = False,
) -> tuple[_FloatND, _ComplexND]: ...
@overload  # worN: complex
def freqz(
    b: onp.ToComplex | onp.ToComplexND,
    a: onp.ToComplex | onp.ToComplexND = 1,
    worN: op.CanIndex | onp.ToComplex1D = 512,
    whole: op.CanBool = False,
    plot: Callable[[_FloatND, _ComplexND], object] | None = None,
    fs: onp.ToFloat = ...,  # 2 * pi
    include_nyquist: bool = False,
) -> tuple[_InexactND, _ComplexND]: ...

#
@overload  # worN: real
def freqz_zpk(
    z: onp.ToComplex1D,
    p: onp.ToComplex1D,
    k: onp.ToComplex1D,
    worN: op.CanIndex | onp.ToFloat1D = 512,
    whole: op.CanBool = False,
    fs: onp.ToFloat = ...,  # 2 * pi
) -> tuple[_FloatND, _ComplexND]: ...
@overload  # worN: complex
def freqz_zpk(
    z: onp.ToComplex1D,
    p: onp.ToComplex1D,
    k: onp.ToComplex1D,
    worN: onp.ToComplex1D,
    whole: op.CanBool = False,
    fs: onp.ToFloat = ...,  # 2 * pi
) -> tuple[_InexactND, _ComplexND]: ...

#
@overload  # w: real
def group_delay(
    system: tuple[onp.ToComplex1D, onp.ToComplex1D],
    w: op.CanIndex | onp.ToFloat1D | None = 512,
    whole: op.CanBool = False,
    fs: onp.ToFloat = ...,  # 2 * pi
) -> tuple[_Float1D, _Float1D]: ...
@overload  # w: complex
def group_delay(
    system: tuple[onp.ToComplex1D, onp.ToComplex1D],
    w: onp.ToComplex1D,
    whole: op.CanBool = False,
    fs: onp.ToFloat = ...,  # 2 * pi
) -> tuple[_Inexact1D, _Float1D]: ...

#
@overload  # worN: real
def sosfreqz(
    sos: onp.ToFloat2D,
    worN: op.CanIndex | onp.ToFloat1D = 512,
    whole: op.CanBool = False,
    fs: onp.ToFloat = ...,  # 2 * pi
) -> tuple[_Float1D, _Complex1D]: ...
@overload  # worN: real
def sosfreqz(
    sos: onp.ToFloat2D,
    worN: onp.ToComplex1D,
    whole: op.CanBool = False,
    fs: onp.ToFloat = ...,  # 2 * pi
) -> tuple[_Inexact1D, _Complex1D]: ...

# TODO(jorenham): https://github.com/jorenham/scipy-stubs/issues/99

def tf2zpk(b: Untyped, a: Untyped) -> Untyped: ...
def zpk2tf(z: Untyped, p: Untyped, k: Untyped) -> Untyped: ...
def tf2sos(b: Untyped, a: Untyped, pairing: Untyped | None = None, *, analog: bool = False) -> Untyped: ...
def sos2tf(sos: Untyped) -> Untyped: ...
def sos2zpk(sos: Untyped) -> Untyped: ...
def zpk2sos(z: Untyped, p: Untyped, k: Untyped, pairing: Untyped | None = None, *, analog: bool = False) -> Untyped: ...
def normalize(b: Untyped, a: Untyped) -> Untyped: ...
def lp2lp(b: Untyped, a: Untyped, wo: float = 1.0) -> Untyped: ...
def lp2hp(b: Untyped, a: Untyped, wo: float = 1.0) -> Untyped: ...
def lp2bp(b: Untyped, a: Untyped, wo: float = 1.0, bw: float = 1.0) -> Untyped: ...
def lp2bs(b: Untyped, a: Untyped, wo: float = 1.0, bw: float = 1.0) -> Untyped: ...
def bilinear(b: Untyped, a: Untyped, fs: float = 1.0) -> Untyped: ...
def iirdesign(
    wp: Untyped,
    ws: Untyped,
    gpass: Untyped,
    gstop: Untyped,
    analog: bool = False,
    ftype: str = "ellip",
    output: str = "ba",
    fs: Untyped | None = None,
) -> Untyped: ...
def iirfilter(
    N: Untyped,
    Wn: Untyped,
    rp: Untyped | None = None,
    rs: Untyped | None = None,
    btype: str = "band",
    analog: bool = False,
    ftype: str = "butter",
    output: str = "ba",
    fs: Untyped | None = None,
) -> Untyped: ...
def bilinear_zpk(z: Untyped, p: Untyped, k: Untyped, fs: Untyped) -> Untyped: ...
def lp2lp_zpk(z: Untyped, p: Untyped, k: Untyped, wo: float = 1.0) -> Untyped: ...
def lp2hp_zpk(z: Untyped, p: Untyped, k: Untyped, wo: float = 1.0) -> Untyped: ...
def lp2bp_zpk(z: Untyped, p: Untyped, k: Untyped, wo: float = 1.0, bw: float = 1.0) -> Untyped: ...
def lp2bs_zpk(z: Untyped, p: Untyped, k: Untyped, wo: float = 1.0, bw: float = 1.0) -> Untyped: ...
def butter(
    N: Untyped,
    Wn: Untyped,
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: Untyped | None = None,
) -> Untyped: ...
def cheby1(
    N: Untyped,
    rp: Untyped,
    Wn: Untyped,
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: Untyped | None = None,
) -> Untyped: ...
def cheby2(
    N: Untyped,
    rs: Untyped,
    Wn: Untyped,
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: Untyped | None = None,
) -> Untyped: ...
def ellip(
    N: Untyped,
    rp: Untyped,
    rs: Untyped,
    Wn: Untyped,
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: Untyped | None = None,
) -> Untyped: ...
def bessel(
    N: Untyped,
    Wn: Untyped,
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    norm: str = "phase",
    fs: Untyped | None = None,
) -> Untyped: ...
def maxflat() -> None: ...
def yulewalk() -> None: ...
def band_stop_obj(
    wp: Untyped,
    ind: Untyped,
    passb: Untyped,
    stopb: Untyped,
    gpass: Untyped,
    gstop: Untyped,
    type: Untyped,
) -> Untyped: ...
def buttord(
    wp: Untyped,
    ws: Untyped,
    gpass: Untyped,
    gstop: Untyped,
    analog: bool = False,
    fs: Untyped | None = None,
) -> Untyped: ...
def cheb1ord(
    wp: Untyped,
    ws: Untyped,
    gpass: Untyped,
    gstop: Untyped,
    analog: bool = False,
    fs: Untyped | None = None,
) -> Untyped: ...
def cheb2ord(
    wp: Untyped,
    ws: Untyped,
    gpass: Untyped,
    gstop: Untyped,
    analog: bool = False,
    fs: Untyped | None = None,
) -> Untyped: ...
def ellipord(
    wp: Untyped,
    ws: Untyped,
    gpass: Untyped,
    gstop: Untyped,
    analog: bool = False,
    fs: Untyped | None = None,
) -> Untyped: ...
def buttap(N: Untyped) -> Untyped: ...
def cheb1ap(N: Untyped, rp: Untyped) -> Untyped: ...
def cheb2ap(N: Untyped, rs: Untyped) -> Untyped: ...
def ellipap(N: Untyped, rp: Untyped, rs: Untyped) -> Untyped: ...
def besselap(N: Untyped, norm: str = "phase") -> Untyped: ...
def iirnotch(w0: Untyped, Q: Untyped, fs: float = 2.0) -> Untyped: ...
def iirpeak(w0: Untyped, Q: Untyped, fs: float = 2.0) -> Untyped: ...
def iircomb(w0: Untyped, Q: Untyped, ftype: str = "notch", fs: float = 2.0, *, pass_zero: bool = False) -> Untyped: ...
def gammatone(
    freq: Untyped,
    ftype: Untyped,
    order: Untyped | None = None,
    numtaps: Untyped | None = None,
    fs: Untyped | None = None,
) -> Untyped: ...
