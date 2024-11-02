from collections.abc import Sequence
from typing import Literal, TypeAlias, overload
from typing_extensions import Unpack

import numpy as np
import optype as op
from scipy._typing import AnyReal

__all__ = [
    "barthann",
    "bartlett",
    "blackman",
    "blackmanharris",
    "bohman",
    "boxcar",
    "chebwin",
    "cosine",
    "dpss",
    "exponential",
    "flattop",
    "gaussian",
    "general_cosine",
    "general_gaussian",
    "general_hamming",
    "get_window",
    "hamming",
    "hann",
    "kaiser",
    "kaiser_bessel_derived",
    "lanczos",
    "nuttall",
    "parzen",
    "taylor",
    "triang",
    "tukey",
]


_Array_f8_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
_Array_f8_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]

_Norm: TypeAlias = Literal[2, "approximate", "subsample "]
_Window: TypeAlias = Literal[
    "barthann",
    "brthan",
    "bth",
    "bartlett",
    "bart",
    "brt",
    "blackman",
    "black",
    "blk",
    "blackmanharris",
    "blackharr",
    "bkh",
    "bohman",
    "bman",
    "bmn",
    "boxcar",
    "box",
    "ones",
    "rect",
    "rectangular",
    "cosine",
    "halfcosine",
    "exponential",
    "poisson",
    "flattop",
    "flat",
    "flt",
    "hamming",
    "hamm",
    "ham",
    "hann",
    "han",
    "lanczos",
    "sinc",
    "nuttall",
    "nutl",
    "nut",
    "parzen",
    "parz",
    "par",
    "taylor",
    "taylorwin",
    "triangle",
    "triang",
    "tri",
    "tukey",
    "tuk",
]
_WindowNeedsParams: TypeAlias = Literal[
    "chebwin",
    "cheb",
    "dpss",
    "gaussian",
    "gauss",
    "gss",
    "general cosine",
    "general_cosine",
    "general gaussian",
    "general_gaussian",
    "general gauss",
    "general_gauss",
    "ggs",
    "general hamming",
    "general_hamming",
    "kaiser",
    "ksr",
    "kaiser bessel derived",
    "kbd",
]

def general_cosine(M: int, a: Sequence[AnyReal], sym: op.CanBool = True) -> _Array_f8_1d: ...
def boxcar(M: int, sym: op.CanBool = True) -> _Array_f8_1d: ...
def triang(M: int, sym: op.CanBool = True) -> _Array_f8_1d: ...
def parzen(M: int, sym: op.CanBool = True) -> _Array_f8_1d: ...
def bohman(M: int, sym: op.CanBool = True) -> _Array_f8_1d: ...
def blackman(M: int, sym: op.CanBool = True) -> _Array_f8_1d: ...
def nuttall(M: int, sym: op.CanBool = True) -> _Array_f8_1d: ...
def blackmanharris(M: int, sym: op.CanBool = True) -> _Array_f8_1d: ...
def flattop(M: int, sym: op.CanBool = True) -> _Array_f8_1d: ...
def bartlett(M: int, sym: op.CanBool = True) -> _Array_f8_1d: ...
def hann(M: int, sym: op.CanBool = True) -> _Array_f8_1d: ...
def tukey(M: int, alpha: AnyReal = 0.5, sym: op.CanBool = True) -> _Array_f8_1d: ...
def barthann(M: int, sym: op.CanBool = True) -> _Array_f8_1d: ...
def general_hamming(M: int, alpha: AnyReal, sym: op.CanBool = True) -> _Array_f8_1d: ...
def hamming(M: int, sym: op.CanBool = True) -> _Array_f8_1d: ...
def kaiser(M: int, beta: AnyReal, sym: op.CanBool = True) -> _Array_f8_1d: ...
def kaiser_bessel_derived(M: int, beta: AnyReal, *, sym: op.CanBool = True) -> _Array_f8_1d: ...
def gaussian(M: int, std: AnyReal, sym: op.CanBool = True) -> _Array_f8_1d: ...
def general_gaussian(M: int, p: AnyReal, sig: AnyReal, sym: op.CanBool = True) -> _Array_f8_1d: ...
def chebwin(M: int, at: AnyReal, sym: op.CanBool = True) -> _Array_f8_1d: ...
def cosine(M: int, sym: op.CanBool = True) -> _Array_f8_1d: ...
def exponential(M: int, center: AnyReal | None = None, tau: AnyReal = 1.0, sym: op.CanBool = True) -> _Array_f8_1d: ...
def taylor(M: int, nbar: int = 4, sll: int = 30, norm: bool = True, sym: op.CanBool = True) -> _Array_f8_1d: ...
def lanczos(M: int, *, sym: op.CanBool = True) -> _Array_f8_1d: ...

#
@overload
def dpss(
    *,
    M: int,
    NW: AnyReal,
    Kmax: int = ...,
    sym: op.CanBool = True,
    norm: _Norm | None = None,
    return_ratios: Literal[False] = ...,
) -> _Array_f8_2d: ...
@overload
def dpss(
    *,
    M: int,
    NW: AnyReal,
    Kmax: None = ...,
    sym: op.CanBool = True,
    norm: _Norm | None = None,
    return_ratios: Literal[False] = ...,
) -> _Array_f8_1d: ...
@overload
def dpss(
    *,
    M: int,
    NW: AnyReal,
    Kmax: int = ...,
    sym: op.CanBool = True,
    norm: _Norm | None = None,
    return_ratios: Literal[True] = ...,
) -> tuple[_Array_f8_2d, _Array_f8_1d]: ...
@overload
def dpss(
    *,
    M: int,
    NW: AnyReal,
    Kmax: None = None,
    sym: op.CanBool = True,
    norm: _Norm | None = None,
    return_ratios: Literal[True] = ...,
) -> tuple[_Array_f8_1d, AnyReal]: ...

#
def get_window(
    window: _Window | AnyReal | tuple[_Window | _WindowNeedsParams, Unpack[tuple[object, ...]]], Nx: int, fftbins: bool = True
) -> _Array_f8_1d: ...