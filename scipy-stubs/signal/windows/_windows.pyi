from typing import Literal, TypeAlias, overload
from typing_extensions import TypeAliasType

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import Falsy, Truthy

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

###

_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]

_Norm: TypeAlias = Literal[2, "approximate", "subsample"]
_ToM: TypeAlias = int | np.int16 | np.int32 | np.int64
_WindowName0: TypeAlias = Literal[
    "barthann", "brthan", "bth",
    "bartlett", "bart", "brt",
    "blackman", "black", "blk",
    "blackmanharris", "blackharr", "bkh",
    "bohman", "bman", "bmn",
    "boxcar", "box", "ones", "rect", "rectangular",
    "cosine", "halfcosine",
    "exponential", "poisson",
    "flattop", "flat", "flt",
    "hamming", "hamm", "ham",
    "hann", "han",
    "lanczos", "sinc",
    "nuttall", "nutl", "nut",
    "parzen", "parz", "par",
    "taylor", "taylorwin",
    "triangle", "triang", "tri",
    "tukey", "tuk",
]  # fmt: skip
_WindowName1: TypeAlias = Literal[
    "chebwin", "cheb",
    "dpss",
    "exponential", "poisson",
    "gaussian", "gauss", "gss",
    "general hamming", "general_hamming",
    "kaiser", "ksr",
    "kaiser bessel derived", "kbd",
    "tukey", "tuk",
]  # fmt: skip
_WindowName2: TypeAlias = Literal[
    "general gaussian", "general_gaussian", "general gauss", "general_gauss", "ggs",
    "exponential", "poisson",
]  # fmt: skip
_WindowName_taylor: TypeAlias = Literal["taylor", "taylorwin"]
_WindowName_gencos: TypeAlias = Literal["general cosine", "general_cosine"]

_ToWindow = TypeAliasType(
    "_ToWindow",
    onp.ToFloat
    | _WindowName0
    | tuple[_WindowName0]
    | tuple[_WindowName1, onp.ToFloat]
    | tuple[_WindowName2, onp.ToFloat1D, onp.ToFloat1D]
    | tuple[_WindowName_taylor, onp.ToInt]
    | tuple[_WindowName_taylor, onp.ToInt, onp.ToInt]
    | tuple[_WindowName_taylor, onp.ToInt, onp.ToInt, op.CanBool]
    | tuple[_WindowName_gencos, onp.ToFloat1D]
    | tuple[Literal["dpss"], onp.ToFloat, op.CanIndex],
)

###

def get_window(window: _ToWindow, Nx: _ToM, fftbins: op.CanBool = True) -> _Float1D: ...

#
def barthann(M: _ToM, sym: op.CanBool = True) -> _Float1D: ...
def bartlett(M: _ToM, sym: op.CanBool = True) -> _Float1D: ...
def blackman(M: _ToM, sym: op.CanBool = True) -> _Float1D: ...
def blackmanharris(M: _ToM, sym: op.CanBool = True) -> _Float1D: ...
def bohman(M: _ToM, sym: op.CanBool = True) -> _Float1D: ...
def boxcar(M: _ToM, sym: op.CanBool = True) -> _Float1D: ...
def cosine(M: _ToM, sym: op.CanBool = True) -> _Float1D: ...
def flattop(M: _ToM, sym: op.CanBool = True) -> _Float1D: ...
def hamming(M: _ToM, sym: op.CanBool = True) -> _Float1D: ...
def hann(M: _ToM, sym: op.CanBool = True) -> _Float1D: ...
def lanczos(M: _ToM, *, sym: op.CanBool = True) -> _Float1D: ...
def nuttall(M: _ToM, sym: op.CanBool = True) -> _Float1D: ...
def parzen(M: _ToM, sym: op.CanBool = True) -> _Float1D: ...
def triang(M: _ToM, sym: op.CanBool = True) -> _Float1D: ...

#
def exponential(M: _ToM, center: onp.ToFloat | None = None, tau: onp.ToFloat = 1.0, sym: op.CanBool = True) -> _Float1D: ...
def taylor(M: _ToM, nbar: onp.ToInt = 4, sll: onp.ToInt = 30, norm: op.CanBool = True, sym: op.CanBool = True) -> _Float1D: ...
def tukey(M: _ToM, alpha: onp.ToFloat = 0.5, sym: op.CanBool = True) -> _Float1D: ...

#
def chebwin(M: _ToM, at: onp.ToFloat, sym: op.CanBool = True) -> _Float1D: ...
def gaussian(M: _ToM, std: onp.ToFloat, sym: op.CanBool = True) -> _Float1D: ...
def general_cosine(M: _ToM, a: onp.ToFloat1D, sym: op.CanBool = True) -> _Float1D: ...
def general_hamming(M: _ToM, alpha: onp.ToFloat, sym: op.CanBool = True) -> _Float1D: ...
def general_gaussian(M: _ToM, p: onp.ToFloat, sig: onp.ToFloat, sym: op.CanBool = True) -> _Float1D: ...
def kaiser(M: _ToM, beta: onp.ToFloat, sym: op.CanBool = True) -> _Float1D: ...
def kaiser_bessel_derived(M: _ToM, beta: onp.ToFloat, *, sym: op.CanBool = True) -> _Float1D: ...
@overload  # `return_ratios` is `False`
def dpss(
    M: _ToM,
    NW: onp.ToFloat,
    Kmax: op.CanIndex,
    sym: op.CanBool = True,
    norm: _Norm | None = None,
    return_ratios: Falsy = False,
) -> _Float2D: ...
@overload  # `return_ratios` is `False`.
def dpss(
    M: _ToM,
    NW: onp.ToFloat,
    Kmax: None = None,
    sym: op.CanBool = True,
    norm: _Norm | None = None,
    return_ratios: Falsy = False,
) -> _Float1D: ...
@overload  # `return_ratios` is `True`, `return_ratios` as a positional argument
def dpss(
    M: _ToM,
    NW: onp.ToFloat,
    Kmax: op.CanIndex,
    sym: op.CanBool,
    norm: _Norm | None,
    return_ratios: Truthy,
) -> tuple[_Float2D, _Float1D]: ...
@overload  # `return_ratios` as a keyword argument
def dpss(
    M: _ToM,
    NW: onp.ToFloat,
    Kmax: op.CanIndex,
    sym: op.CanBool = True,
    norm: _Norm | None = None,
    *,
    return_ratios: Truthy,
) -> tuple[_Float2D, _Float1D]: ...
@overload  # `return_ratios` as a positional argument
def dpss(
    M: _ToM,
    NW: onp.ToFloat,
    Kmax: None,
    sym: op.CanBool,
    norm: _Norm | None,
    return_ratios: Truthy,
) -> tuple[_Float1D, np.float64]: ...
@overload  # `return_ratios` as a keyword argument
def dpss(
    M: _ToM,
    NW: onp.ToFloat,
    Kmax: None = None,
    sym: op.CanBool = True,
    norm: _Norm | None = None,
    *,
    return_ratios: Truthy,
) -> tuple[_Float1D, np.float64]: ...
