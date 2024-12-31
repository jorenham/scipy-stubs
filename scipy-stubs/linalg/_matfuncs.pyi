from collections.abc import Callable
from typing import Any, Literal, TypeAlias, overload

import numpy as np
import optype.numpy as onp
from scipy._typing import Falsy, Truthy
from ._expm_frechet import expm_cond, expm_frechet
from ._matfuncs_sqrtm import sqrtm

__all__ = [
    "coshm",
    "cosm",
    "expm",
    "expm_cond",
    "expm_frechet",
    "fractional_matrix_power",
    "funm",
    "khatri_rao",
    "logm",
    "signm",
    "sinhm",
    "sinm",
    "sqrtm",
    "tanhm",
    "tanm",
]

_ToPosInt: TypeAlias = np.unsignedinteger[Any] | Literal[0, 1, 2, 4, 5, 6, 7, 8]

_Int2D: TypeAlias = onp.Array2D[np.integer[Any]]
_Complex2D: TypeAlias = onp.Array2D[np.complexfloating[Any, Any]]
_Real2D: TypeAlias = onp.Array2D[np.floating[Any] | np.integer[Any]]
_Numeric2D: TypeAlias = onp.Array2D[np.number[Any]]
_Float2D: TypeAlias = onp.Array2D[np.floating[Any]]
_Inexact2D: TypeAlias = onp.Array2D[np.inexact[Any]]

_FloatND: TypeAlias = onp.ArrayND[np.floating[Any]]
_InexactND: TypeAlias = onp.ArrayND[np.inexact[Any]]

_FloatFunc: TypeAlias = Callable[[onp.Array1D[np.float64]], onp.ToFloat1D]
_ComplexFunc: TypeAlias = Callable[[onp.Array1D[np.complex128]], onp.ToComplex1D]

###

@overload  # int, positive int
def fractional_matrix_power(A: onp.ToInt2D, t: _ToPosInt) -> _Int2D: ...
@overload  # real, int
def fractional_matrix_power(A: onp.ToFloat2D, t: onp.ToInt) -> _Real2D: ...
@overload  # complex, int
def fractional_matrix_power(A: onp.ToComplex2D, t: onp.ToInt) -> _Numeric2D: ...
@overload  # complex, float
def fractional_matrix_power(A: onp.ToComplex2D, t: onp.ToJustFloat) -> _Complex2D: ...

# NOTE: return dtype depends on the sign of the values
@overload  # disp: True = ...
def logm(A: onp.ToComplex2D, disp: Truthy = True) -> _Inexact2D: ...
@overload  # disp: False
def logm(A: onp.ToComplex2D, disp: Falsy) -> tuple[_Inexact2D, float]: ...

#
@overload  # real
def expm(A: onp.ToFloatND) -> _FloatND: ...
@overload  # complex
def expm(A: onp.ToComplexND) -> _InexactND: ...

#
@overload  # real
def cosm(A: onp.ToFloat2D) -> _Float2D: ...
@overload  # complex
def cosm(A: onp.ToComplex2D) -> _Inexact2D: ...

#
@overload  # real
def sinm(A: onp.ToFloat2D) -> _Float2D: ...
@overload  # complex
def sinm(A: onp.ToComplex2D) -> _Inexact2D: ...

#
@overload  # real
def tanm(A: onp.ToFloat2D) -> _Float2D: ...
@overload  # complex
def tanm(A: onp.ToComplex2D) -> _Inexact2D: ...

#
@overload  # real
def coshm(A: onp.ToFloat2D) -> _Float2D: ...
@overload  # complex
def coshm(A: onp.ToComplex2D) -> _Inexact2D: ...

#
@overload  # real
def sinhm(A: onp.ToFloat2D) -> _Float2D: ...
@overload  # complex
def sinhm(A: onp.ToComplex2D) -> _Inexact2D: ...

#
@overload  # real
def tanhm(A: onp.ToFloat2D) -> _Float2D: ...
@overload  # complex
def tanhm(A: onp.ToComplex2D) -> _Inexact2D: ...

#
@overload  # real, disp: True = ...
def funm(A: onp.ToFloat2D, func: _FloatFunc, disp: Truthy = True) -> _Float2D: ...
@overload  # real, disp: False
def funm(A: onp.ToFloat2D, func: _FloatFunc, disp: Falsy) -> _Complex2D: ...
@overload  # complex, disp: True = ...
def funm(A: onp.ToComplex2D, func: _ComplexFunc, disp: Truthy = True) -> _Complex2D: ...
@overload  # complex, disp: False
def funm(A: onp.ToComplex2D, func: _ComplexFunc, disp: Falsy) -> tuple[_Complex2D, np.float64]: ...

#
@overload  # real, disp: True = ...
def signm(A: onp.ToFloat2D, disp: Truthy = True) -> _Float2D: ...
@overload  # real, disp: False
def signm(A: onp.ToFloat2D, disp: Falsy) -> tuple[_Float2D, np.float64]: ...
@overload  # complex, disp: True = ...
def signm(A: onp.ToComplex2D, disp: Truthy = True) -> _Inexact2D: ...
@overload  # complex, disp: False
def signm(A: onp.ToComplex2D, disp: Falsy) -> tuple[_Inexact2D, np.float64]: ...

#
@overload  # int
def khatri_rao(a: onp.ToInt2D, b: onp.ToInt2D) -> _Int2D: ...
@overload  # real
def khatri_rao(a: onp.ToFloat2D, b: onp.ToFloat2D) -> _Real2D: ...
@overload  # complex
def khatri_rao(a: onp.ToComplex2D, b: onp.ToComplex2D) -> _Numeric2D: ...
