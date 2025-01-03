from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy._typing import ToRNG

__all__ = ["kmeans", "kmeans2", "vq", "whiten"]

_InitMethod: TypeAlias = Literal["random", "points", "++", "matrix"]
_MissingMethod: TypeAlias = Literal["warn", "raise"]

_Floating: TypeAlias = np.floating[Any]
_Inexact: TypeAlias = np.inexact[Any]

_InexactT = TypeVar("_InexactT", bound=_Inexact)

###

class ClusterError(Exception): ...

#
@overload
def whiten(obs: onp.ArrayND[np.bool_ | np.integer[Any]], check_finite: bool = True) -> onp.Array2D[np.float64]: ...
@overload
def whiten(obs: onp.ArrayND[_InexactT], check_finite: bool = True) -> onp.Array2D[_InexactT]: ...

#
@overload
def vq(
    obs: onp.ToFloat2D,
    code_book: onp.ToFloat2D,
    check_finite: bool = True,
) -> tuple[onp.Array1D[np.int32 | np.intp], onp.Array1D[_Floating]]: ...
@overload
def vq(
    obs: onp.ToComplex2D,
    code_book: onp.ToComplex2D,
    check_finite: bool = True,
) -> tuple[onp.Array1D[np.int32 | np.intp], onp.Array1D[_Inexact]]: ...

#
@overload
def py_vq(
    obs: onp.ToFloat2D,
    code_book: onp.ToFloat2D,
    check_finite: bool = True,
) -> tuple[onp.Array1D[np.intp], onp.Array1D[_Floating]]: ...
@overload
def py_vq(
    obs: onp.ToComplex2D,
    code_book: onp.ToComplex2D,
    check_finite: bool = True,
) -> tuple[onp.Array1D[np.intp], onp.Array1D[_Inexact]]: ...

#
@overload  # real
def kmeans(
    obs: onp.ToFloat2D,
    k_or_guess: onp.ToJustInt | onp.ToFloatND,
    iter: int = 20,
    thresh: float = 1e-5,
    check_finite: bool = True,
    *,
    rng: ToRNG = None,
) -> tuple[onp.Array2D[_Floating], float]: ...
@overload  # complex
def kmeans(
    obs: onp.ToComplex2D,
    k_or_guess: onp.ToJustInt | onp.ToFloatND,
    iter: int = 20,
    thresh: float = 1e-5,
    check_finite: bool = True,
    *,
    rng: ToRNG = None,
) -> tuple[onp.Array2D[_Inexact], float]: ...

#
@overload  # real
def kmeans2(
    data: onp.ToFloat1D | onp.ToFloat2D,
    k: onp.ToJustInt | onp.ToFloatND,
    iter: int = 10,
    thresh: float = 1e-5,
    minit: _InitMethod = "random",
    missing: _MissingMethod = "warn",
    check_finite: bool = True,
    *,
    rng: ToRNG = None,
) -> tuple[onp.Array2D[_Floating], onp.Array1D[np.int32]]: ...
@overload  # complex
def kmeans2(
    data: onp.ToComplex1D | onp.ToComplex2D,
    k: onp.ToJustInt | onp.ToFloatND,
    iter: int = 10,
    thresh: float = 1e-5,
    minit: _InitMethod = "random",
    missing: _MissingMethod = "warn",
    check_finite: bool = True,
    *,
    rng: ToRNG = None,
) -> tuple[onp.Array2D[_Inexact], onp.Array1D[np.int32]]: ...
