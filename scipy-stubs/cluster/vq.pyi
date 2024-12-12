from typing import Any, Literal, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy._typing import Seed

__all__ = ["kmeans", "kmeans2", "vq", "whiten"]

_SCT_fc = TypeVar("_SCT_fc", bound=np.inexact[Any])

###

class ClusterError(Exception): ...

@overload
def whiten(obs: onp.ArrayND[_SCT_fc], check_finite: bool = True) -> onp.Array2D[_SCT_fc]: ...
@overload
def whiten(obs: onp.ToFloat2D, check_finite: bool = True) -> onp.Array2D[np.floating[Any]]: ...
@overload
def whiten(obs: onp.ToComplex2D, check_finite: bool = True) -> onp.Array2D[np.inexact[Any]]: ...

#
def vq(
    obs: onp.ToComplex2D,
    code_book: onp.ToComplex2D,
    check_finite: bool = True,
) -> tuple[onp.Array1D[np.int32 | np.intp], onp.Array1D[_SCT_fc]]: ...

#
def py_vq(
    obs: onp.ToComplex2D,
    code_book: onp.ToComplex2D,
    check_finite: bool = True,
) -> tuple[onp.Array1D[np.intp], onp.Array1D[_SCT_fc]]: ...

#
@overload  # real
def kmeans(
    obs: onp.ToFloat2D,
    k_or_guess: onp.ToJustInt | onp.ToFloatND,
    iter: int = 20,
    thresh: float = 1e-05,
    check_finite: bool = True,
    *,
    seed: Seed | None = None,
) -> tuple[onp.Array2D[np.floating[Any]], float]: ...
@overload  # complex
def kmeans(
    obs: onp.ToComplex2D,
    k_or_guess: onp.ToJustInt | onp.ToFloatND,
    iter: int = 20,
    thresh: float = 1e-05,
    check_finite: bool = True,
    *,
    seed: Seed | None = None,
) -> tuple[onp.Array2D[np.inexact[Any]], float]: ...

#
@overload  # real
def kmeans2(
    data: onp.ToFloat1D | onp.ToFloat2D,
    k: onp.ToJustInt | onp.ToFloatND,
    iter: int = 10,
    thresh: float = 1e-05,
    minit: Literal["random", "points", "++", "matrix"] = "random",
    missing: Literal["warn", "raise"] = "warn",
    check_finite: bool = True,
    *,
    seed: Seed | None = None,
) -> tuple[onp.Array2D[np.floating[Any]], onp.Array1D[np.int32]]: ...
@overload  # complex
def kmeans2(
    data: onp.ToComplex1D | onp.ToComplex2D,
    k: onp.ToJustInt | onp.ToFloatND,
    iter: int = 10,
    thresh: float = 1e-05,
    minit: Literal["random", "points", "++", "matrix"] = "random",
    missing: Literal["warn", "raise"] = "warn",
    check_finite: bool = True,
    *,
    seed: Seed | None = None,
) -> tuple[onp.Array2D[np.inexact[Any]], onp.Array1D[np.int32]]: ...
