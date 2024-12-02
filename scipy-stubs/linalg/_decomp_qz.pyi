from collections.abc import Callable
from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp

__all__ = ["ordqz", "qz"]

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_Tuple4: TypeAlias = tuple[_T2, _T2, _T2, _T2]
_Tuple222: TypeAlias = tuple[_T2, _T2, _T1, _T1, _T2, _T2]

_Float1D: TypeAlias = onp.Array1D[np.floating[Any]]
_Float2D: TypeAlias = onp.Array2D[np.floating[Any]]
_Complex1D: TypeAlias = onp.Array1D[np.complexfloating[Any, Any]]
_Complex2D: TypeAlias = onp.Array2D[np.complexfloating[Any, Any]]
_Inexact1D: TypeAlias = onp.Array1D[np.inexact[Any]]
_Inexact2D: TypeAlias = onp.Array2D[np.inexact[Any]]

_OutputReal: TypeAlias = Literal["real", "r"]
_OutputComplex: TypeAlias = Literal["complex", "c"]

_Sort: TypeAlias = Literal["lhp", "rhp", "iuc", "ouc"] | Callable[[float, float], onp.ToBool]

###

# NOTE: `sort` will raise `ValueError` if not `None`.
@overload  # float, {"real"}
def qz(
    A: onp.ToFloat2D,
    B: onp.ToFloat2D,
    output: _OutputReal = "real",
    lwork: onp.ToJustInt | None = None,
    sort: None = None,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Tuple4[_Float2D]: ...
@overload  # complex, {"real"}
def qz(
    A: onp.ToComplex2D,
    B: onp.ToComplex2D,
    output: _OutputReal = "real",
    lwork: onp.ToJustInt | None = None,
    sort: None = None,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Tuple4[_Inexact2D]: ...
@overload  # complex, {"complex"}
def qz(
    A: onp.ToComplex2D,
    B: onp.ToComplex2D,
    output: _OutputComplex,
    lwork: onp.ToJustInt | None = None,
    sort: None = None,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Tuple4[_Complex2D]: ...

#
@overload  # float, {"real"}
def ordqz(
    A: onp.ToFloat2D,
    B: onp.ToFloat2D,
    sort: _Sort = "lhp",
    output: _OutputReal = "real",
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Tuple222[_Float2D, _Float1D]: ...
@overload  # complex, {"real"}
def ordqz(
    A: onp.ToComplex2D,
    B: onp.ToComplex2D,
    sort: _Sort = "lhp",
    output: _OutputReal = "real",
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Tuple222[_Inexact2D, _Inexact1D]: ...
@overload  # complex, {"complex"} (positional)
def ordqz(
    A: onp.ToComplex2D,
    B: onp.ToComplex2D,
    sort: _Sort,
    output: _OutputComplex,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Tuple222[_Complex2D, _Complex1D]: ...
@overload  # complex, {"complex"} (keyword)
def ordqz(
    A: onp.ToComplex2D,
    B: onp.ToComplex2D,
    sort: _Sort = "lhp",
    *,
    output: _OutputComplex,
    overwrite_a: onp.ToBool = False,
    overwrite_b: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _Tuple222[_Complex2D, _Complex1D]: ...
