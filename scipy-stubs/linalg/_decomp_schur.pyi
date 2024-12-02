from collections.abc import Callable
from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp

__all__ = ["rsf2csf", "schur"]

_T = TypeVar("_T")
_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple2i: TypeAlias = tuple[_T, _T, int]

_Float2D: TypeAlias = onp.Array2D[np.floating[Any]]
_Complex2D: TypeAlias = onp.Array2D[np.complexfloating[Any, Any]]
_Inexact2D: TypeAlias = onp.Array2D[np.inexact[Any]]

_OutputReal: TypeAlias = Literal["real", "r"]
_OutputComplex: TypeAlias = Literal["complex", "c"]

_Sort: TypeAlias = Literal["lhp", "rhp", "iuc", "ouc"] | Callable[[float, float], onp.ToBool]

###

@overload  # float, output: {"real"}, sort: _Sort (positional)
def schur(
    a: onp.ToFloat2D,
    output: _OutputReal = "real",
    lwork: onp.ToJustInt | None = None,
    overwrite_a: onp.ToBool = False,
    sort: None = None,
    check_finite: onp.ToBool = True,
) -> _Tuple2[_Float2D]: ...
@overload  # float, output: {"real"}, sort: _Sort (keyword)
def schur(
    a: onp.ToFloat2D,
    output: _OutputReal = "real",
    lwork: onp.ToJustInt | None = None,
    overwrite_a: onp.ToBool = False,
    *,
    sort: _Sort,
    check_finite: onp.ToBool = True,
) -> _Tuple2i[_Inexact2D]: ...
@overload  # complex, output: {"real"}, sort: _Sort (positional)
def schur(
    a: onp.ToComplex2D,
    output: _OutputReal = "real",
    lwork: onp.ToJustInt | None = None,
    overwrite_a: onp.ToBool = False,
    sort: None = None,
    check_finite: onp.ToBool = True,
) -> _Tuple2[_Inexact2D]: ...
@overload  # complex, output: {"real"}, sort: _Sort (keyword)
def schur(
    a: onp.ToComplex2D,
    output: _OutputReal = "real",
    lwork: onp.ToJustInt | None = None,
    overwrite_a: onp.ToBool = False,
    *,
    sort: _Sort,
    check_finite: onp.ToBool = True,
) -> _Tuple2i[_Inexact2D]: ...
@overload  # complex, output: {"complex"}, sort: _Sort (positional)
def schur(
    a: onp.ToComplex2D,
    output: _OutputComplex,
    lwork: onp.ToJustInt | None = None,
    overwrite_a: onp.ToBool = False,
    sort: None = None,
    check_finite: onp.ToBool = True,
) -> _Tuple2[_Complex2D]: ...
@overload  # complex, output: {"complex"}, sort: _Sort (keyword)
def schur(
    a: onp.ToComplex2D,
    output: _OutputComplex,
    lwork: onp.ToJustInt | None = None,
    overwrite_a: onp.ToBool = False,
    *,
    sort: _Sort,
    check_finite: onp.ToBool = True,
) -> _Tuple2i[_Complex2D]: ...

#
def rsf2csf(T: onp.ToFloat2D, Z: onp.ToComplex2D, check_finite: onp.ToBool = True) -> tuple[_Complex2D, _Complex2D]: ...
