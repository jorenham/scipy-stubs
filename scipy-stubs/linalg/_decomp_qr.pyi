from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = ["qr", "qr_multiply", "rq"]

_Inexact2D: TypeAlias = onp.Array2D[np.inexact[Any]]

# 5/11 of these overloads could've been avoided with keyword-only parameters; so please use them ;)
@overload
def qr(
    a: onp.ToComplex2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: Literal["full", "economic"] = "full",
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[_Inexact2D, _Inexact2D]: ...
@overload
def qr(
    a: onp.ToComplex2D,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["r"],
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[_Inexact2D]: ...
@overload
def qr(
    a: onp.ToComplex2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["r"],
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[_Inexact2D]: ...
@overload
def qr(
    a: onp.ToComplex2D,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["raw"],
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[tuple[_Inexact2D, _Inexact2D], _Inexact2D]: ...
@overload
def qr(
    a: onp.ToComplex2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["raw"],
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[tuple[_Inexact2D, _Inexact2D], _Inexact2D]: ...
@overload
def qr(
    a: onp.ToComplex2D,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["full", "economic"],
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[_Inexact2D, _Inexact2D, onp.Array1D[np.int_]]: ...
@overload
def qr(
    a: onp.ToComplex2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: Literal["full", "economic"] = "full",
    *,
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[_Inexact2D, _Inexact2D, onp.Array1D[np.int_]]: ...
@overload
def qr(
    a: onp.ToComplex2D,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["r"],
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[_Inexact2D, onp.Array1D[np.int_]]: ...
@overload
def qr(
    a: onp.ToComplex2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["r"],
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[_Inexact2D, onp.Array1D[np.int_]]: ...
@overload
def qr(
    a: onp.ToComplex2D,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["raw"],
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[tuple[_Inexact2D, _Inexact2D], _Inexact2D, onp.Array1D[np.int_]]: ...
@overload
def qr(
    a: onp.ToComplex2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["raw"],
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[tuple[_Inexact2D, _Inexact2D], _Inexact2D, onp.Array1D[np.int_]]: ...

#
@overload
def qr_multiply(
    a: onp.ToComplex2D,
    c: npt.ArrayLike,
    mode: Literal["left", "right"] = "right",
    pivoting: Literal[False] = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[_Inexact2D, _Inexact2D]: ...
@overload
def qr_multiply(
    a: onp.ToComplex2D,
    c: npt.ArrayLike,
    mode: Literal["left", "right"],
    pivoting: Literal[True],
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[_Inexact2D, _Inexact2D, onp.Array1D[np.int_]]: ...
@overload
def qr_multiply(
    a: onp.ToComplex2D,
    c: npt.ArrayLike,
    mode: Literal["left", "right"] = "right",
    *,
    pivoting: Literal[True],
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[_Inexact2D, _Inexact2D, onp.Array1D[np.int_]]: ...

#
@overload
def rq(
    a: onp.ToComplex2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: Literal["full", "economic"] = "full",
    check_finite: bool = True,
) -> tuple[_Inexact2D, _Inexact2D]: ...
@overload
def rq(
    a: onp.ToComplex2D,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["r"],
    check_finite: bool = True,
) -> _Inexact2D: ...
@overload
def rq(
    a: onp.ToComplex2D,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["r"],
    check_finite: bool = True,
) -> _Inexact2D: ...
