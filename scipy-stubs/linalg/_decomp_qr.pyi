from typing import Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt

__all__ = ["qr", "qr_multiply", "rq"]

_Array_i_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.intp]]
_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]

# 5/11 of these overloads could've been avoided with keyword-only parameters; so please use them ;)
@overload
def qr(
    a: npt.ArrayLike,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: Literal["full", "economic"] = "full",
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d]: ...
@overload
def qr(
    a: npt.ArrayLike,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["r"],
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d]: ...
@overload
def qr(
    a: npt.ArrayLike,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["r"],
    /,
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d]: ...
@overload
def qr(
    a: npt.ArrayLike,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["raw"],
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[tuple[_Array_fc_2d, _Array_fc_2d], _Array_fc_2d]: ...
@overload
def qr(
    a: npt.ArrayLike,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["raw"],
    /,
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[tuple[_Array_fc_2d, _Array_fc_2d], _Array_fc_2d]: ...
@overload
def qr(
    a: npt.ArrayLike,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: Literal["full", "economic"] = "full",
    *,
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d, _Array_i_1d]: ...
@overload
def qr(
    a: npt.ArrayLike,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["full", "economic"],
    pivoting: Literal[True],
    /,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d, _Array_i_1d]: ...
@overload
def qr(
    a: npt.ArrayLike,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["r"],
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_i_1d]: ...
@overload
def qr(
    a: npt.ArrayLike,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["r"],
    pivoting: Literal[True],
    /,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_i_1d]: ...
@overload
def qr(
    a: npt.ArrayLike,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["raw"],
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[tuple[_Array_fc_2d, _Array_fc_2d], _Array_fc_2d, _Array_i_1d]: ...
@overload
def qr(
    a: npt.ArrayLike,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["raw"],
    pivoting: Literal[True],
    /,
    check_finite: bool = True,
) -> tuple[tuple[_Array_fc_2d, _Array_fc_2d], _Array_fc_2d, _Array_i_1d]: ...
@overload
def qr_multiply(
    a: npt.ArrayLike,
    c: npt.ArrayLike,
    mode: Literal["left", "right"] = "right",
    pivoting: Literal[False] = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[_Array_fc_2d, _Array_fc_2d]: ...
@overload
def qr_multiply(
    a: npt.ArrayLike,
    c: npt.ArrayLike,
    mode: Literal["left", "right"] = "right",
    *,
    pivoting: Literal[True],
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[_Array_fc_2d, _Array_fc_2d, _Array_i_1d]: ...
@overload
def qr_multiply(
    a: npt.ArrayLike,
    c: npt.ArrayLike,
    mode: Literal["left", "right"],
    pivoting: Literal[True],
    /,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[_Array_fc_2d, _Array_fc_2d, _Array_i_1d]: ...
@overload
def rq(
    a: npt.ArrayLike,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: Literal["full", "economic"] = "full",
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d]: ...
@overload
def rq(
    a: npt.ArrayLike,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["r"],
    check_finite: bool = True,
) -> _Array_fc_2d: ...
@overload
def rq(
    a: npt.ArrayLike,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["r"],
    /,
    check_finite: bool = True,
) -> _Array_fc_2d: ...
