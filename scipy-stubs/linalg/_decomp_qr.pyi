from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt

__all__ = ["qr", "qr_multiply", "rq"]

_ArrayLike_2d_fc: TypeAlias = onpt.AnyNumberArray | Sequence[Sequence[complex | np.number[Any]]]
_Array_i_1d: TypeAlias = onpt.Array[tuple[int], np.int_]
_Array_fc_2d: TypeAlias = onpt.Array[tuple[int, int], np.inexact[npt.NBitBase]]

# 5/11 of these overloads could've been avoided with keyword-only parameters; so please use them ;)
@overload
def qr(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: Literal["full", "economic"] = "full",
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d]: ...
@overload
def qr(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["r"],
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d]: ...
@overload
def qr(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["r"],
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d]: ...
@overload
def qr(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["raw"],
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[tuple[_Array_fc_2d, _Array_fc_2d], _Array_fc_2d]: ...
@overload
def qr(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["raw"],
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[tuple[_Array_fc_2d, _Array_fc_2d], _Array_fc_2d]: ...
@overload
def qr(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["full", "economic"],
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d, _Array_i_1d]: ...
@overload
def qr(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: Literal["full", "economic"] = "full",
    *,
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d, _Array_i_1d]: ...
@overload
def qr(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["r"],
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_i_1d]: ...
@overload
def qr(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["r"],
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_i_1d]: ...
@overload
def qr(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["raw"],
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[tuple[_Array_fc_2d, _Array_fc_2d], _Array_fc_2d, _Array_i_1d]: ...
@overload
def qr(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["raw"],
    pivoting: Literal[True],
    check_finite: bool = True,
) -> tuple[tuple[_Array_fc_2d, _Array_fc_2d], _Array_fc_2d, _Array_i_1d]: ...

#
@overload
def qr_multiply(
    a: _ArrayLike_2d_fc,
    c: npt.ArrayLike,
    mode: Literal["left", "right"] = "right",
    pivoting: Literal[False] = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[_Array_fc_2d, _Array_fc_2d]: ...
@overload
def qr_multiply(
    a: _ArrayLike_2d_fc,
    c: npt.ArrayLike,
    mode: Literal["left", "right"],
    pivoting: Literal[True],
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[_Array_fc_2d, _Array_fc_2d, _Array_i_1d]: ...
@overload
def qr_multiply(
    a: _ArrayLike_2d_fc,
    c: npt.ArrayLike,
    mode: Literal["left", "right"] = "right",
    *,
    pivoting: Literal[True],
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[_Array_fc_2d, _Array_fc_2d, _Array_i_1d]: ...

#
@overload
def rq(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool = False,
    lwork: int | None = None,
    mode: Literal["full", "economic"] = "full",
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d]: ...
@overload
def rq(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool,
    lwork: int | None,
    mode: Literal["r"],
    check_finite: bool = True,
) -> _Array_fc_2d: ...
@overload
def rq(
    a: _ArrayLike_2d_fc,
    overwrite_a: bool = False,
    lwork: int | None = None,
    *,
    mode: Literal["r"],
    check_finite: bool = True,
) -> _Array_fc_2d: ...
