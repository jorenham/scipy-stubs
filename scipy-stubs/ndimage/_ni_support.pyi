from collections.abc import Iterable
from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
from scipy.ndimage._typing import _ScalarValueOut

_Mode: TypeAlias = Literal["nearest", "wrap", "reflect", "grid-mirror", "mirror", "constant", "grid-wrap", "grid-constant"]
_ModeCode: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6]

_T = TypeVar("_T")
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])
_ComplexT = TypeVar("_ComplexT", bound=np.complex64 | np.complex128 | np.clongdouble)
_ScalarT = TypeVar("_ScalarT", bound=np.number[Any] | np.bool_)

#
def _extend_mode_to_code(mode: _Mode) -> _ModeCode: ...
@overload
def _normalize_sequence(input: str, rank: int) -> list[str]: ...
@overload
def _normalize_sequence(input: Iterable[_T], rank: int) -> list[_T]: ...
@overload
def _normalize_sequence(input: _T, rank: int) -> list[_T]: ...

#
@overload
def _get_output(
    output: onp.Array[_ShapeT, _ScalarT] | type[_ScalarT] | np.dtype[_ScalarT] | None,
    input: onp.Array[_ShapeT | tuple[int, ...], _ScalarT],
    shape: _ShapeT | None = None,
    complex_output: Literal[False] = False,
) -> onp.ArrayND[_ScalarT]: ...
@overload
def _get_output(
    output: onp.Array[_ShapeT, _ComplexT] | type[_ComplexT] | np.dtype[_ComplexT] | None,
    input: onp.Array[_ShapeT | tuple[int, ...], _ComplexT],
    shape: _ShapeT | None = None,
    *,
    complex_output: Literal[True],
) -> onp.ArrayND[_ComplexT]: ...
@overload
def _get_output(
    output: onp.Array[_ShapeT, _ScalarValueOut] | type[onp.ToComplex] | np.dtype[_ScalarValueOut] | None,
    input: onp.Array[_ShapeT | tuple[int, ...], _ScalarValueOut],
    shape: _ShapeT | None = None,
    *,
    complex_output: Literal[True],
) -> onp.ArrayND[np.complex64 | np.complex128 | np.clongdouble]: ...

#
@overload
def _check_axes(axes: op.CanIndex, ndim: op.CanIndex) -> tuple[int]: ...
@overload
def _check_axes(axes: Iterable[op.CanIndex] | None, ndim: op.CanIndex) -> tuple[int, ...]: ...
