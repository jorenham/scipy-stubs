from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from numpy.linalg import LinAlgError
from scipy._typing import AnyBool, Falsy, Truthy

__all__ = ["LinAlgError", "LinAlgWarning", "norm"]

_Inf: TypeAlias = float
_Order: TypeAlias = Literal["fro", "nuc", 0, 1, -1, 2, -2] | _Inf
_Axis: TypeAlias = op.CanIndex | tuple[op.CanIndex, op.CanIndex]

_SubScalar: TypeAlias = np.complex128 | np.float64 | np.integer[Any] | np.bool_

_NBitT = TypeVar("_NBitT", bound=npt.NBitBase)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

###

class LinAlgWarning(RuntimeWarning): ...

@overload  # scalar, axis: None = ...
def norm(
    a: complex | _SubScalar,
    ord: _Order | None = None,
    axis: None = None,
    keepdims: op.CanBool = False,
    check_finite: AnyBool = True,
) -> np.float64: ...
@overload  # inexact, axis: None = ...
def norm(
    a: np.inexact[_NBitT],
    ord: _Order | None = None,
    axis: None = None,
    keepdims: op.CanBool = False,
    check_finite: AnyBool = True,
) -> np.floating[_NBitT]: ...
@overload  # scalar array, axis: None = ..., keepdims: False = ...
def norm(
    a: onp.CanArrayND[_SubScalar] | onp.SequenceND[onp.CanArrayND[_SubScalar]] | onp.SequenceND[_SubScalar],
    ord: _Order | None = None,
    axis: None = None,
    keepdims: Falsy = False,
    check_finite: AnyBool = True,
) -> np.float64: ...
@overload  # float64-coercible array, keepdims: True (positional)
def norm(
    a: onp.ArrayND[_SubScalar, _ShapeT],
    ord: _Order | None,
    axis: _Axis | None,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.float64, _ShapeT]: ...
@overload  # float64-coercible array, keepdims: True (keyword)
def norm(
    a: onp.ArrayND[_SubScalar, _ShapeT],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.float64, _ShapeT]: ...
@overload  # float64-coercible array-like, keepdims: True (positional)
def norm(
    a: onp.SequenceND[onp.CanArrayND[_SubScalar]] | onp.SequenceND[complex | _SubScalar],
    ord: _Order | None,
    axis: _Axis | None,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # float64-coercible array-like, keepdims: True (keyword)
def norm(
    a: onp.SequenceND[onp.CanArrayND[_SubScalar]] | onp.SequenceND[complex | _SubScalar],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.float64]: ...
@overload  # shaped inexact array, keepdims: True (positional)
def norm(
    a: onp.ArrayND[np.inexact[_NBitT], _ShapeT],
    ord: _Order | None,
    axis: _Axis | None,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.floating[_NBitT], _ShapeT]: ...
@overload  # shaped inexact array, keepdims: True (keyword)
def norm(
    a: onp.ArrayND[np.inexact[_NBitT], _ShapeT],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.floating[_NBitT], _ShapeT]: ...
@overload  # scalar array-like, keepdims: True (positional)
def norm(
    a: onp.SequenceND[onp.CanArrayND[np.inexact[_NBitT]]] | onp.SequenceND[np.inexact[_NBitT]],
    ord: _Order | None,
    axis: _Axis | None,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.floating[_NBitT]]: ...
@overload  # scalar array-like, keepdims: True (keyword)
def norm(
    a: onp.SequenceND[onp.CanArrayND[np.inexact[_NBitT]]] | onp.SequenceND[np.inexact[_NBitT]],
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.floating[_NBitT]]: ...
@overload  # array-like, axis: None = ..., keepdims: False = ...
def norm(
    a: onp.ToComplexND,
    ord: _Order | None = None,
    axis: None = None,
    keepdims: Falsy = False,
    check_finite: AnyBool = True,
) -> np.float64: ...
@overload  # array-like, keepdims: True (positional)
def norm(
    a: onp.ToComplexND,
    ord: _Order | None,
    axis: _Axis | None,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.floating[Any]]: ...
@overload  # array-like, keepdims: True (keyword)
def norm(
    a: onp.ToComplexND,
    ord: _Order | None = None,
    axis: _Axis | None = None,
    *,
    keepdims: Truthy,
    check_finite: AnyBool = True,
) -> onp.ArrayND[np.floating[Any]]: ...
@overload  # catch-all
def norm(
    a: onp.ToArrayND,
    ord: _Order | None = None,
    axis: _Axis | None = None,
    keepdims: AnyBool = False,
    check_finite: AnyBool = True,
) -> np.floating[Any] | onp.ArrayND[np.floating[Any]]: ...
