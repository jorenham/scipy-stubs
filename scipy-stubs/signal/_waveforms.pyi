from collections.abc import Iterable
from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
from numpy._typing import _DTypeLike
from scipy._typing import AnyShape

__all__ = ["chirp", "gausspulse", "sawtooth", "square", "sweep_poly", "unit_impulse"]

_SCT = TypeVar("_SCT", bound=np.generic)

_Truthy: TypeAlias = Literal[1, True]
_Falsy: TypeAlias = Literal[0, False]
_ArrayLikeFloat: TypeAlias = onp.ToFloat | onp.ToFloatND
_Array_f: TypeAlias = onp.ArrayND[np.floating[Any]]
_Array_f8: TypeAlias = onp.ArrayND[np.float64]

_ChirpMethod: TypeAlias = Literal["linear", "quadratic", "logarithmic", "hyperbolic"]

def sawtooth(t: _ArrayLikeFloat, width: _ArrayLikeFloat = 1) -> _Array_f8: ...
def square(t: _ArrayLikeFloat, duty: _ArrayLikeFloat = 0.5) -> _Array_f8: ...

#
@overload  # Arrays
def chirp(
    t: onp.ToFloatND,
    f0: onp.ToFloat,
    t1: onp.ToFloat,
    f1: onp.ToFloat,
    method: _ChirpMethod = "linear",
    phi: onp.ToFloat = 0,
    vertex_zero: op.CanBool = True,
) -> _Array_f: ...
@overload  # Scalars
def chirp(
    t: onp.ToFloat,
    f0: onp.ToFloat,
    t1: onp.ToFloat,
    f1: onp.ToFloat,
    method: _ChirpMethod = "linear",
    phi: onp.ToFloat = 0,
    vertex_zero: op.CanBool = True,
) -> np.floating[Any]: ...

#
def sweep_poly(
    t: _ArrayLikeFloat,
    poly: onp.ToFloatND | np.poly1d,
    phi: onp.ToFloat = 0,
) -> _Array_f8: ...

#
@overload  # dtype is not given
def unit_impulse(
    shape: AnyShape,
    idx: op.CanIndex | Iterable[op.CanIndex] | Literal["mid"] | None = None,
    dtype: type[float] = ...,
) -> _Array_f8: ...
@overload  # dtype is given
def unit_impulse(
    shape: AnyShape,
    idx: op.CanIndex | Iterable[op.CanIndex] | Literal["mid"] | None,
    dtype: _DTypeLike[_SCT],
) -> onp.ArrayND[_SCT]: ...

# Overloads for gausspulse when `t` is `"cutoff"`
@overload  # retquad: False = ..., retenv: False = ...
def gausspulse(
    t: Literal["cutoff"],
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    retquad: op.CanBool = False,
    retenv: op.CanBool = False,
) -> np.float64: ...

# Overloads for gausspulse when `t` is scalar
@overload  # retquad: False = ..., retenv: False = ...
def gausspulse(
    t: onp.ToFloat,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    retquad: _Falsy = False,
    retenv: _Falsy = False,
) -> np.float64: ...
@overload  # retquad: False = ..., retenv: True (keyword)
def gausspulse(
    t: onp.ToFloat,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    retquad: _Falsy = False,
    *,
    retenv: _Truthy,
) -> tuple[np.float64, np.float64]: ...
@overload  # retquad: False (positional), retenv: False (positional)
def gausspulse(
    t: onp.ToFloat,
    fc: onp.ToFloat,
    bw: onp.ToFloat,
    bwr: onp.ToFloat,
    tpr: onp.ToFloat,
    retquad: _Falsy,
    retenv: _Truthy,
) -> tuple[np.float64, np.float64]: ...
@overload  # retquad: True (positional), retenv: False = ...
def gausspulse(
    t: onp.ToFloat,
    fc: onp.ToFloat,
    bw: onp.ToFloat,
    bwr: onp.ToFloat,
    tpr: onp.ToFloat,
    retquad: _Truthy,
    retenv: _Falsy = False,
) -> tuple[np.float64, np.float64]: ...
@overload  # retquad: True (keyword), retenv: False = ...
def gausspulse(
    t: onp.ToFloat,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    *,
    retquad: _Truthy,
    retenv: _Falsy = False,
) -> tuple[np.float64, np.float64]: ...
@overload  # retquad: True (positional), retenv: True (positional/keyword)
def gausspulse(
    t: onp.ToFloat,
    fc: onp.ToFloat,
    bw: onp.ToFloat,
    bwr: onp.ToFloat,
    tpr: onp.ToFloat,
    retquad: _Truthy,
    retenv: _Truthy,
) -> tuple[np.float64, np.float64, np.float64]: ...
@overload  # retquad: True (keyword), retenv: True
def gausspulse(
    t: onp.ToFloat,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    *,
    retquad: _Truthy,
    retenv: _Truthy,
) -> tuple[np.float64, np.float64, np.float64]: ...

# Overloads for `gausspulse` when `t` is a non-scalar array like
@overload  # retquad: False = ..., retenv: False = ...
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    retquad: _Falsy = False,
    retenv: _Falsy = False,
) -> _Array_f8: ...
@overload  # retquad: False = ..., retenv: True (keyword)
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    retquad: _Falsy = False,
    *,
    retenv: _Truthy,
) -> tuple[_Array_f8, _Array_f8]: ...
@overload  # retquad: False (positional), retenv: False (positional)
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat,
    bw: onp.ToFloat,
    bwr: onp.ToFloat,
    tpr: onp.ToFloat,
    retquad: _Falsy,
    retenv: _Truthy,
) -> tuple[_Array_f8, _Array_f8]: ...
@overload  # retquad: True (positional), retenv: False = ...
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat,
    bw: onp.ToFloat,
    bwr: onp.ToFloat,
    tpr: onp.ToFloat,
    retquad: _Truthy,
    retenv: _Falsy = False,
) -> tuple[_Array_f8, _Array_f8]: ...
@overload  # retquad: True (keyword), retenv: False = ...
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    *,
    retquad: _Truthy,
    retenv: _Falsy = False,
) -> tuple[_Array_f8, _Array_f8]: ...
@overload  # retquad: True (positional), retenv: True (positional/keyword)
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat,
    bw: onp.ToFloat,
    bwr: onp.ToFloat,
    tpr: onp.ToFloat,
    retquad: _Truthy,
    retenv: _Truthy,
) -> tuple[_Array_f8, _Array_f8, _Array_f8]: ...
@overload  # retquad: True (keyword), retenv: True
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat = -60,
    *,
    retquad: _Truthy,
    retenv: _Truthy,
) -> tuple[_Array_f8, _Array_f8, _Array_f8]: ...
