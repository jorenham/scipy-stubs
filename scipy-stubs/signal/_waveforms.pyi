from collections.abc import Iterable
from typing import Literal, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from numpy._typing import _DTypeLike, _ShapeLike

__all__ = ["chirp", "gausspulse", "sawtooth", "square", "sweep_poly", "unit_impulse"]

_SCT = TypeVar("_SCT", bound=np.generic)

_Truthy: TypeAlias = Literal[1, True]
_Falsy: TypeAlias = Literal[0, False]
_Array_f8: TypeAlias = npt.NDArray[np.float64]

def sawtooth(t: onp.ToFloatND, width: onp.ToInt = 1) -> _Array_f8: ...
def square(t: onp.ToFloatND, duty: onp.ToFloat = 0.5) -> _Array_f8: ...

#
@overload  # retquad: False = ..., retenv: False = ...
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToInt = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToInt = -6,
    tpr: onp.ToInt = -60,
    retquad: _Falsy = False,
    retenv: _Falsy = False,
) -> _Array_f8: ...
@overload  # retquad: False = ..., retenv: True (keyword)
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToInt = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToInt = -6,
    tpr: onp.ToInt = -60,
    retquad: _Falsy = False,
    *,
    retenv: _Truthy,
) -> tuple[_Array_f8, _Array_f8]: ...
@overload  # retquad: False (positional), retenv: False (positional)
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToInt,
    bw: onp.ToFloat,
    bwr: onp.ToInt,
    tpr: onp.ToInt,
    retquad: _Falsy,
    retenv: _Truthy,
) -> tuple[_Array_f8, _Array_f8]: ...
@overload  # retquad: True (positional), retenv: False = ...
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToInt,
    bw: onp.ToFloat,
    bwr: onp.ToInt,
    tpr: onp.ToInt,
    retquad: _Truthy,
    retenv: _Falsy = False,
) -> tuple[_Array_f8, _Array_f8]: ...
@overload  # retquad: True (keyword), retenv: False = ...
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToInt = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToInt = -6,
    tpr: onp.ToInt = -60,
    *,
    retquad: _Truthy,
    retenv: _Falsy = False,
) -> tuple[_Array_f8, _Array_f8]: ...
@overload  # retquad: True (positional), retenv: True (positional/keyword)
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToInt,
    bw: onp.ToFloat,
    bwr: onp.ToInt,
    tpr: onp.ToInt,
    retquad: _Truthy,
    retenv: _Truthy,
) -> tuple[_Array_f8, _Array_f8, _Array_f8]: ...
@overload  # retquad: True (keyword), retenv: True
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToInt = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToInt = -6,
    tpr: onp.ToInt = -60,
    *,
    retquad: _Truthy,
    retenv: _Truthy,
) -> tuple[_Array_f8, _Array_f8, _Array_f8]: ...

# float16 -> float16, float32 -> float32, ... -> float64
def chirp(
    t: onp.ToFloatND,
    f0: onp.ToFloat,
    t1: onp.ToFloat,
    f1: onp.ToFloat,
    method: Literal["linear", "quadratic", "logarithmic", "hyperbolic"] = "linear",
    phi: onp.ToInt = 0,
    vertex_zero: op.CanBool = True,
) -> npt.NDArray[np.float16 | np.float32 | np.float64]: ...
def sweep_poly(
    t: onp.ToFloatND,
    poly: onp.ToFloatND,
    phi: onp.ToInt = 0,
) -> _Array_f8: ...

#
@overload  # dtype is not given
def unit_impulse(
    shape: _ShapeLike,
    idx: op.CanIndex | Iterable[op.CanIndex] | Literal["mid"] | None = None,
    dtype: type[float] = ...,
) -> _Array_f8: ...
@overload  # dtype is given
def unit_impulse(
    shape: _ShapeLike,
    idx: op.CanIndex | Iterable[op.CanIndex] | Literal["mid"] | None,
    dtype: _DTypeLike[_SCT],
) -> npt.NDArray[_SCT]: ...
