from collections.abc import Iterable
from typing import Literal, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from numpy._typing import _DTypeLike
from scipy._typing import AnyShape

__all__ = ["chirp", "gausspulse", "sawtooth", "square", "sweep_poly", "unit_impulse"]

_SCT = TypeVar("_SCT", bound=np.generic)

_Truthy: TypeAlias = Literal[1, True]
_Falsy: TypeAlias = Literal[0, False]
_Array_f8: TypeAlias = onp.ArrayND[np.float64]

def sawtooth(t: onp.ToFloat | onp.ToFloatND, width: onp.ToFloat | onp.ToFloatND = 1) -> _Array_f8: ...
def square(t: onp.ToFloat | onp.ToFloatND, duty: onp.ToFloat | onp.ToFloatND = 0.5) -> _Array_f8: ...

#
@overload  # retquad: False = ..., retenv: False = ...
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat | Literal["cutoff"] = -60,
    retquad: _Falsy = False,
    retenv: _Falsy = False,
) -> _Array_f8: ...
@overload  # retquad: False = ..., retenv: True (keyword)
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat | Literal["cutoff"] = -60,
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
    tpr: onp.ToFloat | Literal["cutoff"],
    retquad: _Falsy,
    retenv: _Truthy,
) -> tuple[_Array_f8, _Array_f8]: ...
@overload  # retquad: True (positional), retenv: False = ...
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat,
    bw: onp.ToFloat,
    bwr: onp.ToFloat,
    tpr: onp.ToFloat | Literal["cutoff"],
    retquad: _Truthy,
    retenv: _Falsy = False,
) -> tuple[_Array_f8, _Array_f8]: ...
@overload  # retquad: True (keyword), retenv: False = ...
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat | Literal["cutoff"] = -60,
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
    tpr: onp.ToFloat | Literal["cutoff"],
    retquad: _Truthy,
    retenv: _Truthy,
) -> tuple[_Array_f8, _Array_f8, _Array_f8]: ...
@overload  # retquad: True (keyword), retenv: True
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToFloat = 1000,
    bw: onp.ToFloat = 0.5,
    bwr: onp.ToFloat = -6,
    tpr: onp.ToFloat | Literal["cutoff"] = -60,
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
    phi: onp.ToFloat = 0,
    vertex_zero: op.CanBool = True,
) -> npt.NDArray[np.float16 | np.float32 | np.float64]: ...
def sweep_poly(
    t: onp.ToFloat | onp.ToFloatND,
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
) -> npt.NDArray[_SCT]: ...
