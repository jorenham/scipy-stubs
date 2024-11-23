from collections.abc import Iterable
from typing import Literal, TypeVar, overload, TypeAlias

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from numpy._typing import _ShapeLike, _DTypeLike
from scipy._typing import Untyped

__all__ = ["chirp", "gausspulse", "sawtooth", "square", "sweep_poly", "unit_impulse"]

_SCT = TypeVar("_SCT", bound=np.generic)

_Truthy: TypeAlias = Literal[1, True]
_Falsy: TypeAlias = Literal[0, False]

def sawtooth(t: onp.ToFloatND, width: onp.ToInt = 1) -> npt.NDArray[np.float64]: ...
def square(t: onp.ToFloatND, duty: onp.ToFloat = 0.5) -> npt.NDArray[np.float64]: ...

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
) -> npt.NDArray[np.float64]: ...
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
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
@overload  # retquad: False (positional), retenv: False (positional)
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToInt,
    bw: onp.ToFloat,
    bwr: onp.ToInt,
    tpr: onp.ToInt,
    retquad: _Falsy,
    retenv: _Truthy,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
@overload  # retquad: True (positional), retenv: False = ...
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToInt,
    bw: onp.ToFloat,
    bwr: onp.ToInt,
    tpr: onp.ToInt,
    retquad: _Truthy,
    retenv: _Falsy = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
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
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
@overload  # retquad: True (positional), retenv: True (positional/keyword)
def gausspulse(
    t: onp.ToFloatND,
    fc: onp.ToInt,
    bw: onp.ToFloat,
    bwr: onp.ToInt,
    tpr: onp.ToInt,
    retquad: _Truthy,
    retenv: _Truthy,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]: ...
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
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]: ...

#
def chirp(
    t: Untyped,
    f0: Untyped,
    t1: Untyped,
    f1: Untyped,
    method: str = "linear",
    phi: int = 0,
    vertex_zero: bool = True,
) -> Untyped: ...

# float -> float and complex -> complex
def sweep_poly(
    t: onp.ToFloatND | onp.ToComplexND,
    poly: onp.ToFloatND | onp.ToComplexND,
    phi: onp.ToInt = 0,
) -> npt.NDArray[np.float64 | np.complex128]: ...

#
@overload  # dtype is not given
def unit_impulse(
    shape: _ShapeLike,
    idx: op.CanIndex | Iterable[op.CanIndex] | Literal["mid"] | None = None,
    dtype: type[float] = float,
) -> npt.NDArray[np.float64]: ...
@overload  # dtype is given
def unit_impulse(
    shape: _ShapeLike,
    idx: op.CanIndex | Iterable[op.CanIndex] | Literal["mid"] | None,
    dtype: _DTypeLike[_SCT],
) -> npt.NDArray[_SCT]: ...
