from collections.abc import Iterable
from typing import Literal, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from numpy._typing import _ArrayLike, _DTypeLike
from scipy._typing import AnyShape

__all__ = ["chirp", "gausspulse", "sawtooth", "square", "sweep_poly", "unit_impulse"]

_SCT = TypeVar("_SCT", bound=np.generic)

_Truthy: TypeAlias = Literal[1, True]
_Falsy: TypeAlias = Literal[0, False]
_ArrayLikeFloat: TypeAlias = onp.ToFloat | onp.ToFloatND
_Array_f8: TypeAlias = onp.ArrayND[np.float64]

# Type vars to annotate `chirp`
_NBT1 = TypeVar("_NBT1", bound=npt.NBitBase)
_NBT2 = TypeVar("_NBT2", bound=npt.NBitBase)
_NBT3 = TypeVar("_NBT3", bound=npt.NBitBase)
_NBT4 = TypeVar("_NBT4", bound=npt.NBitBase)
_NBT5 = TypeVar("_NBT5", bound=npt.NBitBase)
_ChirpScalar: TypeAlias = float | np.floating[_NBT1] | np.integer[_NBT1]
_ChirpMethod: TypeAlias = Literal["linear", "quadratic", "logarithmic", "hyperbolic"]

def sawtooth(t: _ArrayLikeFloat, width: _ArrayLikeFloat = 1) -> _Array_f8: ...
def square(t: _ArrayLikeFloat, duty: _ArrayLikeFloat = 0.5) -> _Array_f8: ...

#
@overload  # Other dtypes default to np.float64
def chirp(
    t: onp.SequenceND[float],
    f0: onp.ToFloat,
    t1: onp.ToFloat,
    f1: onp.ToFloat,
    method: _ChirpMethod = "linear",
    phi: onp.ToFloat = 0,
    vertex_zero: op.CanBool = True,
) -> _Array_f8: ...
@overload  # Static type checking for float values
def chirp(
    t: _ArrayLike[np.floating[_NBT1] | np.integer[_NBT1]],
    f0: _ChirpScalar[_NBT2],
    t1: _ChirpScalar[_NBT3],
    f1: _ChirpScalar[_NBT4],
    method: _ChirpMethod = "linear",
    phi: _ChirpScalar[_NBT5] = 0,
    vertex_zero: op.CanBool = True,
) -> onp.ArrayND[np.floating[_NBT1 | _NBT2 | _NBT3 | _NBT4 | _NBT5]]: ...

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
) -> npt.NDArray[_SCT]: ...

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
