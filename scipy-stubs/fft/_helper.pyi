from types import ModuleType
from typing import Any, Literal, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
from numpy._typing import _ArrayLike
from scipy._typing import AnyShape

_SCT = TypeVar("_SCT", bound=np.inexact[Any])

def next_fast_len(target: op.CanIndex, real: op.CanBool = False) -> int: ...
def prev_fast_len(target: op.CanIndex, real: op.CanBool = False) -> int: ...

# TODO(jorenham): Array API support (for `xp`)
# https://github.com/jorenham/scipy-stubs/issues/140
@overload  # xp: None -> np.fft.fftfreq
def fftfreq(
    n: int | np.integer[Any],
    d: onp.ToFloat = 1.0,
    *,
    xp: None = None,
    device: Literal["cpu"] | None = None,
) -> onp.Array1D[np.float64]: ...
@overload  # xp: ModuleType -> xp.fft.fftfreq
def fftfreq(n: int, d: float = 1.0, *, xp: ModuleType, device: object | None = None) -> Any: ...  # noqa: ANN401

# TODO(jorenham): Array API support (for `xp`)
# https://github.com/jorenham/scipy-stubs/issues/140
@overload  # np.fft.rfftfreq
def rfftfreq(
    n: int | np.integer[Any],
    d: onp.ToFloat = 1.0,
    *,
    xp: None = None,
    device: Literal["cpu"] | None = None,
) -> onp.Array1D[np.float64]: ...
@overload  # xp.fft.fftfreq
def rfftfreq(n: int, d: float = 1.0, *, xp: ModuleType, device: object | None = None) -> Any: ...  # noqa: ANN401

# TODO(jorenham): Array API support (for `x`)
# https://github.com/jorenham/scipy-stubs/issues/140
@overload
def fftshift(x: onp.ToIntND | onp.SequenceND[float], axes: AnyShape | None = None) -> onp.ArrayND[np.float64]: ...
@overload
def fftshift(x: onp.SequenceND[complex], axes: AnyShape | None = None) -> onp.ArrayND[np.complex128 | np.float64]: ...
@overload
def fftshift(x: _ArrayLike[_SCT], axes: AnyShape | None = None) -> onp.ArrayND[_SCT]: ...
@overload
def fftshift(x: onp.ToComplexND, axes: AnyShape | None = None) -> onp.ArrayND[np.inexact[Any]]: ...

# TODO(jorenham): Array API support (for `x`)
# https://github.com/jorenham/scipy-stubs/issues/140
@overload
def ifftshift(x: onp.ToIntND | onp.SequenceND[float], axes: AnyShape | None = None) -> onp.ArrayND[np.float64]: ...
@overload
def ifftshift(x: onp.SequenceND[complex], axes: AnyShape | None = None) -> onp.ArrayND[np.complex128 | np.float64]: ...
@overload
def ifftshift(x: _ArrayLike[_SCT], axes: AnyShape | None = None) -> onp.ArrayND[_SCT]: ...
@overload
def ifftshift(x: onp.ToComplexND, axes: AnyShape | None = None) -> onp.ArrayND[np.inexact[Any]]: ...
