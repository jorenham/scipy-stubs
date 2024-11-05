from types import ModuleType
from typing import Any, Literal, TypeVar, overload

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onpt
from numpy._typing import _ArrayLike, _ArrayLikeInt_co, _ArrayLikeNumber_co, _NestedSequence
from scipy._typing import AnyReal, AnyShape

_SCT = TypeVar("_SCT", bound=np.inexact[Any])

def next_fast_len(target: op.CanIndex, real: op.CanBool = False) -> int: ...
def prev_fast_len(target: op.CanIndex, real: op.CanBool = False) -> int: ...

# TODO(jorenham): Array API support (for `xp`)
# https://github.com/jorenham/scipy-stubs/issues/140
@overload  # xp: None -> np.fft.fftfreq
def fftfreq(
    n: int | np.integer[Any],
    d: AnyReal = 1.0,
    *,
    xp: None = None,
    device: Literal["cpu"] | None = None,
) -> onpt.Array[tuple[int], np.float64]: ...
@overload  # xp: ModuleType -> xp.fft.fftfreq
def fftfreq(n: int, d: float = 1.0, *, xp: ModuleType, device: object | None = None) -> Any: ...  # noqa: ANN401

# TODO(jorenham): Array API support (for `xp`)
# https://github.com/jorenham/scipy-stubs/issues/140
@overload  # np.fft.rfftfreq
def rfftfreq(
    n: int | np.integer[Any],
    d: AnyReal = 1.0,
    *,
    xp: None = None,
    device: Literal["cpu"] | None = None,
) -> onpt.Array[tuple[int], np.float64]: ...
@overload  # xp.fft.fftfreq
def rfftfreq(n: int, d: float = 1.0, *, xp: ModuleType, device: object | None = None) -> Any: ...  # noqa: ANN401

# TODO(jorenham): Array API support (for `x`)
# https://github.com/jorenham/scipy-stubs/issues/140
@overload
def fftshift(x: _ArrayLike[_SCT], axes: AnyShape | None = None) -> npt.NDArray[_SCT]: ...
@overload
def fftshift(x: _ArrayLikeInt_co | _NestedSequence[float], axes: AnyShape | None = None) -> npt.NDArray[np.float64]: ...
@overload
def fftshift(x: _NestedSequence[complex], axes: AnyShape | None = None) -> npt.NDArray[np.complex128 | np.float64]: ...
@overload
def fftshift(x: _ArrayLikeNumber_co, axes: AnyShape | None = None) -> npt.NDArray[np.inexact[Any]]: ...

# TODO(jorenham): Array API support (for `x`)
# https://github.com/jorenham/scipy-stubs/issues/140
@overload
def ifftshift(x: _ArrayLike[_SCT], axes: AnyShape | None = None) -> npt.NDArray[_SCT]: ...
@overload
def ifftshift(x: _ArrayLikeInt_co | _NestedSequence[float], axes: AnyShape | None = None) -> npt.NDArray[np.float64]: ...
@overload
def ifftshift(x: _NestedSequence[complex], axes: AnyShape | None = None) -> npt.NDArray[np.complex128 | np.float64]: ...
@overload
def ifftshift(x: _ArrayLikeNumber_co, axes: AnyShape | None = None) -> npt.NDArray[np.inexact[Any]]: ...
