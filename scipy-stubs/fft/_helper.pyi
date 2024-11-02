from types import ModuleType
from typing import Any, TypeVar, overload

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLike, _ArrayLikeInt_co, _ArrayLikeNumber_co, _NestedSequence
from optype import CanBool
from scipy._typing import AnyInt, AnyReal, AnyShape

_SCT = TypeVar("_SCT", bound=np.inexact[Any])

def next_fast_len(target: AnyInt, real: CanBool = False) -> None: ...
def prev_fast_len(target: AnyInt, real: CanBool = False) -> None: ...

#
def fftfreq(
    n: AnyInt,
    d: AnyReal = 1.0,
    *,
    xp: ModuleType | None = None,
    device: object | None = None,
) -> npt.NDArray[np.float64]: ...
def rfftfreq(
    n: AnyInt,
    d: AnyReal = 1.0,
    *,
    xp: ModuleType | None = None,
    device: object | None = None,
) -> npt.NDArray[np.float64]: ...

#
@overload
def fftshift(x: _ArrayLike[_SCT], axes: AnyShape | None = None) -> npt.NDArray[_SCT]: ...
@overload
def fftshift(x: _ArrayLikeInt_co | _NestedSequence[float], axes: AnyShape | None = None) -> npt.NDArray[np.float64]: ...
@overload
def fftshift(x: _NestedSequence[complex], axes: AnyShape | None = None) -> npt.NDArray[np.complex128 | np.float64]: ...
@overload
def fftshift(x: _ArrayLikeNumber_co, axes: AnyShape | None = None) -> npt.NDArray[np.inexact[Any]]: ...

#
@overload
def ifftshift(x: _ArrayLike[_SCT], axes: AnyShape | None = None) -> npt.NDArray[_SCT]: ...
@overload
def ifftshift(x: _ArrayLikeInt_co | _NestedSequence[float], axes: AnyShape | None = None) -> npt.NDArray[np.float64]: ...
@overload
def ifftshift(x: _NestedSequence[complex], axes: AnyShape | None = None) -> npt.NDArray[np.complex128 | np.float64]: ...
@overload
def ifftshift(x: _ArrayLikeNumber_co, axes: AnyShape | None = None) -> npt.NDArray[np.inexact[Any]]: ...
