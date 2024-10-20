from collections.abc import Callable
from typing import Any, Concatenate, overload
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
from scipy._typing import AnyInt, AnyReal

_T = TypeVar("_T", bound=np.floating[Any] | npt.NDArray[np.floating[Any]])

@overload
def _derivative(
    func: Callable[Concatenate[float, ...], AnyReal],
    x0: float,
    dx: AnyReal = 1.0,
    n: AnyInt = 1,
    args: tuple[object, ...] = (),
    order: AnyInt = 3,
) -> np.float64: ...
@overload
def _derivative(
    func: Callable[Concatenate[_T, ...], AnyReal],
    x0: _T,
    dx: AnyReal = 1.0,
    n: AnyInt = 1,
    args: tuple[object, ...] = (),
    order: AnyInt = 3,
) -> _T: ...
