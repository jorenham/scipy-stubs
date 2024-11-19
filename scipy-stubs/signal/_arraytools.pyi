from typing import TypeVar

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp

_SCT = TypeVar("_SCT", bound=np.generic)

def axis_slice(
    a: npt.NDArray[_SCT],
    start: op.CanIndex | None = None,
    stop: op.CanIndex | None = None,
    step: op.CanIndex | None = None,
    axis: op.CanIndex = -1,
) -> npt.NDArray[_SCT]: ...
def axis_reverse(a: npt.NDArray[_SCT], axis: op.CanIndex = -1) -> npt.NDArray[_SCT]: ...
def odd_ext(x: npt.NDArray[_SCT], n: onp.ToInt, axis: op.CanIndex = -1) -> npt.NDArray[_SCT]: ...
def even_ext(x: npt.NDArray[_SCT], n: onp.ToInt, axis: op.CanIndex = -1) -> npt.NDArray[_SCT]: ...
def const_ext(x: npt.NDArray[_SCT], n: onp.ToInt, axis: op.CanIndex = -1) -> npt.NDArray[_SCT]: ...
def zero_ext(x: npt.NDArray[_SCT], n: onp.ToInt, axis: op.CanIndex = -1) -> npt.NDArray[_SCT]: ...
