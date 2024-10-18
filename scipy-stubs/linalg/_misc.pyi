from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import optype.typing as opt
from numpy.linalg import LinAlgError  # noqa: ICN003

__all__ = ["LinAlgError", "LinAlgWarning", "norm"]

class LinAlgWarning(RuntimeWarning): ...

@overload
def norm(
    a: npt.ArrayLike,
    ord: Literal["fro", "nuc", 0, 1, -1, 2, -2] | float | None = None,
    axis: None = None,
    keepdims: bool = False,
    check_finite: bool = True,
) -> np.float64: ...
@overload
def norm(
    a: npt.ArrayLike,
    ord: Literal["fro", "nuc", 0, 1, -1, 2, -2] | float | None,
    axis: opt.AnyInt | tuple[opt.AnyInt, ...],
    keepdims: bool = False,
    check_finite: bool = True,
) -> np.float64 | npt.NDArray[np.float64]: ...
@overload
def norm(
    a: npt.ArrayLike,
    ord: Literal["fro", "nuc", 0, 1, -1, 2, -2] | float | None = None,
    *,
    axis: opt.AnyInt | tuple[opt.AnyInt, ...],
    keepdims: bool = False,
    check_finite: bool = True,
) -> np.float64 | npt.NDArray[np.float64]: ...
