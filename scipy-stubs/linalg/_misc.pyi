from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import optype.typing as opt
from numpy.linalg import LinAlgError
from scipy._typing import AnyBool

__all__ = ["LinAlgError", "LinAlgWarning", "norm"]

class LinAlgWarning(RuntimeWarning): ...

@overload
def norm(
    a: npt.ArrayLike,
    ord: Literal["fro", "nuc", 0, 1, -1, 2, -2] | float | None = None,
    axis: None = None,
    keepdims: AnyBool = False,
    check_finite: AnyBool = True,
) -> np.float64: ...
@overload
def norm(
    a: npt.ArrayLike,
    ord: Literal["fro", "nuc", 0, 1, -1, 2, -2] | float | None,
    axis: opt.AnyInt | tuple[opt.AnyInt, ...],
    keepdims: AnyBool = False,
    check_finite: AnyBool = True,
) -> np.float64 | onp.ArrayND[np.float64]: ...
@overload
def norm(
    a: npt.ArrayLike,
    ord: Literal["fro", "nuc", 0, 1, -1, 2, -2] | float | None = None,
    *,
    axis: opt.AnyInt | tuple[opt.AnyInt, ...],
    keepdims: AnyBool = False,
    check_finite: AnyBool = True,
) -> np.float64 | onp.ArrayND[np.float64]: ...
