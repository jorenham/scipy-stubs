from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.linalg import LinAlgError  # noqa: ICN003
from scipy._typing import Untyped

__all__ = ["LinAlgError", "LinAlgWarning", "norm"]

class LinAlgWarning(RuntimeWarning): ...

def norm(
    a: npt.ArrayLike,
    ord: Literal["fro", "nuc", 0, 1, -1, 2, -2] | float | None = None,
    axis: Untyped | None = None,
    keepdims: bool = False,
    check_finite: bool = True,
) -> np.float64 | npt.NDArray[np.float64]: ...
