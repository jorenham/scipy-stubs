from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = ["polar"]

def polar(
    a: npt.ArrayLike,
    side: Literal["left", "right"] = "right",
) -> tuple[onp.Array2D[np.inexact[Any]], onp.Array2D[np.inexact[Any]]]: ...
