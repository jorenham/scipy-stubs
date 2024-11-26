from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = ["ordqz", "qz"]

_Array_fc_1d: TypeAlias = onp.Array2D[np.inexact[Any]]
_Array_fc_2d: TypeAlias = onp.Array2D[np.inexact[Any]]

def qz(
    A: npt.ArrayLike,
    B: npt.ArrayLike,
    output: Literal["real", "complex"] = "real",
    lwork: int | None = None,
    sort: None = None,  # disabled for now
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d, _Array_fc_2d, _Array_fc_2d]: ...
def ordqz(
    A: npt.ArrayLike,
    B: npt.ArrayLike,
    sort: Literal["lhp", "rhp", "iuc", "ouc"] | Callable[[float, float], bool] = "lhp",
    output: Literal["real", "complex"] = "real",
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d, _Array_fc_1d, _Array_fc_1d, _Array_fc_2d, _Array_fc_2d]: ...
