from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = ["ordqz", "qz"]

_Array_fc_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[np.inexact[npt.NBitBase]]]
_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]

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
