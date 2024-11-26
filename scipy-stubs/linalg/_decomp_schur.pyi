from collections.abc import Callable
from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

__all__ = ["rsf2csf", "schur"]

_Array_fc_2d: TypeAlias = onp.Array2D[np.inexact[Any]]
_Array_c_2d: TypeAlias = onp.Array2D[np.complexfloating[Any, Any]]

@overload
def schur(
    a: npt.ArrayLike,
    output: Literal["real", "complex"] = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    sort: None = None,
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d]: ...
@overload
def schur(
    a: npt.ArrayLike,
    output: Literal["real", "complex"] = "real",
    lwork: int | None = None,
    overwrite_a: bool = False,
    *,
    sort: Literal["lhp", "rhp", "iuc", "ouc"] | Callable[[float, float], bool],
    check_finite: bool = True,
) -> tuple[_Array_fc_2d, _Array_fc_2d, int]: ...
def rsf2csf(T: npt.ArrayLike, Z: npt.ArrayLike, check_finite: bool = True) -> tuple[_Array_c_2d, _Array_c_2d]: ...
