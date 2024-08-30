from collections.abc import Callable
from typing import Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt

__all__ = ["rsf2csf", "schur"]

_Array_fc_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.inexact[npt.NBitBase]]]
_Array_c_2d: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.complexfloating[npt.NBitBase, npt.NBitBase]]]

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
