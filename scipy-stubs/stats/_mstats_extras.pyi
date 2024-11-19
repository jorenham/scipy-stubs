from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import AnyReal

__all__ = [
    "compare_medians_ms",
    "hdmedian",
    "hdquantiles",
    "hdquantiles_sd",
    "idealfourths",
    "median_cihs",
    "mjci",
    "mquantiles_cimj",
    "rsh",
    "trimmed_mean_ci",
]

@overload
def hdquantiles(
    data: _ArrayLikeFloat_co,
    prob: _ArrayLikeFloat_co = [0.25, 0.5, 0.75],
    axis: op.CanIndex | None = None,
    var: Literal[0, False] = False,
) -> np.ma.MaskedArray[onp.AtLeast1D, np.dtype[np.float64]]: ...
@overload
def hdquantiles(
    data: _ArrayLikeFloat_co,
    prob: _ArrayLikeFloat_co,
    axis: op.CanIndex | None,
    var: Literal[1, True],
) -> np.ma.MaskedArray[onp.AtLeast2D, np.dtype[np.float64]]: ...
@overload
def hdquantiles(
    data: _ArrayLikeFloat_co,
    prob: _ArrayLikeFloat_co = [0.25, 0.5, 0.75],
    axis: op.CanIndex | None = None,
    *,
    var: Literal[1, True],
) -> np.ma.MaskedArray[onp.AtLeast2D, np.dtype[np.float64]]: ...

#
@overload
def hdmedian(
    data: _ArrayLikeFloat_co,
    axis: op.CanIndex | None = -1,
    var: Literal[0, False] = False,
) -> np.ma.MaskedArray[onp.AtLeast0D, np.dtype[np.float64]]: ...
@overload
def hdmedian(
    data: _ArrayLikeFloat_co,
    axis: op.CanIndex | None,
    var: Literal[1, True],
) -> np.ma.MaskedArray[onp.AtLeast1D, np.dtype[np.float64]]: ...
@overload
def hdmedian(
    data: _ArrayLikeFloat_co,
    axis: op.CanIndex | None = -1,
    *,
    var: Literal[1, True],
) -> np.ma.MaskedArray[onp.AtLeast1D, np.dtype[np.float64]]: ...

#
def hdquantiles_sd(
    data: _ArrayLikeFloat_co,
    prob: _ArrayLikeFloat_co = [0.25, 0.5, 0.75],
    axis: op.CanIndex | None = None,
) -> np.ma.MaskedArray[onp.AtLeast1D, np.dtype[np.float64]]: ...

#
def trimmed_mean_ci(
    data: _ArrayLikeFloat_co,
    limits: tuple[AnyReal, AnyReal] | None = (0.2, 0.2),
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    alpha: float | np.floating[Any] = 0.05,
    axis: op.CanIndex | None = None,
) -> npt.NDArray[np.float64]: ...

#
def mjci(
    data: _ArrayLikeFloat_co,
    prob: _ArrayLikeFloat_co = [0.25, 0.5, 0.75],
    axis: op.CanIndex | None = None,
) -> npt.NDArray[np.float64]: ...

#
def mquantiles_cimj(
    data: _ArrayLikeFloat_co,
    prob: _ArrayLikeFloat_co = [0.25, 0.5, 0.75],
    alpha: float | np.floating[Any] = 0.05,
    axis: op.CanIndex | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

#
@overload
def median_cihs(
    data: _ArrayLikeFloat_co,
    alpha: float | np.floating[Any] = 0.05,
    axis: None = None,
) -> tuple[np.float64, np.float64]: ...
@overload
def median_cihs(
    data: _ArrayLikeFloat_co,
    alpha: float | np.floating[Any],
    axis: op.CanIndex,
) -> tuple[np.float64 | npt.NDArray[np.float64], np.float64 | npt.NDArray[np.float64]]: ...
@overload
def median_cihs(
    data: _ArrayLikeFloat_co,
    alpha: float | np.floating[Any] = 0.05,
    *,
    axis: op.CanIndex,
) -> tuple[np.float64 | npt.NDArray[np.float64], np.float64 | npt.NDArray[np.float64]]: ...

#
@overload
def compare_medians_ms(group_1: _ArrayLikeFloat_co, group_2: _ArrayLikeFloat_co, axis: None = None) -> np.float64: ...
@overload
def compare_medians_ms(
    group_1: _ArrayLikeFloat_co, group_2: _ArrayLikeFloat_co, axis: op.CanIndex
) -> npt.NDArray[np.float64]: ...

#
@overload
def idealfourths(data: _ArrayLikeFloat_co, axis: None = None) -> list[np.float64]: ...
@overload
def idealfourths(data: _ArrayLikeFloat_co, axis: op.CanIndex) -> np.ma.MaskedArray[onp.AtLeast1D, np.dtype[np.float64]]: ...

#
def rsh(data: _ArrayLikeFloat_co, points: _ArrayLikeFloat_co | None = None) -> np.float64: ...
