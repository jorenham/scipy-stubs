from typing import Any, Literal, overload

import numpy as np
import optype as op
import optype.numpy as onp

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
    data: onp.ToFloatND,
    prob: onp.ToFloat | onp.ToFloatND = [0.25, 0.5, 0.75],
    axis: op.CanIndex | None = None,
    var: Literal[0, False] = False,
) -> np.ma.MaskedArray[onp.AtLeast1D, np.dtype[np.float64]]: ...
@overload
def hdquantiles(
    data: onp.ToFloatND,
    prob: onp.ToFloat | onp.ToFloatND,
    axis: op.CanIndex | None,
    var: Literal[1, True],
) -> np.ma.MaskedArray[onp.AtLeast2D, np.dtype[np.float64]]: ...
@overload
def hdquantiles(
    data: onp.ToFloatND,
    prob: onp.ToFloat | onp.ToFloatND = [0.25, 0.5, 0.75],
    axis: op.CanIndex | None = None,
    *,
    var: Literal[1, True],
) -> np.ma.MaskedArray[onp.AtLeast2D, np.dtype[np.float64]]: ...

#
@overload
def hdmedian(
    data: onp.ToFloatND,
    axis: op.CanIndex | None = -1,
    var: Literal[0, False] = False,
) -> np.ma.MaskedArray[onp.AtLeast0D, np.dtype[np.float64]]: ...
@overload
def hdmedian(
    data: onp.ToFloatND,
    axis: op.CanIndex | None,
    var: Literal[1, True],
) -> np.ma.MaskedArray[onp.AtLeast1D, np.dtype[np.float64]]: ...
@overload
def hdmedian(
    data: onp.ToFloatND,
    axis: op.CanIndex | None = -1,
    *,
    var: Literal[1, True],
) -> np.ma.MaskedArray[onp.AtLeast1D, np.dtype[np.float64]]: ...

#
def hdquantiles_sd(
    data: onp.ToFloatND,
    prob: onp.ToFloat | onp.ToFloatND = [0.25, 0.5, 0.75],
    axis: op.CanIndex | None = None,
) -> np.ma.MaskedArray[onp.AtLeast1D, np.dtype[np.float64]]: ...

#
def trimmed_mean_ci(
    data: onp.ToFloatND,
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = (0.2, 0.2),
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    alpha: float | np.floating[Any] = 0.05,
    axis: op.CanIndex | None = None,
) -> onp.ArrayND[np.float64]: ...

#
def mjci(
    data: onp.ToFloatND,
    prob: onp.ToFloat | onp.ToFloatND = [0.25, 0.5, 0.75],
    axis: op.CanIndex | None = None,
) -> onp.ArrayND[np.float64]: ...

#
def mquantiles_cimj(
    data: onp.ToFloatND,
    prob: onp.ToFloat | onp.ToFloatND = [0.25, 0.5, 0.75],
    alpha: float | np.floating[Any] = 0.05,
    axis: op.CanIndex | None = None,
) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]: ...

#
@overload
def median_cihs(
    data: onp.ToFloatND,
    alpha: float | np.floating[Any] = 0.05,
    axis: None = None,
) -> tuple[np.float64, np.float64]: ...
@overload
def median_cihs(
    data: onp.ToFloatND,
    alpha: float | np.floating[Any],
    axis: op.CanIndex,
) -> tuple[np.float64 | onp.ArrayND[np.float64], np.float64 | onp.ArrayND[np.float64]]: ...
@overload
def median_cihs(
    data: onp.ToFloatND,
    alpha: float | np.floating[Any] = 0.05,
    *,
    axis: op.CanIndex,
) -> tuple[np.float64 | onp.ArrayND[np.float64], np.float64 | onp.ArrayND[np.float64]]: ...

#
@overload
def compare_medians_ms(group_1: onp.ToFloatND, group_2: onp.ToFloatND, axis: None = None) -> np.float64: ...
@overload
def compare_medians_ms(group_1: onp.ToFloatND, group_2: onp.ToFloatND, axis: op.CanIndex) -> onp.ArrayND[np.float64]: ...

#
@overload
def idealfourths(data: onp.ToFloatND, axis: None = None) -> list[np.float64]: ...
@overload
def idealfourths(data: onp.ToFloatND, axis: op.CanIndex) -> np.ma.MaskedArray[onp.AtLeast1D, np.dtype[np.float64]]: ...

#
def rsh(data: onp.ToFloatND, points: onp.ToFloatND | None = None) -> np.float64: ...
