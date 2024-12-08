from collections.abc import Callable
from typing import Concatenate, Generic, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp

__all__ = ["RegularGridInterpolator", "interpn"]

_MethodReal: TypeAlias = Literal["linear", "nearest", "slinear", "cubic", "quintic"]
_Method: TypeAlias = Literal[_MethodReal, "pchip"]

_SCT_co = TypeVar("_SCT_co", bound=np.float64 | np.complex128, default=np.float64 | np.complex128, covariant=True)

###

class RegularGridInterpolator(Generic[_SCT_co]):
    grid: tuple[onp.ArrayND[_SCT_co], ...]
    values: onp.ArrayND[_SCT_co]
    method: _Method
    fill_value: float | None
    bounds_error: bool

    @overload
    def __init__(
        self: RegularGridInterpolator[np.float64],
        /,
        points: tuple[onp.ToFloat1D, ...],
        values: onp.ToFloatND,
        method: _Method = "linear",
        bounds_error: onp.ToBool = True,
        fill_value: onp.ToFloat | None = ...,  # np.nan
        *,
        solver: Callable[Concatenate[onp.Array2D[np.float64], ...], tuple[onp.Array1D[np.float64]]] | None = None,
        solver_args: tuple[object, ...] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        points: tuple[onp.ToFloat1D, ...],
        values: onp.ToComplexND,
        method: _MethodReal = "linear",
        bounds_error: onp.ToBool = True,
        fill_value: onp.ToComplex | None = ...,  # np.nan
        *,
        solver: Callable[Concatenate[onp.Array2D[np.float64], ...], tuple[onp.Array1D[np.float64]]] | None = None,
        solver_args: tuple[object, ...] | None = None,
    ) -> None: ...

    #
    def __call__(
        self,
        /,
        xi: onp.ToFloatND,
        method: _Method | None = None,
        *,
        nu: onp.ToJustInt1D | None = None,
    ) -> onp.ArrayND[_SCT_co]: ...

@overload
def interpn(
    points: tuple[onp.ToFloat1D, ...],
    values: onp.ToFloatND,
    xi: onp.ToFloatND,
    method: _Method = "linear",
    bounds_error: onp.ToBool = True,
    fill_value: onp.ToFloat = ...,  # np.nan
) -> onp.ArrayND[np.float64]: ...
@overload
def interpn(
    points: tuple[onp.ToFloat1D, ...],
    values: onp.ToComplex1D,
    xi: onp.ToFloatND,
    method: _Method = "linear",
    bounds_error: onp.ToBool = True,
    fill_value: onp.ToComplex = ...,  # np.nan
) -> onp.ArrayND[np.float64 | np.complex128]: ...
