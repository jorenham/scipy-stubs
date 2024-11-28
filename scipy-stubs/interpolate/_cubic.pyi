from typing import Any, Generic, Literal, TypeAlias, overload
from typing_extensions import Never, TypeVar, override

import numpy as np
import optype.numpy as onp
from ._interpolate import PPoly

_T = TypeVar("_T")
_Tuple2: TypeAlias = tuple[_T, _T]

_ToAxis: TypeAlias = int | np.integer[Any]
_AxisT = TypeVar("_AxisT", bound=_ToAxis)

_CT_co = TypeVar("_CT_co", bound=np.float64 | np.complex128, default=np.float64 | np.complex128, covariant=True)

_Akima1DMethod: TypeAlias = Literal["akima", "makima"]
_Extrapolate: TypeAlias = Literal["periodic"] | bool
_CubicBCName: TypeAlias = Literal["not-a-knot", "clamped", "natural"]
_CubicBCOrder: TypeAlias = Literal[1, 2]
_CubicBCType: TypeAlias = Literal[_CubicBCName, "periodic"] | _Tuple2[_CubicBCName | tuple[_CubicBCOrder, onp.ToComplexND]]

###

__all__ = ["Akima1DInterpolator", "CubicHermiteSpline", "CubicSpline", "PchipInterpolator", "pchip_interpolate"]

class CubicHermiteSpline(PPoly[_CT_co]):
    @overload
    def __init__(
        self: CubicHermiteSpline[np.float64],
        /,
        x: onp.ToFloat1D,
        y: onp.ToFloatND,
        dydx: onp.ToFloatND,
        axis: _ToAxis = 0,
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: CubicHermiteSpline[np.float64 | np.complex128],
        /,
        x: onp.ToFloat1D,
        y: onp.ToComplexND,
        dydx: onp.ToComplexND,
        axis: _ToAxis = 0,
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...

class PchipInterpolator(CubicHermiteSpline[np.float64]):
    def __init__(self, /, x: onp.ToFloat1D, y: onp.ToFloatND, axis: _ToAxis = 0, extrapolate: bool | None = None) -> None: ...

class Akima1DInterpolator(CubicHermiteSpline[np.float64]):
    def __init__(
        self,
        /,
        x: onp.ToFloat1D,
        y: onp.ToFloatND,
        axis: _ToAxis = 0,
        *,
        method: _Akima1DMethod = "akima",
        extrapolate: onp.ToBool | None = None,
    ) -> None: ...

    # the following (class)methods will raise `NotImplementedError` when called
    @override
    def extend(self, /, c: object, x: object, right: object = True) -> Never: ...
    @classmethod
    @override
    def from_spline(cls, tck: object, extrapolate: object = ...) -> Never: ...
    @classmethod
    @override
    def from_bernstein_basis(cls, bp: object, extrapolate: object = ...) -> Never: ...

class CubicSpline(CubicHermiteSpline[_CT_co], Generic[_CT_co]):
    @overload
    def __init__(
        self: CubicSpline[np.float64],
        /,
        x: onp.ToFloat1D,
        y: onp.ToFloatND,
        axis: _ToAxis = 0,
        bc_type: _CubicBCType = "not-a-knot",
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: CubicSpline[np.float64 | np.complex128],
        /,
        x: onp.ToFloat1D,
        y: onp.ToComplexND,
        axis: _ToAxis = 0,
        bc_type: _CubicBCType = "not-a-knot",
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...

def pchip_interpolate(
    xi: onp.ToFloat1D,
    yi: onp.ToFloat1D,
    x: onp.ToFloat | onp.ToFloat1D,
    der: onp.ToInt | onp.ToInt1D = 0,
    axis: _ToAxis = 0,
) -> np.float64 | onp.ArrayND[np.float64]: ...

# undocumented
@overload
def prepare_input(
    x: onp.ToFloat1D,
    y: onp.ToFloatND,
    axis: _AxisT,
    dydx: onp.ToFloatND | None = None,
) -> tuple[
    onp.Array1D[np.float64],  # x
    onp.Array1D[np.float64],  # dx
    onp.ArrayND[np.float64],  # y
    _AxisT,  # axis
    onp.ArrayND[np.float64],  # dydx
]: ...
@overload
def prepare_input(
    x: onp.ToFloat1D,
    y: onp.ToComplexND,
    axis: _AxisT,
    dydx: onp.ToComplexND | None = None,
) -> tuple[
    onp.Array1D[np.float64],  # x
    onp.Array1D[np.float64],  # dx
    onp.ArrayND[np.float64 | np.complex128],  # y
    _AxisT,  # axis
    onp.ArrayND[np.float64 | np.complex128],  # dydx
]: ...
