from typing import Any, Final, Generic, Literal, TypeAlias
from typing_extensions import Never, Self, TypeVar, deprecated

import numpy as np
import optype as op
import optype.numpy as onp
from ._polyint import _Interpolator1D

__all__ = ["BPoly", "NdPPoly", "PPoly", "interp1d", "interp2d", "lagrange"]

_CT_co = TypeVar("_CT_co", bound=np.float64 | np.complex128, default=np.float64 | np.complex128, covariant=True)

_ToAxis: TypeAlias = int | np.integer[Any]
_Extrapolate: TypeAlias = Literal["periodic"] | bool

_Interp1dKind: TypeAlias = Literal["linear", "nearest", "nearest-up", "zero", "slinear", "quadratic", "cubic", "previous", "next"]
_Interp1dFillValue: TypeAlias = onp.ToFloat | onp.ToFloatND | tuple[onp.ToFloat | onp.ToFloatND, onp.ToFloat | onp.ToFloatND]

###

err_mesg: Final = """\
`interp2d` has been removed in SciPy 1.14.0.

For legacy code, nearly bug-for-bug compatible replacements are
`RectBivariateSpline` on regular grids, and `bisplrep`/`bisplev` for
scattered 2D data.

In new code, for regular grids use `RegularGridInterpolator` instead.
For scattered data, prefer `LinearNDInterpolator` or
`CloughTocher2DInterpolator`.

For more details see
https://scipy.github.io/devdocs/tutorial/interpolate/interp_transition_guide.html
"""  # noqa: PYI053  # undocumented

@deprecated(err_mesg)
class interp2d:
    def __init__(
        self,
        /,
        x: Never,
        y: Never,
        z: Never,
        kind: object = ...,
        copy: object = ...,
        bounds_error: object = ...,
        fill_value: object = ...,
    ) -> Never: ...

@deprecated("legacy")
class interp1d(_Interpolator1D):
    copy: bool
    bounds_error: bool
    axis: int
    x: onp.Array1D[np.floating[Any] | np.integer[Any] | np.bool_]
    y: onp.ArrayND[np.inexact[Any]]
    x_bds: onp.Array1D[np.floating[Any]]  # only set if `kind in {"nearest", "nearest-up"}`

    @property
    def fill_value(self, /) -> _Interp1dFillValue: ...
    @fill_value.setter
    def fill_value(self, fill_value: _Interp1dFillValue | Literal["extrapolate"], /) -> None: ...

    #
    def __init__(
        self,
        /,
        x: onp.ToFloat1D,
        y: onp.ToFloatND,
        kind: _Interp1dKind | int = "linear",
        axis: _ToAxis = -1,
        copy: bool = True,
        bounds_error: bool | None = None,
        fill_value: _Interp1dFillValue | Literal["extrapolate"] = ...,  # np.nan
        assume_sorted: bool = False,
    ) -> None: ...

class _PPolyBase(Generic[_CT_co]):
    c: onp.Array[onp.AtLeast2D, _CT_co]
    x: onp.Array1D[np.float64]
    extrapolate: Final[_Extrapolate]
    axis: Final[int]

    @classmethod
    def construct_fast(
        cls,
        c: onp.ToComplexND,  # at least 2d
        x: onp.ToFloat1D,
        extrapolate: _Extrapolate | None = None,
        axis: _ToAxis = 0,
    ) -> Self: ...

    #
    def __init__(
        self,
        /,
        c: onp.ToComplexND,
        x: onp.ToFloat1D,
        extrapolate: _Extrapolate | None = None,
        axis: _ToAxis = 0,
    ) -> None: ...
    def __call__(
        self,
        /,
        x: onp.ToArrayND,
        nu: _ToAxis = 0,
        extrapolate: _Extrapolate | None = None,
    ) -> onp.Array[onp.AtLeast2D, _CT_co]: ...
    def extend(self, /, c: onp.ToComplexND, x: onp.ToFloat1D) -> None: ...

class PPoly(_PPolyBase[_CT_co], Generic[_CT_co]):
    @classmethod
    def from_spline(
        cls,
        tck: tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], int],
        extrapolate: _Extrapolate | None = None,
    ) -> Self: ...
    @classmethod
    def from_bernstein_basis(cls, bp: BPoly[_CT_co], extrapolate: _Extrapolate | None = None) -> Self: ...

    #
    def derivative(self, /, nu: _ToAxis = 1) -> Self: ...
    def antiderivative(self, /, nu: _ToAxis = 1) -> Self: ...
    def integrate(self, /, a: onp.ToFloat, b: onp.ToFloat, extrapolate: _Extrapolate | None = None) -> onp.ArrayND[_CT_co]: ...
    def solve(
        self,
        /,
        y: onp.ToFloat = 0.0,
        discontinuity: onp.ToBool = True,
        extrapolate: _Extrapolate | None = None,
    ) -> _CT_co | onp.ArrayND[_CT_co]: ...
    def roots(
        self,
        /,
        discontinuity: onp.ToBool = True,
        extrapolate: _Extrapolate | None = None,
    ) -> _CT_co | onp.ArrayND[_CT_co]: ...

class BPoly(_PPolyBase[_CT_co], Generic[_CT_co]):
    @classmethod
    def from_power_basis(cls, pp: PPoly[_CT_co], extrapolate: _Extrapolate | None = None) -> Self: ...
    @classmethod
    def from_derivatives(
        cls,
        xi: onp.ToComplex1D,
        yi: onp.ToComplexND,
        orders: onp.ToInt | onp.ToInt1D | None = None,
        extrapolate: _Extrapolate | None = None,
    ) -> Self: ...

    #
    def derivative(self, /, nu: _ToAxis = 1) -> Self: ...
    def antiderivative(self, /, nu: _ToAxis = 1) -> Self: ...
    def integrate(self, /, a: onp.ToFloat, b: onp.ToFloat, extrapolate: _Extrapolate | None = None) -> onp.ArrayND[_CT_co]: ...

class NdPPoly(Generic[_CT_co]):
    c: onp.Array[onp.AtLeast2D, _CT_co]
    x: tuple[onp.Array1D[np.float64], ...]

    @classmethod
    def construct_fast(
        cls,
        c: onp.ToComplexND,  # at least 2d
        x: tuple[onp.ToFloat1D, ...],
        extrapolate: _Extrapolate | None = None,
    ) -> Self: ...

    #
    def __init__(
        self,
        /,
        c: onp.ToComplexND,
        x: tuple[onp.ToFloat1D, ...],
        extrapolate: onp.ToBool | None = None,
    ) -> None: ...
    def __call__(
        self,
        /,
        x: onp.ToFloatND,
        nu: tuple[_ToAxis, ...] | None = None,
        extrapolate: onp.ToBool | None = None,
    ) -> onp.ArrayND[_CT_co]: ...
    def derivative(self, /, nu: tuple[int, ...]) -> Self: ...
    def antiderivative(self, /, nu: tuple[int, ...]) -> Self: ...
    def integrate_1d(
        self,
        /,
        a: onp.ToFloat,
        b: onp.ToFloat,
        axis: op.CanIndex,
        extrapolate: onp.ToBool | None = None,
    ) -> Self | onp.ArrayND[_CT_co]: ...
    def integrate(
        self,
        /,
        ranges: tuple[tuple[onp.ToFloat, onp.ToFloat]],
        extrapolate: onp.ToBool | None = None,
    ) -> onp.ArrayND[_CT_co]: ...

def lagrange(x: onp.ToComplex1D, w: onp.ToComplex1D) -> np.poly1d: ...
