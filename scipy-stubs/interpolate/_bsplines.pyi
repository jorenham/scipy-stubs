from typing import Any, Generic, Literal, TypeAlias, TypeVar
from typing_extensions import Self

import numpy as np
import optype as op
import optype.numpy as onp
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_array

__all__ = ["BSpline", "make_interp_spline", "make_lsq_spline", "make_smoothing_spline"]

_Extrapolate: TypeAlias = Literal["periodic"] | bool
_BCType: TypeAlias = Literal["not-a-knot", "natural", "clamped", "periodic"]

_SCT_co = TypeVar("_SCT_co", bound=np.floating[Any], default=np.floating[Any], covariant=True)

###

class BSpline(Generic[_SCT_co]):
    t: onp.Array1D[np.float64]
    c: onp.Array[onp.AtLeast1D, _SCT_co]
    k: int
    extrapolate: _Extrapolate
    axis: int

    @property
    def tck(self, /) -> tuple[onp.Array1D[np.float64], onp.Array[onp.AtLeast1D, _SCT_co], int]: ...

    #
    def __init__(
        self,
        /,
        t: onp.ToFloat1D,
        c: onp.ToFloatND,
        k: op.CanIndex,
        extrapolate: _Extrapolate = True,
        axis: op.CanIndex = 0,
    ) -> None: ...
    def __call__(self, /, x: onp.ToFloatND, nu: int = 0, extrapolate: _Extrapolate | None = None) -> onp.ArrayND[_SCT_co]: ...

    #
    def derivative(self, /, nu: int = 1) -> Self: ...
    def antiderivative(self, /, nu: int = 1) -> Self: ...
    def integrate(self, /, a: onp.ToFloat, b: onp.ToFloat, extrapolate: _Extrapolate | None = None) -> onp.ArrayND[_SCT_co]: ...
    def insert_knot(self, /, x: onp.ToFloat, m: op.CanIndex = 1) -> Self: ...

    #
    @classmethod
    def basis_element(cls, t: onp.ToFloat1D, extrapolate: _Extrapolate = True) -> Self: ...
    @classmethod
    def design_matrix(
        cls,
        x: onp.ToFloat1D,
        t: onp.ToFloat1D,
        k: op.CanIndex,
        extrapolate: _Extrapolate = False,
    ) -> csr_array: ...
    @classmethod
    def from_power_basis(cls, pp: CubicSpline, bc_type: _BCType = "not-a-knot") -> Self: ...
    @classmethod
    def construct_fast(
        cls,
        t: onp.Array1D[np.float64],
        c: onp.Array[onp.AtLeast1D, _SCT_co],
        k: int,
        extrapolate: _Extrapolate = True,
        axis: int = 0,
    ) -> Self: ...

#
def make_interp_spline(
    x: onp.ToFloat1D,
    y: onp.ToFloatND,
    k: op.CanIndex = 3,
    t: onp.ToFloat1D | None = None,
    bc_type: tuple[onp.ToFloat, onp.ToFloat] | _BCType | None = None,
    axis: op.CanIndex = 0,
    check_finite: onp.ToBool = True,
) -> BSpline: ...

#
def make_lsq_spline(
    x: onp.ToFloat1D,
    y: onp.ToFloatND,
    t: onp.ToFloat1D,
    k: op.CanIndex = 3,
    w: onp.ToFloat1D | None = None,
    axis: op.CanIndex = 0,
    check_finite: onp.ToBool = True,
) -> BSpline: ...

#
def make_smoothing_spline(
    x: onp.ToFloat1D,
    y: onp.ToFloat1D,
    w: onp.ToFloat1D | None = None,
    lam: onp.ToFloat | None = None,
) -> BSpline: ...

#
def fpcheck(x: onp.ToFloat1D, t: onp.ToFloat1D, k: onp.ToJustInt) -> None: ...  # undocumented
