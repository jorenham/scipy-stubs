from typing import Final, Generic, Literal, NoReturn, TypeAlias
from typing_extensions import Self, TypeVar, deprecated

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt, _ArrayLikeNumber_co
from scipy._typing import Untyped
from ._polyint import _Interpolator1D

_CT_co = TypeVar("_CT_co", bound=np.float64 | np.complex128, default=np.float64 | np.complex128, covariant=True)

_Extrapolate: TypeAlias = bool | Literal["periodic"]

###

__all__ = ["BPoly", "NdPPoly", "PPoly", "interp1d", "interp2d", "lagrange"]

err_mesg: str  # undocumented

@deprecated("removed in 1.14.0")
class interp2d:
    def __init__(
        self,
        x: Untyped,
        y: Untyped,
        z: Untyped,
        kind: str = "linear",
        copy: bool = True,
        bounds_error: bool = False,
        fill_value: Untyped | None = None,
    ) -> NoReturn: ...

@deprecated("legacy")
class interp1d(_Interpolator1D):
    bounds_error: Untyped
    copy: Untyped
    axis: Untyped
    y: Untyped
    x: Untyped
    x_bds: Untyped

    def __init__(
        self,
        x: Untyped,
        y: Untyped,
        kind: str = "linear",
        axis: int = -1,
        copy: bool = True,
        bounds_error: Untyped | None = None,
        fill_value: Untyped = ...,
        assume_sorted: bool = False,
    ) -> None: ...
    @property
    def fill_value(self) -> Untyped: ...
    @fill_value.setter
    def fill_value(self, fill_value: Untyped, /) -> None: ...

class _PPolyBase(Generic[_CT_co]):
    c: onp.Array[onp.AtLeast2D, _CT_co]
    x: onp.Array1D[np.float64]
    extrapolate: Final[_Extrapolate]
    axis: Final[int]

    @classmethod
    def construct_fast(
        cls,
        c: _ArrayLikeNumber_co,
        x: _ArrayLikeFloat_co,
        extrapolate: _Extrapolate | None = None,
        axis: int = 0,
    ) -> Self: ...
    def __init__(
        self,
        /,
        c: _ArrayLikeNumber_co,
        x: _ArrayLikeFloat_co,
        extrapolate: _Extrapolate | None = None,
        axis: int = 0,
    ) -> None: ...
    def __call__(
        self,
        /,
        x: _ArrayLikeFloat_co,
        nu: int = 0,
        extrapolate: _Extrapolate | None = None,
    ) -> onp.Array[onp.AtLeast2D, _CT_co]: ...
    def extend(self, /, c: _ArrayLikeNumber_co, x: _ArrayLikeFloat_co) -> None: ...

class PPoly(_PPolyBase[_CT_co], Generic[_CT_co]):
    @classmethod
    def from_spline(cls, tck: Untyped, extrapolate: _Extrapolate | None = None) -> Self: ...
    @classmethod
    def from_bernstein_basis(cls, bp: BPoly[_CT_co], extrapolate: _Extrapolate | None = None) -> Self: ...
    def derivative(self, nu: int = 1) -> Self: ...
    def antiderivative(self, nu: int = 1) -> Self: ...
    def integrate(self, a: onp.ToFloat, b: onp.ToFloat, extrapolate: _Extrapolate | None = None) -> npt.NDArray[_CT_co]: ...
    def solve(
        self,
        y: onp.ToFloat = 0.0,
        discontinuity: bool = True,
        extrapolate: _Extrapolate | None = None,
    ) -> _CT_co | npt.NDArray[_CT_co]: ...
    def roots(self, discontinuity: bool = True, extrapolate: _Extrapolate | None = None) -> _CT_co | npt.NDArray[_CT_co]: ...

class BPoly(_PPolyBase[_CT_co], Generic[_CT_co]):
    @classmethod
    def from_power_basis(cls, pp: PPoly[_CT_co], extrapolate: _Extrapolate | None = None) -> Self: ...
    @classmethod
    def from_derivatives(
        cls,
        xi: _ArrayLikeNumber_co,
        yi: _ArrayLikeNumber_co,
        orders: _ArrayLikeInt | None = None,
        extrapolate: _Extrapolate | None = None,
    ) -> Self: ...
    def derivative(self, nu: int = 1) -> Self: ...
    def antiderivative(self, nu: int = 1) -> Self: ...
    def integrate(self, a: onp.ToFloat, b: onp.ToFloat, extrapolate: _Extrapolate | None = None) -> npt.NDArray[_CT_co]: ...

class NdPPoly(Generic[_CT_co]):
    c: onp.Array[onp.AtLeast2D, _CT_co]
    x: tuple[onp.Array1D[np.float64], ...]

    @classmethod
    def construct_fast(
        cls,
        c: _ArrayLikeNumber_co,
        x: tuple[_ArrayLikeFloat_co, ...],
        extrapolate: bool | None = None,
    ) -> Self: ...
    def __init__(
        self,
        c: _ArrayLikeNumber_co,
        x: tuple[_ArrayLikeFloat_co, ...],
        extrapolate: bool | None = None,
    ) -> None: ...
    def __call__(
        self,
        x: _ArrayLikeFloat_co,
        nu: tuple[int, ...] | None = None,
        extrapolate: bool | None = None,
    ) -> npt.NDArray[_CT_co]: ...
    def derivative(self, nu: tuple[int, ...]) -> Self: ...
    def antiderivative(self, nu: tuple[int, ...]) -> Self: ...
    def integrate_1d(
        self,
        a: onp.ToFloat,
        b: onp.ToFloat,
        axis: op.CanIndex,
        extrapolate: bool | None = None,
    ) -> Self | npt.NDArray[_CT_co]: ...
    def integrate(
        self,
        ranges: tuple[tuple[onp.ToFloat, onp.ToFloat]],
        extrapolate: bool | None = None,
    ) -> npt.NDArray[_CT_co]: ...

def lagrange(x: _ArrayLikeNumber_co, w: _ArrayLikeNumber_co) -> np.poly1d: ...
