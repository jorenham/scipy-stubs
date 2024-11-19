from collections.abc import Iterable
from typing import Generic, Literal, NoReturn, TypeAlias, overload
from typing_extensions import TypeVar, override

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeNumber_co
from ._interpolate import PPoly

_T = TypeVar("_T")
_CT = TypeVar("_CT", bound=np.float64 | np.complex128)
_CT_co = TypeVar("_CT_co", bound=np.float64 | np.complex128, default=np.float64 | np.complex128, covariant=True)

_Tuple2: TypeAlias = tuple[_T, _T]

_Extrapolate: TypeAlias = Literal["periodic"] | bool
_CubicBCName: TypeAlias = Literal["not-a-knot", "clamped", "natural"]
_CubicBCOrder: TypeAlias = Literal[1, 2]
_CubicBCType: TypeAlias = Literal[_CubicBCName, "periodic"] | _Tuple2[_CubicBCName | tuple[_CubicBCOrder, _ArrayLikeNumber_co]]

###

__all__ = ["Akima1DInterpolator", "CubicHermiteSpline", "CubicSpline", "PchipInterpolator", "pchip_interpolate"]

class CubicHermiteSpline(PPoly[_CT_co]):
    @overload
    def __init__(
        self: CubicHermiteSpline[np.float64],
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeFloat_co,
        dydx: _ArrayLikeFloat_co,
        axis: int = 0,
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: CubicHermiteSpline[np.float64 | np.complex128],
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeNumber_co,
        dydx: _ArrayLikeNumber_co,
        axis: int = 0,
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...

class PchipInterpolator(CubicHermiteSpline[np.float64]):
    def __init__(self, x: _ArrayLikeFloat_co, y: _ArrayLikeFloat_co, axis: int = 0, extrapolate: bool | None = None) -> None: ...

class Akima1DInterpolator(CubicHermiteSpline[np.float64]):
    def __init__(
        self,
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeFloat_co,
        axis: int = 0,
        *,
        method: Literal["akima", "makima"] = "akima",
        extrapolate: bool | None = None,
    ) -> None: ...
    @override
    def extend(self, c: object, x: object, right: object = True) -> NoReturn: ...  # not implemented
    @classmethod
    @override
    def from_spline(cls, tck: object, extrapolate: object = ...) -> NoReturn: ...  # not implemented
    @classmethod
    @override
    def from_bernstein_basis(cls, bp: object, extrapolate: object = ...) -> NoReturn: ...  # not implemented

class CubicSpline(CubicHermiteSpline[_CT_co], Generic[_CT_co]):
    @overload
    def __init__(
        self: CubicSpline[np.float64],
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeFloat_co,
        axis: onp.ToInt = 0,
        bc_type: _CubicBCType = "not-a-knot",
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: CubicSpline[np.float64 | np.complex128],
        x: _ArrayLikeFloat_co,
        y: _ArrayLikeNumber_co,
        axis: onp.ToInt = 0,
        bc_type: _CubicBCType = "not-a-knot",
        extrapolate: _Extrapolate | None = None,
    ) -> None: ...

def pchip_interpolate(
    xi: _ArrayLikeFloat_co,
    yi: _ArrayLikeFloat_co,
    x: _ArrayLikeFloat_co,
    der: int | Iterable[int] = 0,
    axis: int = 0,
) -> np.float64 | npt.NDArray[np.float64]: ...

# undocumented
def prepare_input(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeNumber_co,
    axis: int,
    dydx: _ArrayLikeNumber_co | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[_CT], int, npt.NDArray[_CT]]: ...
