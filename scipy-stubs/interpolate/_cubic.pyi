from typing import Literal
from typing_extensions import override

from scipy._typing import Untyped
from ._interpolate import PPoly

__all__ = ["Akima1DInterpolator", "CubicHermiteSpline", "CubicSpline", "PchipInterpolator", "pchip_interpolate"]

class CubicHermiteSpline(PPoly):
    def __init__(self, x: Untyped, y: Untyped, dydx: Untyped, axis: int = 0, extrapolate: Untyped | None = None) -> None: ...

class PchipInterpolator(CubicHermiteSpline):
    def __init__(self, x: Untyped, y: Untyped, axis: int = 0, extrapolate: Untyped | None = None) -> None: ...

class Akima1DInterpolator(CubicHermiteSpline):
    def __init__(
        self,
        x: Untyped,
        y: Untyped,
        axis: int = 0,
        *,
        method: Literal["akima", "makima"] = "akima",
        extrapolate: bool | None = None,
    ) -> None: ...
    @override
    def extend(self, c: Untyped, x: Untyped, right: bool = True) -> None: ...

class CubicSpline(CubicHermiteSpline):
    def __init__(
        self,
        x: Untyped,
        y: Untyped,
        axis: int = 0,
        bc_type: str = "not-a-knot",
        extrapolate: Untyped | None = None,
    ) -> None: ...

def prepare_input(x: Untyped, y: Untyped, axis: Untyped, dydx: Untyped | None = None) -> Untyped: ...  # undocumented
def pchip_interpolate(xi: Untyped, yi: Untyped, x: Untyped, der: int = 0, axis: int = 0) -> Untyped: ...
