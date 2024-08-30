from typing import Literal

from scipy._typing import Untyped
from scipy.linalg import solve as solve, solve_banded as solve_banded
from . import PPoly as PPoly

def prepare_input(x, y, axis, dydx: Untyped | None = None) -> Untyped: ...

class CubicHermiteSpline(PPoly):
    axis: Untyped
    def __init__(self, x, y, dydx, axis: int = 0, extrapolate: Untyped | None = None): ...

class PchipInterpolator(CubicHermiteSpline):
    axis: Untyped
    def __init__(self, x, y, axis: int = 0, extrapolate: Untyped | None = None): ...

def pchip_interpolate(xi, yi, x, der: int = 0, axis: int = 0) -> Untyped: ...

class Akima1DInterpolator(CubicHermiteSpline):
    axis: Untyped
    def __init__(self, x, y, axis: int = 0, *, method: Literal["akima", "makima"] = "akima", extrapolate: bool | None = None): ...
    def extend(self, c, x, right: bool = True): ...
    @classmethod
    def from_spline(cls, tck, extrapolate: Untyped | None = None): ...
    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate: Untyped | None = None): ...

class CubicSpline(CubicHermiteSpline):
    axis: Untyped
    def __init__(self, x, y, axis: int = 0, bc_type: str = "not-a-knot", extrapolate: Untyped | None = None): ...
