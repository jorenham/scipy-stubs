from ._bsplines import make_interp_spline as make_interp_spline
from ._cubic import PchipInterpolator as PchipInterpolator
from ._fitpack2 import RectBivariateSpline as RectBivariateSpline
from ._ndbspline import make_ndbspl as make_ndbspl
from ._rgi_cython import evaluate_linear_2d as evaluate_linear_2d, find_indices as find_indices
from scipy._typing import Untyped

class RegularGridInterpolator:
    method: Untyped
    bounds_error: Untyped
    values: Untyped
    fill_value: Untyped
    def __init__(
        self,
        points,
        values,
        method: str = "linear",
        bounds_error: bool = True,
        fill_value=...,
        *,
        solver: Untyped | None = None,
        solver_args: Untyped | None = None,
    ): ...
    def __call__(self, xi, method: Untyped | None = None, *, nu: Untyped | None = None) -> Untyped: ...

def interpn(points, values, xi, method: str = "linear", bounds_error: bool = True, fill_value=...) -> Untyped: ...
