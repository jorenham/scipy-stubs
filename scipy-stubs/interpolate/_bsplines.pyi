from scipy._typing import Untyped

__all__ = ["BSpline", "make_interp_spline", "make_lsq_spline", "make_smoothing_spline"]

class BSpline:
    k: Untyped
    c: Untyped
    t: Untyped
    extrapolate: Untyped
    axis: Untyped
    @property
    def tck(self) -> Untyped: ...
    def __init__(self, t: Untyped, c: Untyped, k: Untyped, extrapolate: bool = True, axis: int = 0) -> None: ...
    def __call__(self, x: Untyped, nu: int = 0, extrapolate: Untyped | None = None) -> Untyped: ...
    def derivative(self, nu: int = 1) -> Untyped: ...
    def antiderivative(self, nu: int = 1) -> Untyped: ...
    def integrate(self, a: Untyped, b: Untyped, extrapolate: Untyped | None = None) -> Untyped: ...
    def insert_knot(self, x: Untyped, m: int = 1) -> Untyped: ...
    @classmethod
    def basis_element(cls, t: Untyped, extrapolate: bool = True) -> Untyped: ...
    @classmethod
    def design_matrix(cls, x: Untyped, t: Untyped, k: Untyped, extrapolate: bool = False) -> Untyped: ...
    @classmethod
    def from_power_basis(cls, pp: Untyped, bc_type: str = "not-a-knot") -> Untyped: ...
    @classmethod
    def construct_fast(cls, t: Untyped, c: Untyped, k: Untyped, extrapolate: bool = True, axis: int = 0) -> Untyped: ...

def make_interp_spline(
    x: Untyped,
    y: Untyped,
    k: int = 3,
    t: Untyped | None = None,
    bc_type: Untyped | None = None,
    axis: int = 0,
    check_finite: bool = True,
) -> Untyped: ...
def make_lsq_spline(
    x: Untyped,
    y: Untyped,
    t: Untyped,
    k: int = 3,
    w: Untyped | None = None,
    axis: int = 0,
    check_finite: bool = True,
) -> Untyped: ...
def make_smoothing_spline(
    x: Untyped,
    y: Untyped,
    w: Untyped | None = None,
    lam: Untyped | None = None,
) -> Untyped: ...
def fpcheck(x: Untyped, t: Untyped, k: Untyped) -> None: ...  # undocumented
