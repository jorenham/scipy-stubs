from scipy._typing import Untyped

__all__ = [
    "BarycentricInterpolator",
    "KroghInterpolator",
    "approximate_taylor_polynomial",
    "barycentric_interpolate",
    "krogh_interpolate",
]

class _Interpolator1D:  # undocumented
    dtype: Untyped
    def __init__(self, xi: Untyped | None = None, yi: Untyped | None = None, axis: Untyped | None = None) -> None: ...
    def __call__(self, x: Untyped) -> Untyped: ...

class _Interpolator1DWithDerivatives(_Interpolator1D):  # undocumented
    def derivatives(self, x: Untyped, der: Untyped | None = None) -> Untyped: ...
    def derivative(self, x: Untyped, der: int = 1) -> Untyped: ...

class KroghInterpolator(_Interpolator1DWithDerivatives):
    xi: Untyped
    yi: Untyped
    c: Untyped
    def __init__(self, xi: Untyped, yi: Untyped, axis: int = 0) -> None: ...

class BarycentricInterpolator(_Interpolator1DWithDerivatives):
    xi: Untyped
    n: Untyped
    wi: Untyped
    yi: Untyped

    def __init__(
        self,
        xi: Untyped,
        yi: Untyped | None = None,
        axis: int = 0,
        *,
        wi: Untyped | None = None,
        random_state: Untyped | None = None,
    ) -> None: ...
    def set_yi(self, yi: Untyped, axis: Untyped | None = None) -> None: ...
    def add_xi(self, xi: Untyped, yi: Untyped | None = None) -> None: ...

def krogh_interpolate(xi: Untyped, yi: Untyped, x: Untyped, der: int = 0, axis: int = 0) -> Untyped: ...
def approximate_taylor_polynomial(
    f: Untyped,
    x: Untyped,
    degree: Untyped,
    scale: Untyped,
    order: Untyped | None = None,
) -> Untyped: ...
def barycentric_interpolate(xi: Untyped, yi: Untyped, x: Untyped, axis: int = 0, *, der: int = 0) -> Untyped: ...
