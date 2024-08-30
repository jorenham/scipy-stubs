from typing_extensions import override

from scipy._typing import Untyped

dfitpack_int: Untyped

class UnivariateSpline:
    def __init__(
        self,
        x,
        y,
        /,
        *,
        w: Untyped | None = None,
        bbox=...,
        k: int = 3,
        s: Untyped | None = None,
        ext: int = 0,
        check_finite: bool = False,
    ): ...
    @staticmethod
    def validate_input(x, y, w, bbox, k, s, ext, check_finite) -> Untyped: ...
    def set_smoothing_factor(self, s): ...
    def __call__(self, x, nu: int = 0, ext: Untyped | None = None) -> Untyped: ...
    def get_knots(self) -> Untyped: ...
    def get_coeffs(self) -> Untyped: ...
    def get_residual(self) -> Untyped: ...
    def integral(self, a, b) -> Untyped: ...
    def derivatives(self, x) -> Untyped: ...
    def roots(self) -> Untyped: ...
    def derivative(self, n: int = 1) -> Untyped: ...
    def antiderivative(self, n: int = 1) -> Untyped: ...

class InterpolatedUnivariateSpline(UnivariateSpline): ...

class LSQUnivariateSpline(UnivariateSpline):
    def __init__(
        self,
        x,
        y,
        t,
        /,
        *,
        w: Untyped | None = None,
        bbox=...,
        k: int = 3,
        ext: int = 0,
        check_finite: bool = False,
    ) -> None: ...

class _BivariateSplineBase:
    def get_residual(self) -> Untyped: ...
    def get_knots(self) -> Untyped: ...
    def get_coeffs(self) -> Untyped: ...
    def __call__(self, x, y, dx: int = 0, dy: int = 0, /, *, grid: bool = True) -> Untyped: ...
    def partial_derivative(self, dx, dy) -> Untyped: ...

class BivariateSpline(_BivariateSplineBase):
    def ev(self, xi, yi, dx: int = 0, dy: int = 0) -> Untyped: ...
    def integral(self, xa, xb, ya, yb) -> Untyped: ...

class _DerivedBivariateSpline(_BivariateSplineBase):
    @property
    def fp(self) -> Untyped: ...

class SmoothBivariateSpline(BivariateSpline):
    fp: Untyped
    tck: Untyped
    degrees: Untyped
    def __init__(
        self,
        x,
        y,
        z,
        /,
        *,
        w: Untyped | None = None,
        bbox=...,
        kx: int = 3,
        ky: int = 3,
        s: Untyped | None = None,
        eps: float = 1e-16,
    ) -> None: ...

class LSQBivariateSpline(BivariateSpline):
    fp: Untyped
    tck: Untyped
    degrees: Untyped
    def __init__(
        self,
        x,
        y,
        z,
        /,
        tx,
        ty,
        *,
        w: Untyped | None = None,
        bbox=...,
        kx: int = 3,
        ky: int = 3,
        eps: Untyped | None = None,
    ) -> None: ...

class RectBivariateSpline(BivariateSpline):
    fp: Untyped
    tck: Untyped
    degrees: Untyped
    def __init__(self, x, y, z, /, *, bbox=..., kx: int = 3, ky: int = 3, s: int = 0) -> None: ...

class SphereBivariateSpline(_BivariateSplineBase):
    @override
    def __call__(self, theta, phi, dtheta: int = 0, dphi: int = 0, /, *, grid: bool = True) -> Untyped: ...
    def ev(self, theta, phi, dtheta: int = 0, dphi: int = 0) -> Untyped: ...

class SmoothSphereBivariateSpline(SphereBivariateSpline):
    fp: Untyped
    tck: Untyped
    degrees: Untyped
    def __init__(self, theta, phi, r, /, *, w: Untyped | None = None, s: float = 0.0, eps: float = 1e-16) -> None: ...
    @override
    def __call__(self, theta, phi, dtheta: int = 0, dphi: int = 0, /, *, grid: bool = True) -> Untyped: ...

class LSQSphereBivariateSpline(SphereBivariateSpline):
    fp: Untyped
    tck: Untyped
    degrees: Untyped
    def __init__(self, theta, phi, r, /, tt, tp, *, w: Untyped | None = None, eps: float = 1e-16) -> None: ...
    @override
    def __call__(self, theta, phi, dtheta: int = 0, dphi: int = 0, /, *, grid: bool = True) -> Untyped: ...

class RectSphereBivariateSpline(SphereBivariateSpline):
    fp: Untyped
    tck: Untyped
    degrees: Untyped
    v0: Untyped
    def __init__(
        self,
        u,
        v,
        r,
        /,
        s: float = 0.0,
        *,
        pole_continuity: bool = False,
        pole_values: Untyped | None = None,
        pole_exact: bool = False,
        pole_flat: bool = False,
    ) -> None: ...
    @override
    def __call__(self, theta, phi, dtheta: int = 0, dphi: int = 0, /, *, grid: bool = True) -> Untyped: ...
