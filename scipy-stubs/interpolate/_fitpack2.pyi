from typing_extensions import override

import numpy.typing as npt
from scipy._typing import Untyped

__all__ = [
    "BivariateSpline",
    "InterpolatedUnivariateSpline",
    "LSQBivariateSpline",
    "LSQSphereBivariateSpline",
    "LSQUnivariateSpline",
    "RectBivariateSpline",
    "RectSphereBivariateSpline",
    "SmoothBivariateSpline",
    "SmoothSphereBivariateSpline",
    "UnivariateSpline",
]

dfitpack_int: Untyped  # undocumented

class UnivariateSpline:
    # at runtime the `__init__` might change the `__class__` attribute...
    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        w: npt.ArrayLike | None = None,
        bbox: npt.ArrayLike = ...,
        k: int = 3,
        s: float | None = None,
        ext: int | str = 0,
        check_finite: bool = False,
    ) -> None: ...
    def __call__(self, x: Untyped, nu: int = 0, ext: Untyped | None = None) -> Untyped: ...
    def set_smoothing_factor(self, s: Untyped) -> None: ...
    def get_knots(self) -> Untyped: ...
    def get_coeffs(self) -> Untyped: ...
    def get_residual(self) -> Untyped: ...
    def integral(self, a: Untyped, b: Untyped) -> Untyped: ...
    def derivatives(self, x: Untyped) -> Untyped: ...
    def roots(self) -> Untyped: ...
    def derivative(self, n: int = 1) -> Untyped: ...
    def antiderivative(self, n: int = 1) -> Untyped: ...
    @staticmethod
    def validate_input(
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        w: npt.ArrayLike,
        bbox: npt.ArrayLike,
        k: int,
        s: float | None,
        ext: int,
        check_finite: bool,
    ) -> Untyped: ...

class InterpolatedUnivariateSpline(UnivariateSpline):
    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        w: npt.ArrayLike | None = None,
        bbox: npt.ArrayLike = ...,
        k: int = 3,
        ext: int | str = 0,
        check_finite: bool = False,
    ) -> None: ...

class LSQUnivariateSpline(UnivariateSpline):
    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        t: npt.ArrayLike,
        w: npt.ArrayLike | None = None,
        bbox: npt.ArrayLike = ...,
        k: int = 3,
        ext: int = 0,
        check_finite: bool = False,
    ) -> None: ...

class _BivariateSplineBase:  # undocumented
    def __call__(self, x: Untyped, y: Untyped, dx: int = 0, dy: int = 0, grid: bool = True) -> Untyped: ...
    def get_residual(self) -> Untyped: ...
    def get_knots(self) -> Untyped: ...
    def get_coeffs(self) -> Untyped: ...
    def partial_derivative(self, dx: Untyped, dy: Untyped) -> Untyped: ...

class BivariateSpline(_BivariateSplineBase):
    def ev(self, xi: Untyped, yi: Untyped, dx: int = 0, dy: int = 0) -> Untyped: ...
    def integral(self, xa: Untyped, xb: Untyped, ya: Untyped, yb: Untyped) -> Untyped: ...

class _DerivedBivariateSpline(_BivariateSplineBase):  # undocumented
    @property
    def fp(self) -> Untyped: ...

class SmoothBivariateSpline(BivariateSpline):
    fp: Untyped
    tck: Untyped
    degrees: Untyped

    def __init__(
        self,
        x: Untyped,
        y: Untyped,
        z: Untyped,
        w: Untyped | None = None,
        bbox: Untyped = ...,
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
        x: Untyped,
        y: Untyped,
        z: Untyped,
        tx: Untyped,
        ty: Untyped,
        w: Untyped | None = None,
        bbox: Untyped = ...,
        kx: int = 3,
        ky: int = 3,
        eps: Untyped | None = None,
    ) -> None: ...

class RectBivariateSpline(BivariateSpline):
    fp: Untyped
    tck: Untyped
    degrees: Untyped

    def __init__(
        self,
        x: Untyped,
        y: Untyped,
        z: Untyped,
        bbox: Untyped = ...,
        kx: int = 3,
        ky: int = 3,
        s: int = 0,
    ) -> None: ...

class SphereBivariateSpline(_BivariateSplineBase):
    @override
    def __call__(  # type: ignore[override]
        self,
        theta: Untyped,
        phi: Untyped,
        dtheta: int = 0,
        dphi: int = 0,
        grid: bool = True,
    ) -> Untyped: ...
    def ev(self, theta: Untyped, phi: Untyped, dtheta: int = 0, dphi: int = 0) -> Untyped: ...

class SmoothSphereBivariateSpline(SphereBivariateSpline):
    fp: Untyped
    tck: Untyped
    degrees: Untyped

    def __init__(
        self,
        theta: Untyped,
        phi: Untyped,
        r: Untyped,
        w: Untyped | None = None,
        s: float = 0.0,
        eps: float = 1e-16,
    ) -> None: ...
    @override
    def __call__(  # type: ignore[override]
        self,
        theta: Untyped,
        phi: Untyped,
        dtheta: int = 0,
        dphi: int = 0,
        grid: bool = True,
    ) -> Untyped: ...

class LSQSphereBivariateSpline(SphereBivariateSpline):
    fp: Untyped
    tck: Untyped
    degrees: Untyped
    def __init__(
        self,
        theta: Untyped,
        phi: Untyped,
        r: Untyped,
        tt: Untyped,
        tp: Untyped,
        w: Untyped | None = None,
        eps: float = 1e-16,
    ) -> None: ...
    @override
    def __call__(  # type: ignore[override]
        self,
        theta: Untyped,
        phi: Untyped,
        dtheta: int = 0,
        dphi: int = 0,
        grid: bool = True,
    ) -> Untyped: ...

class RectSphereBivariateSpline(SphereBivariateSpline):
    fp: Untyped
    tck: Untyped
    degrees: Untyped
    v0: Untyped
    def __init__(
        self,
        u: Untyped,
        v: Untyped,
        r: Untyped,
        s: float = 0.0,
        pole_continuity: bool = False,
        pole_values: Untyped | None = None,
        pole_exact: bool = False,
        pole_flat: bool = False,
    ) -> None: ...
    @override
    def __call__(  # type: ignore[override]
        self,
        theta: Untyped,
        phi: Untyped,
        dtheta: int = 0,
        dphi: int = 0,
        grid: bool = True,
    ) -> Untyped: ...
