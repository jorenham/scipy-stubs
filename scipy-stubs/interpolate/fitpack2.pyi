# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

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

@deprecated("will be removed in SciPy v2.0.0")
class UnivariateSpline:
    def __init__(
        self,
        x: object,
        y: object,
        w: object = ...,
        bbox: object = ...,
        k: object = ...,
        s: object = ...,
        ext: object = ...,
        check_finite: object = ...,
    ) -> None: ...
    @staticmethod
    def validate_input(
        x: object,
        y: object,
        w: object,
        bbox: object,
        k: object,
        s: object,
        ext: object,
        check_finite: object,
    ) -> object: ...
    def set_smoothing_factor(self, s: object) -> None: ...
    def __call__(self, x: object, nu: object = ..., ext: object = ...) -> object: ...
    def get_knots(self) -> object: ...
    def get_coeffs(self) -> object: ...
    def get_residual(self) -> object: ...
    def integral(self, a: object, b: object) -> object: ...
    def derivatives(self, x: object) -> object: ...
    def roots(self) -> object: ...
    def derivative(self, n: object = ...) -> object: ...
    def antiderivative(self, n: object = ...) -> object: ...

@deprecated("will be removed in SciPy v2.0.0")
class InterpolatedUnivariateSpline:
    def __init__(
        self,
        x: object,
        y: object,
        w: object = ...,
        bbox: object = ...,
        k: object = ...,
        ext: object = ...,
        check_finite: object = ...,
    ) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
class LSQUnivariateSpline:
    def __init__(
        self,
        x: object,
        y: object,
        t: object,
        w: object = ...,
        bbox: object = ...,
        k: object = ...,
        ext: object = ...,
        check_finite: object = ...,
    ) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
class BivariateSpline:
    def ev(self, xi: object, yi: object, dx: object = ..., dy: object = ...) -> object: ...
    def integral(self, xa: object, xb: object, ya: object, yb: object) -> object: ...

@deprecated("will be removed in SciPy v2.0.0")
class SmoothBivariateSpline:
    def __init__(
        self,
        x: object,
        y: object,
        z: object,
        w: object = ...,
        bbox: object = ...,
        kx: object = ...,
        ky: object = ...,
        s: object = ...,
        eps: object = ...,
    ) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
class LSQBivariateSpline:
    def __init__(
        self,
        x: object,
        y: object,
        z: object,
        tx: object,
        ty: object,
        w: object = ...,
        bbox: object = ...,
        kx: object = ...,
        ky: object = ...,
        eps: object = ...,
    ) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
class RectBivariateSpline:
    def __init__(
        self,
        x: object,
        y: object,
        z: object,
        bbox: object = ...,
        kx: object = ...,
        ky: object = ...,
        s: object = ...,
    ) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
class SmoothSphereBivariateSpline:
    def __init__(self, theta: object, phi: object, r: object, w: object = ..., s: object = ..., eps: object = ...) -> None: ...
    def __call__(self, theta: object, phi: object, dtheta: object = ..., dphi: object = ..., grid: object = ...) -> object: ...

@deprecated("will be removed in SciPy v2.0.0")
class LSQSphereBivariateSpline:
    def __init__(
        self,
        theta: object,
        phi: object,
        r: object,
        tt: object,
        tp: object,
        w: object = ...,
        eps: object = ...,
    ) -> None: ...
    def __call__(self, theta: object, phi: object, dtheta: object = ..., dphi: object = ..., grid: object = ...) -> object: ...

@deprecated("will be removed in SciPy v2.0.0")
class RectSphereBivariateSpline:
    def __init__(
        self,
        u: object,
        v: object,
        r: object,
        s: object = ...,
        pole_continuity: object = ...,
        pole_values: object = ...,
        pole_exact: object = ...,
        pole_flat: object = ...,
    ) -> None: ...
    def __call__(self, theta: object, phi: object, dtheta: object = ..., dphi: object = ..., grid: object = ...) -> object: ...
