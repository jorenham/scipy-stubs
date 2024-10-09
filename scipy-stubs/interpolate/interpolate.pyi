# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

from ._bsplines import BSpline as _BSpline
from ._fitpack2 import RectBivariateSpline as _RectBivariateSpline
from ._interpolate import BPoly as _BPoly, NdPPoly as _NdPPoly, PPoly as _PPoly, interp1d as _interp1d, interp2d as _interp2d
from ._rgi import RegularGridInterpolator as _RegularGridInterpolator

__all__ = [
    "BPoly",
    "BSpline",
    "NdPPoly",
    "PPoly",
    "RectBivariateSpline",
    "RegularGridInterpolator",
    "interp1d",
    "interp2d",
    "interpn",
    "lagrange",
    "make_interp_spline",
]

@deprecated("will be removed in SciPy v2.0.0")
class RectBivariateSpline(_RectBivariateSpline): ...

@deprecated("will be removed in SciPy v2.0.0")
class RegularGridInterpolator(_RegularGridInterpolator): ...

@deprecated("will be removed in SciPy v2.0.0")
class BPoly(_BPoly): ...

@deprecated("will be removed in SciPy v2.0.0")
class NdPPoly(_NdPPoly): ...

@deprecated("will be removed in SciPy v2.0.0")
class PPoly(_PPoly): ...

@deprecated("will be removed in SciPy v2.0.0")
class BSpline(_BSpline): ...

@deprecated("will be removed in SciPy v2.0.0")
class interp1d(_interp1d): ...

@deprecated("will be removed in SciPy v2.0.0")
class interp2d(_interp2d): ...

@deprecated("will be removed in SciPy v2.0.0")
def interpn(
    points: object,
    values: object,
    xi: object,
    method: object = ...,
    bounds_error: object = ...,
    fill_value: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def lagrange(x: object, w: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def make_interp_spline(
    x: object,
    y: object,
    k: object = ...,
    t: object = ...,
    bc_type: object = ...,
    axis: object = ...,
    check_finite: object = ...,
) -> object: ...
