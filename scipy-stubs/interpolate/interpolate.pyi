# This module is not meant for public use and will be removed in SciPy v2.0.0.
# This stub simply re-exports the imported functions.
# TODO: Add type annotated dummy functions marked deprecated.
from ._fitpack2 import *
from ._interpolate import *
from ._rgi import *

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
