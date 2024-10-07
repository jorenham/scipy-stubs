# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

__all__ = ["cspline2d", "qspline2d", "sepfir2d", "symiirorder1", "symiirorder2"]

# _spline_filters
@deprecated("will be removed in SciPy v2.0.0")
def qspline2d(signal: object, lamb: object = ..., precision: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def cspline2d(signal: object, lamb: object = ..., precision: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def symiirorder1(signal: object, c0: object, z1: object, precision: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def symiirorder2(input: object, r: object, omega: object, precision: object = ...) -> object: ...

# _spline
@deprecated("will be removed in SciPy v2.0.0")
def sepfir2d(input: object, hrow: object, hcol: object) -> object: ...
