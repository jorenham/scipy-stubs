# This file is not meant for public use and will be removed in SciPy v2.0.0.

from ._cobyla_py import fmin_cobyla
from ._optimize import OptimizeResult

__all__ = ["OptimizeResult", "fmin_cobyla"]
