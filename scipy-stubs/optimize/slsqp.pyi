# This file is not meant for public use and will be removed in SciPy v2.0.0.

from numpy import zeros  # noqa: ICN003
from ._optimize import OptimizeResult
from ._slsqp_py import fmin_slsqp, slsqp

__all__ = ["OptimizeResult", "fmin_slsqp", "slsqp", "zeros"]
