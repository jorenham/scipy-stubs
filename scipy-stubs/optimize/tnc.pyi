# This file is not meant for public use and will be removed in SciPy v2.0.0.

from numpy import zeros  # noqa: ICN003
from ._optimize import OptimizeResult
from ._tnc import fmin_tnc

__all__ = ["OptimizeResult", "fmin_tnc", "zeros"]
