# This file is not meant for public use and will be removed in SciPy v2.0.0.

from numpy.linalg import norm  # noqa: ICN003
from ._decomp import eigvals
from ._decomp_schur import rsf2csf, schur
from ._misc import LinAlgError
from .lapack import get_lapack_funcs

__all__ = ["LinAlgError", "eigvals", "get_lapack_funcs", "norm", "rsf2csf", "schur"]
