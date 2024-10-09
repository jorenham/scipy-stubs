# This file is not meant for public use and will be removed in SciPy v2.0.0.

from ._decomp_svd import diagsvd, null_space, orth, subspace_angles, svd, svdvals
from ._misc import LinAlgError
from .lapack import get_lapack_funcs

__all__ = ["LinAlgError", "diagsvd", "get_lapack_funcs", "null_space", "orth", "subspace_angles", "svd", "svdvals"]
