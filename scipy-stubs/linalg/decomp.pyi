# This file is not meant for public use and will be removed in SciPy v2.0.0.

from ._decomp import (
    cdf2rdf,
    eig,
    eig_banded,
    eigh,
    eigh_tridiagonal,
    eigvals,
    eigvals_banded,
    eigvalsh,
    eigvalsh_tridiagonal,
    hessenberg,
)
from ._misc import LinAlgError, norm
from .lapack import get_lapack_funcs

__all__ = [
    "LinAlgError",
    "cdf2rdf",
    "eig",
    "eig_banded",
    "eigh",
    "eigh_tridiagonal",
    "eigvals",
    "eigvals_banded",
    "eigvalsh",
    "eigvalsh_tridiagonal",
    "get_lapack_funcs",
    "hessenberg",
    "norm",
]
