# This module is not meant for public use and will be removed in SciPy v2.0.0.
# This stub simply re-exports the imported functions.
# TODO: Add type annotated dummy functions marked deprecated.
# TODO: The imports are actually from a generated file which doesn't currently have a stub.
from scipy._lib._util import check_random_state, rng_integers
from ._base import *
from ._bsr import bsr_matrix
from ._construct import *
from ._coo import coo_matrix
from ._csc import csc_matrix
from ._csr import csr_matrix
from ._dia import dia_matrix
from ._sputils import get_index_dtype, isscalarlike, upcast

__all__ = [
    "block_diag",
    "bmat",
    "bsr_matrix",
    "check_random_state",
    "coo_matrix",
    "csc_matrix",
    "csr_hstack",
    "csr_matrix",
    "dia_matrix",
    "diags",
    "eye",
    "get_index_dtype",
    "hstack",
    "identity",
    "isscalarlike",
    "issparse",
    "kron",
    "kronsum",
    "rand",
    "random",
    "rng_integers",
    "spdiags",
    "upcast",
    "vstack",
]
