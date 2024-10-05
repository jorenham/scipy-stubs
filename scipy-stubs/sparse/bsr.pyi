# This module is not meant for public use and will be removed in SciPy v2.0.0.
# This stub simply re-exports the imported functions.
# TODO: Add type annotated dummy functions marked deprecated.
# TODO: The imports are actually from a generated file which doesn't currently have a stub.
from ._bsr import *

__all__ = [
    "bsr_matmat",
    "bsr_matrix",
    "bsr_matvec",
    "bsr_matvecs",
    "bsr_sort_indices",
    "bsr_tocsr",
    "bsr_transpose",
    "check_shape",
    "csr_matmat_maxnnz",
    "getdata",
    "getdtype",
    "isshape",
    "isspmatrix_bsr",
    "spmatrix",
    "to_native",
    "upcast",
    "warn",
]
