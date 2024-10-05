# This module is not meant for public use and will be removed in SciPy v2.0.0.
# This stub simply re-exports the imported functions.
# TODO: Add type annotated dummy functions marked deprecated.
from ._base import *
from ._compressed import *
from ._index import *
from ._sputils import *

__all__ = [
    "IndexMixin",
    "SparseEfficiencyWarning",
    "check_shape",
    "csr_column_index1",
    "csr_column_index2",
    "csr_row_index",
    "csr_row_slice",
    "csr_sample_offsets",
    "csr_sample_values",
    "csr_todense",
    "downcast_intp_index",
    "get_csr_submatrix",
    "get_sum_dtype",
    "getdtype",
    "is_pydata_spmatrix",
    "isdense",
    "isintlike",
    "isscalarlike",
    "isshape",
    "operator",
    "to_native",
    "upcast",
    "upcast_char",
    "warn",
]
