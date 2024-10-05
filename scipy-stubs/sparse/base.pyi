# This module is not meant for public use and will be removed in SciPy v2.0.0.
# This stub simply re-exports the imported functions.
from ._base import (
    MAXPRINT,
    SparseEfficiencyWarning,
    SparseFormatWarning,
    SparseWarning,
    issparse,
    isspmatrix,
    spmatrix,
)
from ._sputils import *

__all__ = [
    "MAXPRINT",
    "SparseEfficiencyWarning",
    "SparseFormatWarning",
    "SparseWarning",
    "asmatrix",
    "check_reshape_kwargs",
    "check_shape",
    "get_sum_dtype",
    "isdense",
    "isscalarlike",
    "issparse",
    "isspmatrix",
    "spmatrix",
    "validateaxis",
]
