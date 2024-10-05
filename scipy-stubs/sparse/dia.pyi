# This module is not meant for public use and will be removed in SciPy v2.0.0.
# This stub simply re-exports the imported functions.
# TODO: Add type annotated dummy functions marked deprecated.
# TODO: The imports are actually from a generated file which doesn't currently have a stub.
from ._base import *
from ._construct import *
from ._dia import *
from ._dia import dia_matrix
from ._matrix import spmatrix
from ._sputils import *

__all__ = [
    "check_shape",
    "dia_matrix",
    "dia_matvec",
    "get_sum_dtype",
    "getdtype",
    "isshape",
    "isspmatrix_dia",
    "spmatrix",
    "upcast_char",
    "validateaxis",
]
