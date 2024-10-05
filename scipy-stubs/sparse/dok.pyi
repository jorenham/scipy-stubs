# This module is not meant for public use and will be removed in SciPy v2.0.0.
# This stub simply re-exports the imported functions.
# TODO: Add type annotated dummy functions marked deprecated.
# TODO: The imports are actually from a generated file which doesn't currently have a stub.

from ._dok import *
from ._index import IndexMixin
from ._matrix import spmatrix
from ._sputils import check_shape, getdtype, isdense, isintlike, isscalarlike, isshape, upcast, upcast_scalar

__all__ = [
    "IndexMixin",
    "check_shape",
    "dok_matrix",
    "getdtype",
    "isdense",
    "isintlike",
    "isscalarlike",
    "isshape",
    "isspmatrix_dok",
    "spmatrix",
    "upcast",
    "upcast_scalar",
]
