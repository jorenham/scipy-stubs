from .._lib._util import copy_if_needed as copy_if_needed
from ._base import issparse as issparse, sparray as sparray
from ._data import _data_matrix
from ._matrix import spmatrix as spmatrix
from ._sparsetools import dia_matvec as dia_matvec
from ._sputils import (
    check_shape as check_shape,
    get_sum_dtype as get_sum_dtype,
    getdtype as getdtype,
    isshape as isshape,
    upcast_char as upcast_char,
    validateaxis as validateaxis,
)
from scipy._typing import Untyped

__docformat__: str

class _dia_base(_data_matrix):
    data: Untyped
    offsets: Untyped
    def __init__(
        self,
        arg1,
        shape: Untyped | None = None,
        dtype: Untyped | None = None,
        copy: bool = False,
        *,
        maxprint: Untyped | None = None,
    ): ...
    def count_nonzero(self, axis: Untyped | None = None) -> Untyped: ...
    def sum(self, axis: Untyped | None = None, dtype: Untyped | None = None, out: Untyped | None = None) -> Untyped: ...
    def todia(self, copy: bool = False) -> Untyped: ...
    def transpose(self, axes: Untyped | None = None, copy: bool = False) -> Untyped: ...
    def diagonal(self, k: int = 0) -> Untyped: ...
    def tocsc(self, copy: bool = False) -> Untyped: ...
    def tocoo(self, copy: bool = False) -> Untyped: ...
    def resize(self, *shape): ...

def isspmatrix_dia(x) -> Untyped: ...

class dia_array(_dia_base, sparray): ...
class dia_matrix(spmatrix, _dia_base): ...
