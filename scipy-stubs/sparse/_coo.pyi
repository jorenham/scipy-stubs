from typing import Literal
from typing_extensions import override

from scipy._typing import Untyped
from ._base import sparray
from ._data import _data_matrix, _minmax_mixin
from ._matrix import spmatrix

__all__ = ["coo_array", "coo_matrix", "isspmatrix_coo"]

# TODO(jorenham): generic dtype and shape
class _coo_base(_data_matrix, _minmax_mixin):
    coords: Untyped
    data: Untyped
    has_canonical_format: bool

    @property
    @override
    def format(self, /) -> Literal["coo"]: ...
    #
    @property
    def row(self, /) -> Untyped: ...
    @row.setter
    def row(self, /, new_row: Untyped) -> None: ...
    #
    @property
    def col(self, /) -> Untyped: ...
    @col.setter
    def col(self, /, new_col: Untyped) -> None: ...

    #
    def __init__(
        self,
        /,
        arg1: Untyped,
        shape: Untyped | None = None,
        dtype: Untyped | None = None,
        copy: bool = False,
    ) -> None: ...

    #
    def sum_duplicates(self, /) -> None: ...
    def eliminate_zeros(self, /) -> Untyped: ...

class coo_array(_coo_base, sparray): ...
class coo_matrix(spmatrix, _coo_base): ...

def isspmatrix_coo(x: Untyped) -> bool: ...
