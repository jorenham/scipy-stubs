import abc
from typing import Literal
from typing_extensions import override

import optype as op
from scipy._typing import Untyped
from ._coo import _coo_base
from ._data import _data_matrix, _minmax_mixin
from ._index import IndexMixin

__all__: list[str] = []

# TODO(jorenham): generic dtype
class _cs_matrix(_data_matrix, _minmax_mixin, IndexMixin):
    data: Untyped
    indices: Untyped
    indptr: Untyped

    @property
    @override
    @abc.abstractmethod
    def format(self, /) -> Literal["bsr", "csc", "csr"]: ...

    #
    @property
    def has_canonical_format(self, /) -> bool: ...
    @has_canonical_format.setter
    def has_canonical_format(self, /, val: bool) -> None: ...
    #
    @property
    def has_sorted_indices(self, /) -> bool: ...
    @has_sorted_indices.setter
    def has_sorted_indices(self, /, val: bool) -> None: ...

    #
    def __init__(
        self,
        /,
        arg1: Untyped,
        shape: tuple[op.CanIndex, op.CanIndex] | None = None,
        dtype: Untyped | None = None,
        copy: bool = False,
    ) -> None: ...

    #
    @override
    def count_nonzero(self, /, axis: None = None) -> int: ...

    #
    def check_format(self, /, full_check: bool = True) -> Untyped: ...
    def eliminate_zeros(self, /) -> Untyped: ...
    def sum_duplicates(self, /) -> Untyped: ...
    def sorted_indices(self, /) -> Untyped: ...

    #
    def sort_indices(self, /) -> None: ...
    def prune(self, /) -> None: ...

    #
    @override
    def tocoo(self, /, copy: bool = True) -> _coo_base: ...
