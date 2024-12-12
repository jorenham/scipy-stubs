# mypy: disable-error-code="misc, override"
# pyright: reportIncompatibleMethodOverride=false

from collections.abc import Iterable
from typing import Literal, NoReturn
from typing_extensions import Never, Self, override

import optype as op
from scipy._typing import Untyped
from ._base import _spbase, sparray
from ._index import IndexMixin
from ._matrix import spmatrix

__all__ = ["dok_array", "dok_matrix", "isspmatrix_dok"]

# TODO(jorenham): generic dtype and shape
class _dok_base(_spbase, IndexMixin, dict[tuple[int, int], Untyped]):
    dtype: Untyped

    @property
    @override
    def format(self, /) -> Literal["dok"]: ...

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
    @override
    def __delitem__(self, key: op.CanIndex, /) -> None: ...
    @override
    def __or__(self, other: Never, /) -> NoReturn: ...
    @override
    def __ror__(self, other: Never, /) -> NoReturn: ...
    @override
    def __ior__(self, other: Never, /) -> Self: ...
    @override
    def get(self, key: Untyped, /, default: float = 0.0) -> Untyped: ...
    @classmethod
    @override
    def fromkeys(cls, iterable: Iterable[tuple[int, ...]], value: int = 1, /) -> Self: ...

    #
    @override
    def update(self, /, val: Untyped) -> None: ...
    @override
    def setdefault(self, key: Untyped, default: Untyped | None = None, /) -> Untyped: ...

    #
    def conjtransp(self, /) -> Untyped: ...

class dok_array(_dok_base, sparray): ...

class dok_matrix(spmatrix, _dok_base):
    @property
    @override
    def shape(self, /) -> tuple[int, int]: ...
    @override
    def get_shape(self, /) -> tuple[int, int]: ...

def isspmatrix_dok(x: Untyped) -> bool: ...
