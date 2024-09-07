from collections.abc import Iterable
from typing_extensions import Self, override

from scipy._typing import Untyped
from ._base import _spbase, sparray
from ._index import IndexMixin
from ._matrix import spmatrix

__all__ = ["dok_array", "dok_matrix", "isspmatrix_dok"]

class _dok_base(_spbase, IndexMixin, dict[tuple[int, ...], Untyped]):  # type: ignore[misc]
    dtype: Untyped
    def __init__(
        self,
        arg1: Untyped,
        shape: Untyped | None = None,
        dtype: Untyped | None = None,
        copy: bool = False,
        *,
        maxprint: Untyped | None = None,
    ) -> None: ...
    @override
    def update(self, val: Untyped): ...  # type: ignore[override]
    @override
    def setdefault(self, key: Untyped, default: Untyped | None = None, /) -> Untyped: ...
    @override
    def __delitem__(self, key: Untyped, /) -> None: ...
    @override
    def __or__(self, other: Untyped, /): ...  # type: ignore[override]
    @override
    def __ror__(self, other: Untyped, /): ...  # type: ignore[override]
    @override
    def __ior__(self, other: Untyped, /) -> Self: ...  # type: ignore[override]
    @override
    def get(self, key, /, default: float = 0.0) -> Untyped: ...  # type: ignore[override]
    def conjtransp(self) -> Untyped: ...
    @classmethod
    @override
    def fromkeys(cls, iterable: Iterable[tuple[int, ...]], value: int = 1, /) -> Self: ...  # type: ignore[override]

class dok_array(_dok_base, sparray): ...  # type: ignore[misc]

class dok_matrix(spmatrix, _dok_base):  # type: ignore[misc]
    @property
    @override
    def shape(self) -> tuple[int, int]: ...
    @override
    def get_shape(self) -> tuple[int, int]: ...

def isspmatrix_dok(x: Untyped) -> bool: ...
