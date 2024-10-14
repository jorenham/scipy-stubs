# This module is not meant for public use and will be removed in SciPy v2.0.0.
import itertools
from typing import type_check_only
from typing_extensions import Self, deprecated

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
    "itertools",
    "spmatrix",
    "upcast",
    "upcast_scalar",
]

@deprecated("will be removed in SciPy v2.0.0")
class spmatrix:
    @property
    def shape(self) -> tuple[int, ...]: ...
    def __mul__(self, other: object, /) -> object: ...
    def __rmul__(self, other: object, /) -> object: ...
    def __pow__(self, power: object, /) -> object: ...
    def set_shape(self, shape: object) -> None: ...
    def get_shape(self) -> tuple[int, ...]: ...
    def asfptype(self) -> object: ...
    def getmaxprint(self) -> object: ...
    def getformat(self) -> object: ...
    def getnnz(self, axis: object = ...) -> object: ...
    def getH(self) -> object: ...
    def getcol(self, j: int) -> object: ...
    def getrow(self, i: int) -> object: ...

@deprecated("will be removed in SciPy v2.0.0")
class dok_matrix:
    def __init__(
        self,
        arg1: object,
        shape: object = ...,
        dtype: object = ...,
        copy: object = ...,
    ) -> None: ...
    def update(self, val: object) -> None: ...
    def setdefault(self, key: object, default: object = ..., /) -> object: ...
    def __delitem__(self, key: object, /) -> None: ...
    def __or__(self, other: object, /) -> object: ...
    def __ror__(self, other: object, /) -> object: ...
    def __ior__(self, other: object, /) -> Self: ...
    def __iter__(self) -> object: ...
    def __reversed__(self) -> object: ...
    def get(self, key: object, default: object = ...) -> object: ...
    def conjtransp(self) -> object: ...
    @classmethod
    def fromkeys(cls, iterable: object, value: object = ..., /) -> Self: ...
    @property
    def shape(self) -> object: ...
    def get_shape(self) -> object: ...
    def set_shape(self, shape: object) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
class IndexMixin:
    def __getitem__(self, key: object, /) -> object: ...
    def __setitem__(self, key: object, x: object, /) -> None: ...

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def upcast(*args: object) -> object: ...
@type_check_only
@deprecated("will be removed in SciPy v2.0.0")
def to_native(A: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def getdtype(dtype: object, a: object = ..., default: object = ...) -> object: ...
@type_check_only
@deprecated("will be removed in SciPy v2.0.0")
def getdata(obj: object, dtype: object = ..., copy: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isshape(x: object, nonneg: object = ..., *, allow_1d: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def check_shape(args: object, current_shape: object = ..., *, allow_1d: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isdense(x: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isintlike(x: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isscalarlike(x: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def upcast_scalar(dtype: object, scalar: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def isspmatrix_dok(x: object) -> object: ...
