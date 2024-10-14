from typing import Any
from typing_extensions import Never, Self

import numpy as np
from scipy._typing import Untyped
from ._base import _spbase
from ._sputils import _ScalarLike

__all__: list[str] = []

class _data_matrix(_spbase):
    def __init__(self, /, arg1: Untyped) -> None: ...
    def __imul__(self, other: _ScalarLike, /) -> Self: ...  # type: ignore[misc,override]
    def __itruediv__(self, other: _ScalarLike, /) -> Self: ...  # type: ignore[misc,override]
    @property
    def dtype(self, /) -> np.dtype[np.number[Any] | np.bool_]: ...
    @dtype.setter
    def dtype(self, newtype: Never, /) -> None: ...

    #
    def arcsin(self, /) -> Self: ...
    def arcsinh(self, /) -> Self: ...
    def arctan(self, /) -> Self: ...
    def arctanh(self, /) -> Self: ...
    def ceil(self, /) -> Self: ...
    def deg2rad(self, /) -> Self: ...
    def expm1(self, /) -> Self: ...
    def floor(self, /) -> Self: ...
    def log1p(self, /) -> Self: ...
    def rad2deg(self, /) -> Self: ...
    def rint(self, /) -> Self: ...
    def sign(self, /) -> Self: ...
    def sin(self, /) -> Self: ...
    def sinh(self, /) -> Self: ...
    def sqrt(self, /) -> Self: ...
    def tan(self, /) -> Self: ...
    def tanh(self, /) -> Self: ...
    def trunc(self, /) -> Self: ...

class _minmax_mixin:
    def max(self, axis: int | None = None, out: Untyped | None = None) -> Untyped: ...
    def min(self, axis: int | None = None, out: Untyped | None = None) -> Untyped: ...
    def nanmax(self, axis: int | None = None, out: Untyped | None = None) -> Untyped: ...
    def nanmin(self, axis: int | None = None, out: Untyped | None = None) -> Untyped: ...
    def argmax(self, axis: int | None = None, out: Untyped | None = None) -> Untyped: ...
    def argmin(self, axis: int | None = None, out: Untyped | None = None) -> Untyped: ...
