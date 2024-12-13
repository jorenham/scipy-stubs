import abc
from typing import Any, Generic, Literal, overload
from typing_extensions import Never, Self, TypeVar, override

import numpy as np
import optype as op
from scipy._typing import Untyped
from ._base import _spbase
from ._sputils import _ScalarLike
from ._typing import Scalar

__all__: list[str] = []

_SCT_co = TypeVar("_SCT_co", bound=Scalar, default=Scalar, covariant=True)

###

class _data_matrix(_spbase[_SCT_co], Generic[_SCT_co]):
    #
    @property
    @abc.abstractmethod
    @override
    def format(self, /) -> Literal["bsr", "coo", "csc", "csr", "dia"]: ...

    #
    @property
    def dtype(self, /) -> np.dtype[_SCT_co]: ...
    @dtype.setter
    def dtype(self, newtype: Never, /) -> None: ...

    #
    def __init__(self, /, arg1: Untyped) -> None: ...

    #
    def __imul__(self, other: _ScalarLike, /) -> Self: ...  # type: ignore[misc]
    def __itruediv__(self, other: _ScalarLike, /) -> Self: ...  # type: ignore[misc]

    # NOTE: The following methods do not convert the scalar type
    def sign(self, /) -> Self: ...
    def ceil(self, /) -> Self: ...
    def floor(self, /) -> Self: ...
    def rint(self, /) -> Self: ...
    def trunc(self, /) -> Self: ...

    # TODO: No HKT: Need to override in subtypes
    #
    @overload
    def sqrt(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64]: ...
    @overload
    def sqrt(self, /) -> Self: ...
    #
    @overload
    def expm1(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64]: ...
    @overload
    def expm1(self, /) -> Self: ...
    #
    @overload
    def log1p(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64]: ...
    @overload
    def log1p(self, /) -> Self: ...

    #
    @overload
    def sin(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64]: ...
    @overload
    def sin(self, /) -> Self: ...
    #
    @overload
    def arcsin(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64]: ...
    @overload
    def arcsin(self, /) -> Self: ...
    #
    @overload
    def sinh(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64]: ...
    @overload
    def sinh(self, /) -> Self: ...
    #
    @overload
    def arcsinh(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64]: ...
    @overload
    def arcsinh(self, /) -> Self: ...
    #
    @overload
    def tan(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64]: ...
    @overload
    def tan(self, /) -> Self: ...
    #
    @overload
    def arctan(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64]: ...
    @overload
    def arctan(self, /) -> Self: ...
    #
    @overload
    def tanh(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64]: ...
    @overload
    def tanh(self, /) -> Self: ...
    #
    @overload
    def arctanh(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64]: ...
    @overload
    def arctanh(self, /) -> Self: ...

    #
    @overload
    def deg2rad(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64]: ...
    @overload
    def deg2rad(self, /) -> Self: ...
    #
    @overload
    def rad2deg(self: _data_matrix[np.integer[Any] | np.bool_], /) -> _data_matrix[np.float32 | np.float64]: ...
    @overload
    def rad2deg(self, /) -> Self: ...

# TODO(jorenham)
class _minmax_mixin(Generic[_SCT_co]):
    def max(self, /, axis: op.CanIndex | None = None, out: Untyped | None = None) -> Untyped: ...
    def min(self, /, axis: op.CanIndex | None = None, out: Untyped | None = None) -> Untyped: ...
    def nanmax(self, /, axis: op.CanIndex | None = None, out: Untyped | None = None) -> Untyped: ...
    def nanmin(self, /, axis: op.CanIndex | None = None, out: Untyped | None = None) -> Untyped: ...
    def argmax(self, /, axis: op.CanIndex | None = None, out: Untyped | None = None) -> Untyped: ...
    def argmin(self, /, axis: op.CanIndex | None = None, out: Untyped | None = None) -> Untyped: ...
