# pyright: reportUnannotatedClassAttribute=false

# needed (once) for `numpy>=2.2.0`
# mypy: disable-error-code="overload-overlap"

from collections.abc import Sequence
from typing import Generic, TypeAlias, overload
from typing_extensions import Self, TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
import optype.typing as opt
from ._base import _spbase
from ._bsr import bsr_matrix
from ._coo import coo_matrix
from ._csc import csc_matrix
from ._csr import csr_matrix
from ._dia import dia_matrix
from ._dok import dok_matrix
from ._lil import lil_matrix
from ._typing import Complex, Float, Int, Scalar, SPFormat, ToShape2D

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=Scalar)
_SCT_co = TypeVar("_SCT_co", bound=Scalar, default=Scalar, covariant=True)

_IntInT = TypeVar("_IntInT", bound=Int | Float | Complex)
_FloatInT = TypeVar("_FloatInT", bound=Float | Complex)
_ComplexInT = TypeVar("_ComplexInT", bound=Complex)

_SPIntInT = TypeVar("_SPIntInT", bound=spmatrix[Int | Float | Complex])
_SPFloatInT = TypeVar("_SPFloatInT", bound=spmatrix[Float | Complex])
_SPComplexInT = TypeVar("_SPComplexInT", bound=spmatrix[Complex])

_ToBool: TypeAlias = np.bool_
_ToInt: TypeAlias = Int | _ToBool
_ToFloat: TypeAlias = Float | _ToInt

_DualMatrixLike: TypeAlias = _T | _SCT | _spbase[_SCT]
_DualArrayLike: TypeAlias = Sequence[Sequence[_T | _SCT] | onp.CanArrayND[_SCT]] | onp.CanArrayND[_SCT]

###

class spmatrix(Generic[_SCT_co]):
    @property
    def _bsr_container(self, /) -> bsr_matrix[_SCT_co]: ...
    @property
    def _coo_container(self, /) -> coo_matrix[_SCT_co]: ...
    @property
    def _csc_container(self, /) -> csc_matrix[_SCT_co]: ...
    @property
    def _csr_container(self, /) -> csr_matrix[_SCT_co]: ...
    @property
    def _dia_container(self, /) -> dia_matrix[_SCT_co]: ...
    @property
    def _dok_container(self, /) -> dok_matrix[_SCT_co]: ...
    @property
    def _lil_container(self, /) -> lil_matrix[_SCT_co]: ...

    #
    @property
    def shape(self, /) -> tuple[int, int]: ...
    def get_shape(self, /) -> tuple[int, int]: ...
    def set_shape(self, /, shape: ToShape2D) -> None: ...

    #
    @overload  # other: matrix-like _SCT_co
    def __mul__(self, other: spmatrix[_SCT_co], /) -> Self: ...
    @overload  # other: scalar- or matrix-like +Bool
    def __mul__(self, other: bool | np.bool_ | _spbase[np.bool_], /) -> Self: ...
    @overload  # other: array-like +Bool
    def __mul__(self, other: onp.ToBool2D, /) -> onp.Array2D[np.bool_]: ...
    @overload  # Self[+Bool], other: scalar- or matrix-like ~Int
    def __mul__(self: spmatrix[_ToBool], other: _DualMatrixLike[opt.JustInt, Int], /) -> spmatrix[Int]: ...
    @overload  # Self[+bool], other: array-like ~Int
    def __mul__(self: spmatrix[_ToBool], other: _DualArrayLike[opt.JustInt, Int], /) -> onp.Array2D[Int]: ...
    @overload  # Self[-Int], other: scalar- or matrix-like +Int
    def __mul__(self: _SPIntInT, other: onp.ToInt | _spbase[Int], /) -> _SPIntInT: ...
    @overload  # Self[-Int], other: array-like +Int
    def __mul__(self: spmatrix[_IntInT], other: onp.ToInt2D, /) -> onp.Array2D[_IntInT]: ...
    @overload  # Self[+Int], other: scalar- or matrix-like ~Float
    def __mul__(self: spmatrix[_ToInt], other: _DualMatrixLike[opt.Just[float], Float], /) -> spmatrix[Float]: ...
    @overload  # Self[+Int], other: array-like ~Float
    def __mul__(self: spmatrix[_ToInt], other: _DualArrayLike[opt.Just[float], Float], /) -> onp.Array2D[Float]: ...
    @overload  # Self[-Float], other: scalar- or matrix-like +Float
    def __mul__(self: _SPFloatInT, other: onp.ToFloat | _spbase[Float], /) -> _SPFloatInT: ...
    @overload  # Self[-Float], other: array-like +Float
    def __mul__(self: spmatrix[_FloatInT], other: onp.ToFloat2D, /) -> onp.Array2D[_FloatInT]: ...
    @overload  # Self[+Float], other: scalar- or matrix-like ~Complex
    def __mul__(self: spmatrix[_ToFloat], other: _DualMatrixLike[opt.Just[complex], Complex], /) -> spmatrix[Complex]: ...
    @overload  # Self[+Float], other: array-like ~Complex
    def __mul__(self: spmatrix[_ToFloat], other: _DualArrayLike[opt.Just[complex], Complex], /) -> onp.Array2D[Complex]: ...
    @overload  # Self[-Complex], other: scalar- or matrix-like +Complex
    def __mul__(self: _SPComplexInT, other: onp.ToComplex | _spbase, /) -> _SPComplexInT: ...
    @overload  # Self[-Complex], other: array-like +Complex
    def __mul__(self: spmatrix[_ComplexInT], other: onp.ToComplex2D, /) -> onp.Array2D[_ComplexInT]: ...
    __rmul__ = __mul__

    #
    def __pow__(self, rhs: op.CanIndex, /) -> Self: ...

    #
    def getmaxprint(self, /) -> int: ...
    def getformat(self, /) -> SPFormat: ...
    # NOTE: `axis` is only supported by `{coo,csc,csr,lil}_matrix`
    def getnnz(self, /, axis: None = None) -> int: ...
    def getH(self, /) -> Self: ...
    def getcol(self, /, j: onp.ToJustInt) -> csc_matrix[_SCT_co]: ...
    def getrow(self, /, i: onp.ToJustInt) -> csr_matrix[_SCT_co]: ...

    # NOTE: mypy reports a false positive for overlapping overloads
    @overload
    def asfptype(self: spmatrix[np.bool_ | np.int8 | np.int16 | np.uint8 | np.uint16], /) -> spmatrix[np.float32]: ...
    @overload
    def asfptype(self: spmatrix[np.int32 | np.int64 | np.uint32 | np.uint64], /) -> spmatrix[np.float64]: ...
    @overload
    def asfptype(self, /) -> Self: ...
