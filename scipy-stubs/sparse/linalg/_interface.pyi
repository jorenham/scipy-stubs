# mypy: disable-error-code="override"
# pyright: reportUnannotatedClassAttribute=false
# pyright: reportOverlappingOverload=false

from collections.abc import Callable, Iterable
from typing import Any, ClassVar, Final, Generic, Literal, Protocol, TypeAlias, final, overload, type_check_only
from typing_extensions import Self, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp
from scipy.sparse._base import _spbase
from scipy.sparse._typing import Complex, Float, Int, Matrix, Scalar, ToDTypeComplex, ToDTypeFloat, ToDTypeInt

__all__ = ["LinearOperator", "aslinearoperator"]

_Real: TypeAlias = np.bool_ | Int | Float
_Inexact: TypeAlias = Float | Complex
_Number: TypeAlias = Int | _Inexact

_SCT = TypeVar("_SCT", bound=Scalar)
_SCT_co = TypeVar("_SCT_co", bound=Scalar, default=_Inexact, covariant=True)
_SCT1_co = TypeVar("_SCT1_co", bound=Scalar, default=_Inexact, covariant=True)
_SCT2_co = TypeVar("_SCT2_co", bound=Scalar, default=_SCT1_co, covariant=True)
_FunMatVecT_co = TypeVar("_FunMatVecT_co", bound=_FunMatVec, default=_FunMatVec, covariant=True)

_InexactT = TypeVar("_InexactT", bound=_Inexact)

_ToShape: TypeAlias = Iterable[op.CanIndex]
_ToDType: TypeAlias = type[_SCT] | np.dtype[_SCT] | onp.HasDType[np.dtype[_SCT]]

_FunMatVec: TypeAlias = Callable[[onp.Array1D[_Number] | onp.Array2D[_Number]], onp.ToComplex1D | onp.ToComplex2D]
_FunMatMat: TypeAlias = Callable[[onp.Array2D[_Number]], onp.ToComplex2D]

_Array1D2D: TypeAlias = onp.Array1D[_SCT] | onp.Array2D[_SCT]

###

class LinearOperator(Generic[_SCT_co]):
    __array_ufunc__: ClassVar[None]

    ndim: ClassVar[Literal[2]] = 2
    shape: Final[tuple[int, int]]
    dtype: np.dtype[_SCT_co]

    #
    @property
    def H(self, /) -> _AdjointLinearOperator[_SCT_co]: ...
    def adjoint(self, /) -> _AdjointLinearOperator[_SCT_co]: ...
    @property
    def T(self, /) -> _TransposedLinearOperator[_SCT_co]: ...
    def transpose(self, /) -> _TransposedLinearOperator[_SCT_co]: ...

    #
    def __new__(cls, *args: Any, **kwargs: Any) -> Self: ...

    #
    @overload
    def __init__(self, /, dtype: _ToDType[_SCT_co], shape: _ToShape) -> None: ...
    @overload
    def __init__(self: LinearOperator[np.int_], /, dtype: ToDTypeInt, shape: _ToShape) -> None: ...
    @overload
    def __init__(self: LinearOperator[np.float64], /, dtype: ToDTypeFloat, shape: _ToShape) -> None: ...
    @overload
    def __init__(self: LinearOperator[np.complex128], /, dtype: ToDTypeComplex, shape: _ToShape) -> None: ...
    @overload
    def __init__(self, /, dtype: onp.AnyInexactDType | None, shape: _ToShape) -> None: ...

    #
    @overload  # float array 1d
    def matvec(self, /, x: onp.ToFloatStrict1D) -> onp.Array1D[_SCT_co]: ...
    @overload  # float matrix
    def matvec(self, /, x: Matrix[_Real]) -> Matrix[_SCT_co]: ...
    @overload  # float array 2d
    def matvec(self, /, x: onp.ToFloatStrict2D) -> onp.Array2D[_SCT_co]: ...
    @overload  # complex array 1d
    def matvec(self, /, x: onp.ToComplexStrict1D) -> onp.Array1D[_SCT_co | np.complex128]: ...
    @overload  # complex matrix
    def matvec(self, /, x: Matrix[_Number]) -> Matrix[_SCT_co | np.complex128]: ...
    @overload  # complex array 2d
    def matvec(self, /, x: onp.ToComplexStrict2D) -> onp.Array2D[_SCT_co | np.complex128]: ...
    @overload  # float array
    def matvec(self, /, x: onp.ToFloat2D) -> onp.Array1D[_SCT_co] | onp.Array2D[_SCT_co]: ...
    @overload  # complex array
    def matvec(self, /, x: onp.ToComplex2D) -> _Array1D2D[_SCT_co | np.complex128]: ...
    rmatvec = matvec

    #
    def matmat(self, /, X: onp.ToComplex2D) -> onp.Array[tuple[int, int], _SCT_co | np.complex128]: ...
    rmatmat = matmat

    #
    @overload
    def dot(self, /, x: LinearOperator[_SCT]) -> _ProductLinearOperator[_SCT_co, _SCT]: ...
    @overload
    def dot(self, /, x: onp.ToFloat) -> _ScaledLinearOperator[_SCT_co]: ...
    @overload
    def dot(self, /, x: onp.ToComplex) -> _ScaledLinearOperator[_SCT_co | np.complex128]: ...
    @overload
    def dot(self, /, x: onp.ToFloatStrict1D) -> onp.Array1D[_SCT_co]: ...
    @overload
    def dot(self, /, x: onp.ToComplexStrict1D) -> onp.Array1D[_SCT_co | np.complex128]: ...
    @overload
    def dot(self, /, x: onp.ToFloatStrict2D) -> onp.Array2D[_SCT_co]: ...
    @overload
    def dot(self, /, x: onp.ToComplexStrict2D) -> onp.Array2D[_SCT_co | np.complex128]: ...
    @overload
    def dot(self, /, x: onp.ToFloatND) -> onp.Array1D[_SCT_co] | onp.Array2D[_SCT_co]: ...
    @overload
    def dot(self, /, x: onp.ToComplexND) -> _Array1D2D[_SCT_co | np.complex128]: ...
    __mul__ = dot
    __rmul__ = dot
    __call__ = dot

    #
    @overload
    def __matmul__(self, /, x: LinearOperator[_SCT]) -> _ProductLinearOperator[_SCT_co, _SCT]: ...
    @overload
    def __matmul__(self, /, x: onp.ToFloatStrict1D) -> onp.Array1D[_SCT_co]: ...
    @overload
    def __matmul__(self, /, x: onp.ToComplexStrict1D) -> onp.Array1D[_SCT_co | np.complex128]: ...
    @overload
    def __matmul__(self, /, x: onp.ToFloatStrict2D) -> onp.Array2D[_SCT_co]: ...
    @overload
    def __matmul__(self, /, x: onp.ToComplexStrict2D) -> onp.Array2D[_SCT_co | np.complex128]: ...
    @overload
    def __matmul__(self, /, x: onp.ToFloatND) -> onp.Array1D[_SCT_co] | onp.Array2D[_SCT_co]: ...
    @overload
    def __matmul__(self, /, x: onp.ToComplexND) -> _Array1D2D[_SCT_co | np.complex128]: ...
    __rmatmul__ = __matmul__

    #
    @overload
    def __truediv__(self, other: onp.ToFloat, /) -> _ScaledLinearOperator[_SCT_co]: ...
    @overload
    def __truediv__(self, other: onp.ToComplex, /) -> _ScaledLinearOperator[_SCT_co | np.complex128]: ...

    #
    def __neg__(self, /) -> _ScaledLinearOperator[_SCT_co]: ...
    def __add__(self, x: LinearOperator[_SCT], /) -> _SumLinearOperator[_SCT_co, _SCT]: ...
    __sub__ = __add__
    def __pow__(self, p: onp.ToInt, /) -> _PowerLinearOperator[_SCT_co]: ...

@final
class _CustomLinearOperator(LinearOperator[_SCT_co], Generic[_SCT_co, _FunMatVecT_co]):
    args: tuple[()]

    @overload  # no dtype
    def __init__(
        self,
        /,
        shape: _ToShape,
        matvec: _FunMatVec,
        rmatvec: _FunMatVec | None = None,
        matmat: _FunMatMat | None = None,
        dtype: None = None,
        rmatmat: _FunMatMat | None = None,
    ) -> None: ...
    @overload  # dtype known (positional)
    def __init__(
        self,
        /,
        shape: _ToShape,
        matvec: _FunMatVec,
        rmatvec: _FunMatVec | None,
        matmat: _FunMatMat | None,
        dtype: _ToDType[_SCT_co],
        rmatmat: _FunMatMat | None = None,
    ) -> None: ...
    @overload  # dtype known (keyword)
    def __init__(
        self,
        /,
        shape: _ToShape,
        matvec: _FunMatVec,
        rmatvec: _FunMatVec | None = None,
        matmat: _FunMatMat | None = None,
        *,
        dtype: _ToDType[_SCT_co],
        rmatmat: _FunMatMat | None = None,
    ) -> None: ...
    @overload  # dtype-like float64 (positional)
    def __init__(
        self: _CustomLinearOperator[np.float64],
        /,
        shape: _ToShape,
        matvec: _FunMatVec,
        rmatvec: _FunMatVec | None,
        matmat: _FunMatMat | None,
        dtype: ToDTypeFloat,
        rmatmat: _FunMatMat | None = None,
    ) -> None: ...
    @overload  # dtype-like float64 (keyword)
    def __init__(
        self: _CustomLinearOperator[np.float64],
        /,
        shape: _ToShape,
        matvec: _FunMatVec,
        rmatvec: _FunMatVec | None = None,
        matmat: _FunMatMat | None = None,
        *,
        dtype: ToDTypeFloat,
        rmatmat: _FunMatMat | None = None,
    ) -> None: ...
    @overload  # dtype-like complex128 (positional)
    def __init__(
        self: _CustomLinearOperator[np.complex128],
        /,
        shape: _ToShape,
        matvec: _FunMatVec,
        rmatvec: _FunMatVec | None,
        matmat: _FunMatMat | None,
        dtype: ToDTypeComplex,
        rmatmat: _FunMatMat | None = None,
    ) -> None: ...
    @overload  # dtype-like complex128 (keyword)
    def __init__(
        self: _CustomLinearOperator[np.complex128],
        /,
        shape: _ToShape,
        matvec: _FunMatVec,
        rmatvec: _FunMatVec | None = None,
        matmat: _FunMatMat | None = None,
        *,
        dtype: ToDTypeComplex,
        rmatmat: _FunMatMat | None = None,
    ) -> None: ...

@type_check_only
class _UnaryLinearOperator(LinearOperator[_SCT_co], Generic[_SCT_co]):
    A: LinearOperator[_SCT_co]
    args: tuple[LinearOperator[_SCT_co]]

    def __init__(self, /, A: LinearOperator[_SCT_co]) -> None: ...

@final
class _AdjointLinearOperator(_UnaryLinearOperator[_SCT_co], Generic[_SCT_co]): ...

@final
class _TransposedLinearOperator(_UnaryLinearOperator[_SCT_co], Generic[_SCT_co]): ...

@final
class _SumLinearOperator(LinearOperator[_SCT1_co | _SCT2_co], Generic[_SCT1_co, _SCT2_co]):
    args: tuple[LinearOperator[_SCT1_co], LinearOperator[_SCT2_co]]

    def __init__(self, /, A: LinearOperator[_SCT1_co], B: LinearOperator[_SCT2_co]) -> None: ...

@final
class _ProductLinearOperator(LinearOperator[_SCT1_co | _SCT2_co], Generic[_SCT1_co, _SCT2_co]):
    args: tuple[LinearOperator[_SCT1_co], LinearOperator[_SCT2_co]]

    def __init__(self, /, A: LinearOperator[_SCT1_co], B: LinearOperator[_SCT2_co]) -> None: ...

@final
class _ScaledLinearOperator(LinearOperator[_SCT_co], Generic[_SCT_co]):
    args: tuple[LinearOperator[_SCT_co], _SCT_co | complex]
    @overload
    def __init__(self, /, A: LinearOperator[_SCT_co], alpha: _SCT_co | complex) -> None: ...
    @overload
    def __init__(self: _ScaledLinearOperator[np.float64], /, A: LinearOperator[Float], alpha: float) -> None: ...
    @overload
    def __init__(self: _ScaledLinearOperator[np.complex128], /, A: LinearOperator, alpha: complex) -> None: ...

@final
class _PowerLinearOperator(LinearOperator[_SCT_co], Generic[_SCT_co]):
    args: tuple[LinearOperator[_SCT_co], op.CanIndex]

    def __init__(self, /, A: LinearOperator[_SCT_co], p: op.CanIndex) -> None: ...

class MatrixLinearOperator(LinearOperator[_SCT_co], Generic[_SCT_co]):
    A: _spbase | onp.Array2D[_SCT_co]
    args: tuple[_spbase | onp.Array2D[_SCT_co]]

    def __init__(self, /, A: _spbase | onp.ArrayND[_SCT_co]) -> None: ...

@final
class _AdjointMatrixOperator(MatrixLinearOperator[_SCT_co], Generic[_SCT_co]):
    args: tuple[MatrixLinearOperator[_SCT_co]]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleVariableOverride]
    @property
    @override
    def dtype(self, /) -> np.dtype[_SCT_co]: ...  # pyright: ignore[reportIncompatibleVariableOverride]
    def __init__(self, /, adjoint: LinearOperator) -> None: ...

class IdentityOperator(LinearOperator[_SCT_co], Generic[_SCT_co]):
    @overload
    def __init__(self, /, shape: _ToShape, dtype: _ToDType[_SCT_co]) -> None: ...
    @overload
    def __init__(self: IdentityOperator[np.float64], /, shape: _ToShape, dtype: ToDTypeFloat | None = None) -> None: ...
    @overload
    def __init__(self: IdentityOperator[np.complex128], /, shape: _ToShape, dtype: ToDTypeComplex) -> None: ...
    @overload
    def __init__(self, /, shape: _ToShape, dtype: onp.AnyInexactDType) -> None: ...

@type_check_only
class _HasShapeAndMatVec(Protocol[_SCT_co]):
    shape: tuple[int, int]
    @overload
    def matvec(self, /, x: onp.CanArray1D[np.float64]) -> onp.CanArray1D[_SCT_co]: ...
    @overload
    def matvec(self, /, x: onp.CanArray1D[np.complex128]) -> onp.ToComplex1D: ...
    @overload
    def matvec(self, /, x: onp.CanArray2D[np.float64]) -> onp.CanArray2D[_SCT_co]: ...
    @overload
    def matvec(self, /, x: onp.CanArray2D[np.complex128]) -> onp.ToComplex2D: ...

@type_check_only
class _HasShapeAndDTypeAndMatVec(Protocol[_SCT_co]):
    shape: tuple[int, int]
    @property
    def dtype(self, /) -> np.dtype[_SCT_co]: ...
    @overload
    def matvec(self, /, x: onp.CanArray1D[np.float64] | onp.CanArray1D[np.complex128]) -> onp.ToComplex1D: ...
    @overload
    def matvec(self, /, x: onp.CanArray2D[np.float64] | onp.CanArray2D[np.complex128]) -> onp.ToComplex2D: ...

@overload
def aslinearoperator(A: onp.CanArrayND[_InexactT]) -> MatrixLinearOperator[_InexactT]: ...
@overload
def aslinearoperator(A: _spbase[_InexactT]) -> MatrixLinearOperator[_InexactT]: ...
@overload
def aslinearoperator(A: onp.ArrayND[np.bool_ | Int] | _spbase[np.bool_ | Int]) -> MatrixLinearOperator[np.float64]: ...
@overload
def aslinearoperator(A: _HasShapeAndDTypeAndMatVec[_InexactT]) -> MatrixLinearOperator[_InexactT]: ...
@overload
def aslinearoperator(A: _HasShapeAndMatVec[_InexactT]) -> MatrixLinearOperator[_InexactT]: ...
