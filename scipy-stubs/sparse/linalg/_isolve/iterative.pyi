from collections.abc import Callable
from typing import Any, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp
from scipy._typing import Untyped
from scipy.sparse._base import _spbase
from scipy.sparse._typing import Scalar
from scipy.sparse.linalg import LinearOperator

__all__ = ["bicg", "bicgstab", "cg", "cgs", "gmres", "qmr"]

_Float: TypeAlias = np.float32 | np.float64
_Complex: TypeAlias = np.complex64 | np.complex128

_ToInt: TypeAlias = np.integer[Any] | np.bool_
_ToFloat: TypeAlias = _Float | _ToInt
_ToComplex: TypeAlias = _Complex | _ToFloat

_FloatT = TypeVar("_FloatT", bound=_Float, default=np.float64)
_ComplexT = TypeVar("_ComplexT", bound=_Complex)
_ScalarT = TypeVar("_ScalarT", bound=Scalar)

_ToLinearOperator: TypeAlias = onp.CanArrayND[_ScalarT] | _spbase[_ScalarT] | LinearOperator[_ScalarT]

_Ignored: TypeAlias = object
_Callback: TypeAlias = Callable[[onp.Array1D[_ScalarT]], _Ignored]

###

def bicg(
    A: Untyped,
    b: Untyped,
    x0: Untyped | None = None,
    *,
    rtol: float = 1e-05,
    atol: float = 0.0,
    maxiter: int | None = None,
    M: Untyped | None = None,
    callback: Untyped | None = None,
) -> Untyped: ...
def bicgstab(
    A: Untyped,
    b: Untyped,
    x0: Untyped | None = None,
    *,
    rtol: float = 1e-05,
    atol: float = 0.0,
    maxiter: int | None = None,
    M: Untyped | None = None,
    callback: Untyped | None = None,
) -> Untyped: ...

#
def cg(
    A: Untyped,
    b: Untyped,
    x0: Untyped | None = None,
    *,
    rtol: float = 1e-05,
    atol: float = 0.0,
    maxiter: int | None = None,
    M: Untyped | None = None,
    callback: Untyped | None = None,
) -> Untyped: ...
def cgs(
    A: Untyped,
    b: Untyped,
    x0: Untyped | None = None,
    *,
    rtol: float = 1e-05,
    atol: float = 0.0,
    maxiter: int | None = None,
    M: Untyped | None = None,
    callback: Untyped | None = None,
) -> Untyped: ...

#
def gmres(
    A: Untyped,
    b: Untyped,
    x0: Untyped | None = None,
    *,
    rtol: float = 1e-05,
    atol: float = 0.0,
    restart: Untyped | None = None,
    maxiter: int | None = None,
    M: Untyped | None = None,
    callback: Untyped | None = None,
    callback_type: Untyped | None = None,
) -> Untyped: ...

#
@overload  # real
def qmr(
    A: _ToLinearOperator[_FloatT | _ToInt],
    b: onp.ToFloat1D,
    x0: onp.ToFloat1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | None = None,
    M1: _ToLinearOperator[_ToFloat] | None = None,
    M2: _ToLinearOperator[_ToFloat] | None = None,
    callback: _Callback[_FloatT] | None = None,
) -> tuple[onp.Array1D[_FloatT], int]: ...
@overload  # complex
def qmr(
    A: _ToLinearOperator[_ComplexT],
    b: onp.ToComplex1D,
    x0: onp.ToComplex1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | None = None,
    M1: _ToLinearOperator[_ToComplex] | None = None,
    M2: _ToLinearOperator[_ToComplex] | None = None,
    callback: _Callback[_ComplexT] | None = None,
) -> tuple[onp.Array1D[_ComplexT], int]: ...
