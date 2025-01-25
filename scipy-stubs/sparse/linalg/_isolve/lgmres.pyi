from collections.abc import Callable
from typing import Any, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy.sparse._base import _spbase
from scipy.sparse._typing import Numeric
from scipy.sparse.linalg import LinearOperator

__all__ = ["lgmres"]

_Float: TypeAlias = np.float32 | np.float64
_Complex: TypeAlias = np.complex64 | np.complex128

_FloatT = TypeVar("_FloatT", bound=_Float, default=np.float64)
_ComplexT = TypeVar("_ComplexT", bound=_Complex, default=np.complex128)
_ScalarT = TypeVar("_ScalarT", bound=Numeric)

_ToInt: TypeAlias = np.integer[Any] | np.bool_
_ToLinearOperator: TypeAlias = onp.CanArrayND[_ScalarT] | _spbase[_ScalarT] | LinearOperator[_ScalarT]

_Ignored: TypeAlias = object
_Callback: TypeAlias = Callable[[onp.Array1D[_ScalarT]], _Ignored]

###

@overload
def lgmres(
    A: _ToLinearOperator[_FloatT | _ToInt],
    b: onp.ToFloat1D,
    x0: onp.ToFloat1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | None = None,
    M: _ToLinearOperator[_FloatT | _ToInt] | None = None,
    callback: _Callback[_FloatT] | None = None,
    inner_m: int = 30,
    outer_k: int = 3,
    outer_v: list[tuple[onp.ArrayND[_Float], onp.ArrayND[_Float] | None]] | None = None,
    store_outer_Av: onp.ToBool = True,
    prepend_outer_v: onp.ToBool = False,
) -> tuple[onp.Array1D[_FloatT], int]: ...
@overload
def lgmres(
    A: _ToLinearOperator[_ComplexT],
    b: onp.ToComplex1D,
    x0: onp.ToComplex1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | None = None,
    M: _ToLinearOperator[_ComplexT] | None = None,
    callback: _Callback[_ComplexT] | None = None,
    inner_m: int = 30,
    outer_k: int = 3,
    outer_v: list[tuple[onp.ArrayND[_Float | _Complex], onp.ArrayND[_Complex] | None]] | None = None,
    store_outer_Av: onp.ToBool = True,
    prepend_outer_v: onp.ToBool = False,
) -> tuple[onp.Array1D[_ComplexT], int]: ...
