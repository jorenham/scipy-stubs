from collections.abc import Callable
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy.sparse._base import _spbase
from scipy.sparse._typing import Scalar
from scipy.sparse.linalg import LinearOperator

__all__ = ["gcrotmk"]

_Float: TypeAlias = np.float32 | np.float64
_Complex: TypeAlias = np.complex64 | np.complex128

_FloatT = TypeVar("_FloatT", bound=_Float, default=np.float64)
_ComplexT = TypeVar("_ComplexT", bound=_Complex, default=np.complex128)
_ScalarT = TypeVar("_ScalarT", bound=Scalar)

_ToInt: TypeAlias = np.integer[Any] | np.bool_
_ToLinearOperator: TypeAlias = onp.CanArrayND[_ScalarT] | _spbase[_ScalarT] | LinearOperator[_ScalarT]

_Ignored: TypeAlias = object
_Callback: TypeAlias = Callable[[onp.Array1D[_ScalarT]], _Ignored]

_Truncate: TypeAlias = Literal["oldest", "newest"]

###

@overload
def gcrotmk(
    A: _ToLinearOperator[_FloatT | _ToInt],
    b: onp.ToFloat1D,
    x0: onp.ToFloat1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | None = None,
    M: _ToLinearOperator[_FloatT | _ToInt] | None = None,
    callback: _Callback[_FloatT] | None = None,
    m: int = 20,
    k: int | None = None,
    CU: list[tuple[list[onp.ArrayND[_Float]], list[onp.ArrayND[_Float]] | None]] | None = None,
    discard_C: onp.ToBool = False,
    truncate: _Truncate = "oldest",
) -> tuple[onp.Array1D[_FloatT], int]: ...
@overload
def gcrotmk(
    A: _ToLinearOperator[_ComplexT],
    b: onp.ToComplex1D,
    x0: onp.ToComplex1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    atol: onp.ToFloat = 0.0,
    maxiter: int | None = None,
    M: _ToLinearOperator[_ComplexT] | None = None,
    callback: _Callback[_ComplexT] | None = None,
    m: int = 20,
    k: int | None = None,
    CU: list[tuple[list[onp.ArrayND[_Float | _Complex]], list[onp.ArrayND[_Float | _Complex]] | None]] | None = None,
    discard_C: onp.ToBool = False,
    truncate: _Truncate = "oldest",
) -> tuple[onp.Array1D[_ComplexT], int]: ...
