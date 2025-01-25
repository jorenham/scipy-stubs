from collections.abc import Callable
from typing import Any, TypeAlias
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy.sparse._base import _spbase
from scipy.sparse._typing import Numeric
from scipy.sparse.linalg import LinearOperator

__all__ = ["minres"]

_FloatT = TypeVar("_FloatT", bound=np.float32 | np.float64, default=np.float64)
_ScalarT = TypeVar("_ScalarT", bound=Numeric)

_Ignored: TypeAlias = object
_ToInt: TypeAlias = np.integer[Any] | np.bool_
_ToLinearOperator: TypeAlias = onp.CanArrayND[_ScalarT] | _spbase[_ScalarT] | LinearOperator[_ScalarT]

###

def minres(
    A: _ToLinearOperator[_FloatT | _ToInt],
    b: onp.ToFloat1D,
    x0: onp.ToFloat1D | None = None,
    *,
    rtol: onp.ToFloat = 1e-5,
    shift: onp.ToFloat = 0.0,
    maxiter: int | None = None,
    M: _ToLinearOperator[_FloatT | _ToInt] | None = None,
    callback: Callable[[onp.Array1D[_FloatT]], _Ignored] | None = None,
    show: onp.ToBool = False,
    check: onp.ToBool = False,
) -> tuple[onp.Array1D[_FloatT], int]: ...
