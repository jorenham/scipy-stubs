from typing import Any, TypeVar, overload

import numpy as np
import optype as op
import optype.numpy as onp
from scipy.sparse._typing import Numeric
from scipy.sparse.linalg._interface import LinearOperator

__all__ = ["expm_multiply"]

_SCT = TypeVar("_SCT", bound=Numeric)

###

@overload  # 1-d
def expm_multiply(
    A: LinearOperator[_SCT],
    B: onp.Array1D[_SCT | np.integer[Any] | np.float16 | np.bool_],
    start: onp.ToFloat | None = None,
    stop: onp.ToFloat | None = None,
    num: op.CanIndex | None = None,
    endpoint: bool | None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.Array1D[_SCT]: ...
@overload  # 2-d
def expm_multiply(
    A: LinearOperator[_SCT],
    B: onp.Array2D[_SCT | np.integer[Any] | np.float16 | np.bool_],
    start: onp.ToFloat | None = None,
    stop: onp.ToFloat | None = None,
    num: op.CanIndex | None = None,
    endpoint: bool | None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.Array2D[_SCT]: ...
@overload  # 1-d or 2-d
def expm_multiply(
    A: LinearOperator[_SCT],
    B: onp.ArrayND[_SCT | np.float16 | np.integer[Any] | np.bool_],
    start: onp.ToFloat | None = None,
    stop: onp.ToFloat | None = None,
    num: op.CanIndex | None = None,
    endpoint: bool | None = None,
    traceA: onp.ToComplex | None = None,
) -> onp.Array1D[_SCT] | onp.Array2D[_SCT]: ...
