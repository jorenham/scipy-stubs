from collections.abc import Callable, Mapping
from typing import Literal, TypeAlias
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onpt
from scipy.optimize import OptimizeResult
from scipy.optimize._typing import SolverLSQ
from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...], default=tuple[int, ...])

_ValueFloat: TypeAlias = float | np.float64

_ArrayFloat: TypeAlias = onpt.Array[_ShapeT, np.float64]
_MatrixFloat: TypeAlias = onpt.Array[_ShapeT, np.float64] | sparray | spmatrix | LinearOperator

_FunObj: TypeAlias = Callable[[_ArrayFloat[tuple[int]], _ValueFloat], _MatrixFloat]
_FunJac: TypeAlias = Callable[[_ArrayFloat[tuple[int]], _ValueFloat], _MatrixFloat]
_FunLoss: TypeAlias = Callable[[_ValueFloat], _ValueFloat]

# TODO: custom `OptimizeResult``

def trf(
    fun: _FunObj,
    jac: _FunJac,
    x0: _ArrayFloat,
    f0: _ValueFloat,
    J0: _MatrixFloat,
    lb: _ArrayFloat,
    ub: _ArrayFloat,
    ftol: _ValueFloat,
    xtol: _ValueFloat,
    gtol: _ValueFloat,
    max_nfev: int,
    x_scale: Literal["jac"] | _ValueFloat | _ArrayFloat,
    loss_function: _FunLoss,
    tr_solver: SolverLSQ,
    tr_options: Mapping[str, object],
    verbose: bool,
) -> OptimizeResult: ...
def trf_bounds(
    fun: _FunObj,
    jac: _FunJac,
    x0: _ArrayFloat,
    f0: _ValueFloat,
    J0: _MatrixFloat,
    lb: _ArrayFloat,
    ub: _ArrayFloat,
    ftol: _ValueFloat,
    xtol: _ValueFloat,
    gtol: _ValueFloat,
    max_nfev: int,
    x_scale: Literal["jac"] | _ValueFloat | _ArrayFloat,
    loss_function: _FunLoss,
    tr_solver: SolverLSQ,
    tr_options: Mapping[str, object],
    verbose: bool,
) -> OptimizeResult: ...
def trf_no_bounds(
    fun: _FunObj,
    jac: _FunJac,
    x0: _ArrayFloat,
    f0: _ValueFloat,
    J0: _MatrixFloat,
    ftol: _ValueFloat,
    xtol: _ValueFloat,
    gtol: _ValueFloat,
    max_nfev: int,
    x_scale: Literal["jac"] | _ValueFloat | _ArrayFloat,
    loss_function: _FunLoss,
    tr_solver: SolverLSQ,
    tr_options: Mapping[str, object],
    verbose: bool,
) -> OptimizeResult: ...
def select_step(
    x: _ArrayFloat,
    J_h: _MatrixFloat,
    diag_h: _ArrayFloat,
    g_h: _ArrayFloat,
    p: _ArrayFloat,
    p_h: _ArrayFloat,
    d: _ArrayFloat,
    Delta: _ValueFloat,
    lb: _ArrayFloat,
    ub: _ArrayFloat,
    theta: _ValueFloat,
) -> tuple[_ArrayFloat, _ArrayFloat, _ValueFloat]: ...
