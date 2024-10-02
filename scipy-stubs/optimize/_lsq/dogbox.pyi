from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeAlias
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onpt
from scipy.optimize import OptimizeResult
from scipy.optimize._typing import SolverLSQ
from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator

# TODO: custom `OptimizeResult``

_SCT_i = TypeVar("_SCT_i", bound=np.integer[Any], default=np.int_)
_SCT_f = TypeVar("_SCT_f", bound=np.floating[Any], default=np.float64)

_N_x = TypeVar("_N_x", bound=int, default=int)
_N_f = TypeVar("_N_f", bound=int, default=int)

_ValueFloat: TypeAlias = float | _SCT_f

_VectorBool: TypeAlias = onpt.Array[tuple[_N_x], np.bool_]
_VectorInt: TypeAlias = onpt.Array[tuple[_N_x], _SCT_i]
_VectorFloat: TypeAlias = onpt.Array[tuple[_N_x], _SCT_f]
_MatrixFloat: TypeAlias = onpt.Array[tuple[_N_x, _N_f], _SCT_f] | sparray | spmatrix | LinearOperator

_FunResid: TypeAlias = Callable[[_VectorFloat[_N_x]], _VectorFloat[_N_f]]
# this type-alias is a workaround to get the correct oder of type params
_FunJac: TypeAlias = Callable[[_VectorFloat[_N_x], _VectorFloat[_N_f]], _MatrixFloat[_N_f, _N_x]]
_FunLoss: TypeAlias = Callable[[_VectorFloat[_N_x]], _ValueFloat]

def lsmr_operator(
    Jop: LinearOperator,
    d: _VectorFloat[_N_x, _SCT_f],
    active_set: _VectorBool[_N_x],
) -> LinearOperator: ...
def find_intersection(
    x: _VectorFloat[_N_x],
    tr_bounds: _VectorFloat[_N_x],
    lb: _VectorFloat[_N_x],
    ub: _VectorFloat[_N_x],
) -> tuple[
    _VectorFloat[_N_x],
    _VectorFloat[_N_x],
    _VectorBool[_N_x],
    _VectorBool[_N_x],
    _VectorBool[_N_x],
    _VectorBool[_N_x],
]: ...
def dogleg_step(
    x: _VectorFloat[_N_x],
    newton_step: _VectorFloat[_N_x],
    g: _VectorFloat[_N_x],
    a: _ValueFloat,
    b: _ValueFloat,
    tr_bounds: _VectorFloat[_N_x],
    lb: _VectorFloat[_N_x],
    ub: _VectorFloat[_N_x],
) -> tuple[_VectorFloat[_N_x], _VectorInt[_N_x], np.bool_]: ...
def dogbox(
    fun: _FunResid[_N_x, _N_f],
    jac: _FunJac[_N_x, _N_f],
    x0: _VectorFloat[_N_x],
    f0: _VectorFloat[_N_f],
    J0: _MatrixFloat[_N_f, _N_x],
    lb: _VectorFloat[_N_x],
    ub: _VectorFloat[_N_x],
    ftol: _ValueFloat,
    xtol: _ValueFloat,
    gtol: _ValueFloat,
    max_nfev: int,
    x_scale: Literal["jac"] | _ValueFloat | _VectorFloat[_N_f],
    loss_function: _FunLoss[_N_x],
    tr_solver: SolverLSQ,
    tr_options: Mapping[str, object],
    verbose: bool,
) -> OptimizeResult: ...
