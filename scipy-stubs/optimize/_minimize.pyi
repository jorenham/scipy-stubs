from collections.abc import Callable, Mapping, Sequence
from typing import Any, Concatenate, Final, Literal, Protocol, TypeAlias, TypedDict, overload, type_check_only

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLike
from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator
from ._constraints import Bounds
from ._hessian_update_strategy import HessianUpdateStrategy
from ._optimize import OptimizeResult
from ._typing import Constraint, MethodJac, MethodMimimize, MethodMinimizeScalar

__all__ = ["minimize", "minimize_scalar"]

_RealScalar: TypeAlias = np.floating[Any] | np.integer[Any] | np.bool_
_RealVector: TypeAlias = onpt.Array[tuple[int], np.floating[Any]]

_RealScalarLike: TypeAlias = float | _RealScalar
_RealVectorLike: TypeAlias = _ArrayLike[_RealScalar] | Sequence[_RealScalarLike]
_RealMatrixLike: TypeAlias = _ArrayLike[_RealScalar] | Sequence[Sequence[_RealScalarLike]] | spmatrix | sparray | LinearOperator

_Bound: TypeAlias = tuple[_RealScalarLike | None, _RealScalarLike | None]

_FunctionObj: TypeAlias = Callable[Concatenate[_RealVector, ...], _RealScalarLike]
_FunctionJac: TypeAlias = Callable[Concatenate[_RealVector, ...], _RealVectorLike]
_FunctionObjJac: TypeAlias = Callable[Concatenate[_RealVector, ...], tuple[_RealScalarLike, _RealVectorLike]]
_FunctionHess: TypeAlias = Callable[Concatenate[_RealVector, ...], _RealMatrixLike]

@type_check_only
class _CallbackResult(Protocol):
    def __call__(self, /, intermediate_result: OptimizeResult) -> None: ...

@type_check_only
class _CallbackVector(Protocol):
    def __call__(self, /, xk: _RealVector) -> None: ...

@type_check_only
class _MinimizeOptions(TypedDict, total=False):
    # Nelder-Mead, Powell, CG, BFGS, Newton-CG
    return_all: bool
    # Nelder-Mead, Powell, CG, BFGS, L-BFGS-B, Newton-CG, TNC, COBYLA, COBYQA, SLSQP, trust-constr
    disp: bool
    # Nelder-Mead, Powell, CG, BFGS, L-BFGS-B, Newton-CG, COBYLA, SLSQP, trust-constr
    maxiter: int
    # Nelder-Mead, Powell, COBYQA
    maxfev: int
    # TNC
    maxCGit: int
    # L-BFGS-B, TNC
    maxfun: int
    # L-BFGS-B
    maxcor: int
    iprint: int
    maxls: int
    # Nelder-Mead
    initial_simplex: _RealMatrixLike
    adaptive: bool
    # COBYLA
    tol: float
    # Nelder-Mead
    xatol: float
    fatol: float
    # CG, BFGS, L-BFGS-B, dogleg, trust-ncg, trust-exact, TNC, trust-constr
    gtol: float
    # Powell, Newton-CG, TNC, trust-constr
    xtol: float
    # Powell, L-BFGS-B, TNC, SLSQP
    ftol: float
    # BFGS
    xrtol: float
    # COBYLA
    catool: float
    # COBYQA
    feasibility_tol: float
    # trust-constr
    barrier_tol: float
    # BFGS
    hess_inv0: npt.NDArray[np.floating[Any]]
    # Powell
    direc: npt.NDArray[np.floating[Any]]
    # CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP
    eps: float | npt.NDArray[np.floating[Any]]
    # CG, BFGS, Newton-CG
    c1: float
    c2: float
    # CG, BFGS
    norm: float
    # CG, BFGS, L-BFGS-B, TNC, SLSQP, trust-constr
    finite_diff_rel_step: _RealScalarLike | _RealVectorLike
    # dogleg, trust-ncg, trust-exact
    initial_trust_radius: float
    max_trust_radius: float
    # COBYQA, trust-constr
    initial_tr_radius: float
    # COBYQA
    final_tr_radius: float
    # trust-constr
    sparse_jacobian: bool
    initial_constr_penalty: float
    initial_barrier_parameter: float
    initial_barrier_tolerance: float
    factorization_method: Literal["NormalEquation", "AugmentedSystem", "QRFactorization", "SVDFactorization"]
    # dogleg, trust-ncg, trust-exact, TNC
    eta: float
    # trust-krylov
    inexact: bool
    # TNC (list of floats), COBYQA (bool)
    scale: Sequence[float] | bool
    # TNC
    offset: float
    stepmx: float
    accuracy: float
    minfev: float
    rescale: float
    # COBYLA
    rhobeg: float
    f_target: float
    # trust-constr
    verbose: Literal[0, 1, 2, 3]

###

MINIMIZE_METHODS: Final[Sequence[MethodMimimize]] = ...
MINIMIZE_METHODS_NEW_CB: Final[Sequence[MethodMimimize]] = ...
MINIMIZE_SCALAR_METHODS: Final[Sequence[MethodMinimizeScalar]] = ...

@overload
def minimize(
    fun: _FunctionObj,
    x0: _RealVectorLike,
    args: tuple[object, ...] = (),
    method: MethodMimimize | Callable[..., OptimizeResult] | None = None,
    jac: _FunctionJac | MethodJac | Literal[False] | None = None,
    hess: _FunctionHess | MethodJac | HessianUpdateStrategy | None = None,
    hessp: Callable[Concatenate[_RealVector, _RealVector, ...], _RealVectorLike] | None = None,
    bounds: Bounds | Sequence[_Bound] | None = None,
    constraints: Constraint | Sequence[Constraint] | tuple[()] = (),
    tol: float | None = None,
    callback: _CallbackResult | _CallbackVector | None = None,
    options: _MinimizeOptions | None = None,
) -> OptimizeResult: ...
@overload
def minimize(
    fun: _FunctionObjJac,
    x0: _RealVectorLike,
    args: tuple[object, ...],
    method: MethodMimimize | Callable[..., OptimizeResult] | None,
    jac: Literal[True],
    hess: _FunctionHess | MethodJac | HessianUpdateStrategy | None = None,
    hessp: Callable[Concatenate[_RealVector, _RealVector, ...], _RealVectorLike] | None = None,
    bounds: Bounds | Sequence[_Bound] | None = None,
    constraints: Constraint | Sequence[Constraint] = (),
    tol: float | None = None,
    callback: _CallbackResult | _CallbackVector | None = None,
    options: _MinimizeOptions | None = None,
) -> OptimizeResult: ...
@overload
def minimize(
    fun: _FunctionObjJac,
    x0: _RealVectorLike,
    args: tuple[object, ...] = (),
    method: MethodMimimize | Callable[..., OptimizeResult] | None = None,
    *,
    jac: Literal[True],
    hess: _FunctionHess | MethodJac | HessianUpdateStrategy | None = None,
    hessp: Callable[Concatenate[_RealVector, _RealVector, ...], _RealVectorLike] | None = None,
    bounds: Bounds | Sequence[_Bound] | None = None,
    constraints: Constraint | Sequence[Constraint] | tuple[()] = (),
    tol: float | None = None,
    callback: _CallbackResult | _CallbackVector | None = None,
    options: _MinimizeOptions | None = None,
) -> OptimizeResult: ...

#
def minimize_scalar(
    fun: Callable[Concatenate[float, ...], float | np.floating[Any]],
    bracket: Sequence[tuple[float, float] | tuple[float, float, float]] | None = None,
    bounds: _Bound | None = None,
    args: tuple[object, ...] = (),
    method: MethodMinimizeScalar | Callable[..., OptimizeResult] | None = None,
    tol: float | None = None,
    options: Mapping[str, object] | None = None,
) -> OptimizeResult: ...

#
def standardize_bounds(  # undocumented
    bounds: Sequence[_Bound] | Bounds,
    x0: _RealVectorLike,
    meth: MethodMimimize,
) -> Bounds | list[_Bound]: ...

#
def standardize_constraints(  # undocumented
    constraints: Constraint | Sequence[Constraint],
    x0: _RealVectorLike,
    meth: object,  # unused
) -> list[Constraint]: ...
