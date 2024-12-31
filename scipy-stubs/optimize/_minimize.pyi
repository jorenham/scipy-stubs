from collections.abc import Callable, Mapping, Sequence
from typing import Any, Concatenate, Final, Literal, Protocol, TypeAlias, TypedDict, TypeVar, overload, type_check_only

import numpy as np
import optype.numpy as onp
from scipy._typing import Falsy, Truthy
from scipy.sparse.linalg import LinearOperator
from ._hessian_update_strategy import HessianUpdateStrategy
from ._typing import Bound, Bounds, Constraint, Constraints, MethodMimimize, MethodMinimizeScalar
from .optimize import OptimizeResult as _OptimizeResult

__all__ = ["minimize", "minimize_scalar"]

###

_Args: TypeAlias = tuple[object, ...]

_Float: TypeAlias = float | np.float64
_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]

_RT = TypeVar("_RT")
_Fun0D: TypeAlias = Callable[Concatenate[float, ...], _RT] | Callable[Concatenate[np.float64, ...], _RT]
_Fun1D: TypeAlias = Callable[Concatenate[_Float1D, ...], _RT]
_Fun1Dp: TypeAlias = Callable[Concatenate[_Float1D, _Float1D, ...], _RT]

_FDMethod: TypeAlias = Literal["2-point", "3-point", "cs"]

@type_check_only
class _CallbackResult(Protocol):
    def __call__(self, /, intermediate_result: OptimizeResult) -> None: ...

@type_check_only
class _CallbackVector(Protocol):
    def __call__(self, /, xk: _Float1D) -> None: ...

@type_check_only
class _MinimizeMethodFun(Protocol):
    def __call__(self, fun: _Fun1D[onp.ToFloat], x0: onp.ToFloat1D, /, args: _Args, **kwargs: Any) -> OptimizeResult: ...

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
    offset: float
    stepmx: float
    accuracy: float
    minfev: float
    rescale: float
    # L-BFGS-B, TNC
    maxfun: int
    # L-BFGS-B
    maxcor: int
    iprint: int
    maxls: int
    # Nelder-Mead
    initial_simplex: onp.ToArrayND
    adaptive: bool
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
    hess_inv0: onp.ArrayND[np.floating[Any]]
    # COBYLA
    tol: float
    catool: float
    rhobeg: float
    f_target: float
    # COBYQA
    feasibility_tol: float
    final_tr_radius: float
    # Powell
    direc: onp.ArrayND[np.floating[Any]]
    # CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP
    eps: float | onp.ArrayND[np.floating[Any]]
    # CG, BFGS, Newton-CG
    c1: float
    c2: float
    # CG, BFGS
    norm: float
    # CG, BFGS, L-BFGS-B, TNC, SLSQP, trust-constr
    finite_diff_rel_step: onp.ToFloat | onp.ToArrayND
    # dogleg, trust-ncg, trust-exact
    initial_trust_radius: float
    max_trust_radius: float
    # COBYQA, trust-constr
    initial_tr_radius: float
    # trust-constr
    barrier_tol: float
    sparse_jacobian: bool
    initial_constr_penalty: float
    initial_barrier_parameter: float
    initial_barrier_tolerance: float
    factorization_method: Literal["NormalEquation", "AugmentedSystem", "QRFactorization", "SVDFactorization"]
    verbose: Literal[0, 1, 2, 3]
    # dogleg, trust-ncg, trust-exact, TNC
    eta: float
    # trust-krylov
    inexact: bool
    # TNC (list of floats), COBYQA (bool)
    scale: Sequence[float] | bool

@type_check_only
class _MinimizeScalarResult(_OptimizeResult):
    x: _Float
    fun: _Float
    success: bool
    message: str
    nit: int
    nfev: int

###

MINIMIZE_METHODS: Final[list[MethodMimimize]] = ...
MINIMIZE_METHODS_NEW_CB: Final[list[MethodMimimize]] = ...
MINIMIZE_SCALAR_METHODS: Final[list[MethodMinimizeScalar]] = ...

# NOTE: This `OptimizeResult` "flavor" is Specific to `minimize`
class OptimizeResult(_OptimizeResult):
    success: bool
    status: int
    message: str
    x: _Float1D
    nit: int
    maxcv: float  # requires `bounds`
    fun: _Float
    nfev: int
    jac: _Float1D  # requires `jac`
    njev: int  # requires `jac`
    hess: _Float2D  # requires `hess` or `hessp`
    hess_inv: _Float2D | LinearOperator  # requires `hess` or `hessp`, depends on solver
    nhev: int  # requires `hess` or `hessp`

@overload  # `fun` return scalar, `jac` not truthy
def minimize(
    fun: _Fun1D[onp.ToFloat],
    x0: onp.ToFloat1D,
    args: _Args = (),
    method: MethodMimimize | _MinimizeMethodFun | None = None,
    jac: _Fun1D[onp.ToFloat1D] | _FDMethod | Falsy | None = None,
    hess: _Fun1D[onp.ToFloat2D] | _FDMethod | HessianUpdateStrategy | None = None,
    hessp: _Fun1Dp[onp.ToFloat1D] | None = None,
    bounds: Bounds | None = None,
    constraints: Constraints = (),
    tol: onp.ToFloat | None = None,
    callback: _CallbackResult | _CallbackVector | None = None,
    options: _MinimizeOptions | None = None,
) -> OptimizeResult: ...
@overload  # fun` return (scalar, vector), `jac` truthy  (positional)
def minimize(
    fun: _Fun1D[tuple[onp.ToFloat, onp.ToFloat1D]],
    x0: onp.ToFloat1D,
    args: _Args,
    method: MethodMimimize | _MinimizeMethodFun | None,
    jac: Truthy,
    hess: _Fun1D[onp.ToFloat2D] | _FDMethod | HessianUpdateStrategy | None = None,
    hessp: _Fun1Dp[onp.ToFloat1D] | None = None,
    bounds: Bounds | None = None,
    constraints: Constraints = (),
    tol: onp.ToFloat | None = None,
    callback: _CallbackResult | _CallbackVector | None = None,
    options: _MinimizeOptions | None = None,
) -> OptimizeResult: ...
@overload  # fun` return (scalar, vector), `jac` truthy  (keyword)
def minimize(
    fun: _Fun1D[tuple[onp.ToFloat, onp.ToFloat1D]],
    x0: onp.ToFloat1D,
    args: _Args = (),
    method: _MinimizeMethodFun | MethodMimimize | None = None,
    *,
    jac: Truthy,
    hess: _Fun1D[onp.ToFloat2D] | _FDMethod | HessianUpdateStrategy | None = None,
    hessp: _Fun1Dp[onp.ToFloat1D] | None = None,
    bounds: Bounds | None = None,
    constraints: Constraints = (),
    tol: onp.ToFloat | None = None,
    callback: _CallbackResult | _CallbackVector | None = None,
    options: _MinimizeOptions | None = None,
) -> OptimizeResult: ...

#
def minimize_scalar(
    fun: _Fun0D[onp.ToFloat],
    bracket: Sequence[tuple[onp.ToFloat, onp.ToFloat] | tuple[onp.ToFloat, onp.ToFloat, onp.ToFloat]] | None = None,
    bounds: Bound | None = None,
    args: _Args = (),
    method: MethodMinimizeScalar | Callable[..., _MinimizeScalarResult] | None = None,
    tol: onp.ToFloat | None = None,
    options: Mapping[str, object] | None = None,  # TODO(jorenham): TypedDict
) -> _MinimizeScalarResult: ...

# undocumented
def standardize_bounds(bounds: Bounds, x0: onp.ToFloat1D, meth: MethodMimimize) -> Bounds | list[Bound]: ...
def standardize_constraints(constraints: Constraints, x0: onp.ToFloat1D, meth: MethodMimimize) -> list[Constraint]: ...
