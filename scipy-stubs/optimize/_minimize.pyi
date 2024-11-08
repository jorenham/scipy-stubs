from collections.abc import Callable, Mapping, Sequence
from typing import Any, Concatenate, Final, Literal, Protocol, TypeAlias, TypedDict, overload, type_check_only

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import AnyReal
from scipy.sparse.linalg import LinearOperator
from ._hessian_update_strategy import HessianUpdateStrategy
from ._typing import Bound, Bounds, Constraint, Constraints, MethodMimimize, MethodMinimizeScalar
from .optimize import OptimizeResult as _OptimizeResult

__all__ = ["minimize", "minimize_scalar"]

_Array_f8_1d: TypeAlias = onpt.Array[tuple[int], np.float64]
_Array_f8_2d: TypeAlias = onpt.Array[tuple[int, int], np.float64]

_FunctionObj: TypeAlias = Callable[Concatenate[_Array_f8_1d, ...], AnyReal]
_FunctionJac: TypeAlias = Callable[Concatenate[_Array_f8_1d, ...], _ArrayLikeFloat_co]
_FunctionObjJac: TypeAlias = Callable[Concatenate[_Array_f8_1d, ...], tuple[AnyReal, _ArrayLikeFloat_co]]
_FunctionHess: TypeAlias = Callable[Concatenate[_Array_f8_1d, ...], _ArrayLikeFloat_co]

_MethodJac: TypeAlias = Literal["2-point", "3-point", "cs"]

@type_check_only
class _CallbackResult(Protocol):
    def __call__(self, /, intermediate_result: OptimizeResult) -> None: ...

@type_check_only
class _CallbackVector(Protocol):
    def __call__(self, /, xk: _Array_f8_1d) -> None: ...

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
    initial_simplex: _ArrayLikeFloat_co
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
    hess_inv0: npt.NDArray[np.floating[Any]]
    # COBYLA
    tol: float
    catool: float
    rhobeg: float
    f_target: float
    # COBYQA
    feasibility_tol: float
    final_tr_radius: float
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
    finite_diff_rel_step: AnyReal | _ArrayLikeFloat_co
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
class _OptimizeResult_scalar(_OptimizeResult):
    x: float | np.float64
    fun: float | np.float64

    success: bool
    message: str
    nit: int
    nfev: int

class OptimizeResult(_OptimizeResult):
    x: _Array_f8_1d
    fun: float | np.float64
    jac: _Array_f8_1d  # requires `jac`
    hess: _Array_f8_2d  # requires `hess` or `hessp`
    hess_inv: _Array_f8_2d | LinearOperator  # requires `hess` or `hessp`, depends on solver

    success: bool
    status: int
    message: str
    nit: int
    nfev: int
    njev: int  # requires `jac`
    nhev: int  # requires `hess` or `hessp`
    maxcv: float  # requires `bounds`

###

MINIMIZE_METHODS: Final[list[MethodMimimize]] = ...
MINIMIZE_METHODS_NEW_CB: Final[list[MethodMimimize]] = ...
MINIMIZE_SCALAR_METHODS: Final[list[MethodMinimizeScalar]] = ...

@overload  # jac: False = ...
def minimize(
    fun: _FunctionObj,
    x0: _ArrayLikeFloat_co,
    args: tuple[object, ...] = (),
    method: MethodMimimize | Callable[..., OptimizeResult] | None = None,
    jac: _FunctionJac | _MethodJac | Literal[False] | None = None,
    hess: _FunctionHess | _MethodJac | HessianUpdateStrategy | None = None,
    hessp: Callable[Concatenate[_Array_f8_1d, _Array_f8_1d, ...], _ArrayLikeFloat_co] | None = None,
    bounds: Bounds | None = None,
    constraints: Constraints = (),
    tol: AnyReal | None = None,
    callback: _CallbackResult | _CallbackVector | None = None,
    options: _MinimizeOptions | None = None,
) -> OptimizeResult: ...
@overload  # jac: True  (positional)
def minimize(
    fun: _FunctionObjJac,
    x0: _ArrayLikeFloat_co,
    args: tuple[object, ...],
    method: MethodMimimize | Callable[..., OptimizeResult] | None,
    jac: Literal[1, True],
    hess: _FunctionHess | _MethodJac | HessianUpdateStrategy | None = None,
    hessp: Callable[Concatenate[_Array_f8_1d, _Array_f8_1d, ...], _ArrayLikeFloat_co] | None = None,
    bounds: Bounds | None = None,
    constraints: Constraints = (),
    tol: AnyReal | None = None,
    callback: _CallbackResult | _CallbackVector | None = None,
    options: _MinimizeOptions | None = None,
) -> OptimizeResult: ...
@overload  # jac: True  (keyword)
def minimize(
    fun: _FunctionObjJac,
    x0: _ArrayLikeFloat_co,
    args: tuple[object, ...] = (),
    method: MethodMimimize | Callable[..., OptimizeResult] | None = None,
    *,
    jac: Literal[1, True],
    hess: _FunctionHess | _MethodJac | HessianUpdateStrategy | None = None,
    hessp: Callable[Concatenate[_Array_f8_1d, _Array_f8_1d, ...], _ArrayLikeFloat_co] | None = None,
    bounds: Bounds | None = None,
    constraints: Constraints = (),
    tol: AnyReal | None = None,
    callback: _CallbackResult | _CallbackVector | None = None,
    options: _MinimizeOptions | None = None,
) -> OptimizeResult: ...

#
def minimize_scalar(
    fun: Callable[Concatenate[float, ...], AnyReal] | Callable[Concatenate[np.float64, ...], AnyReal],
    bracket: Sequence[tuple[AnyReal, AnyReal] | tuple[AnyReal, AnyReal, AnyReal]] | None = None,
    bounds: Bound | None = None,
    args: tuple[object, ...] = (),
    method: MethodMinimizeScalar | Callable[..., _OptimizeResult_scalar] | None = None,
    tol: AnyReal | None = None,
    options: Mapping[str, object] | None = None,  # TODO(jorenham): TypedDict
) -> _OptimizeResult_scalar: ...

# undocumented
def standardize_bounds(bounds: Constraints, x0: _ArrayLikeFloat_co, meth: MethodMimimize) -> Bounds | list[Bound]: ...
def standardize_constraints(constraints: Constraints, x0: _ArrayLikeFloat_co, meth: MethodMimimize) -> list[Constraint]: ...
