from collections.abc import Callable, Sequence
from typing import Concatenate, Literal, TypeAlias, type_check_only
from typing_extensions import NotRequired, TypedDict

import numpy as np
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import AnyReal
from scipy.sparse.linalg import LinearOperator
from ._constraints import Bounds as _Bounds, LinearConstraint, NonlinearConstraint
from ._optimize import OptimizeResult as _OptimizeResult

_Array_1d_f8: TypeAlias = onpt.Array[tuple[int], np.float64]
_Array_2d_f8: TypeAlias = onpt.Array[tuple[int, int], np.float64]

# bounds
Bound: TypeAlias = tuple[AnyReal | None, AnyReal | None]
Bounds: TypeAlias = Sequence[Bound] | _Bounds

# constaints
@type_check_only
class ConstraintDict(TypedDict):
    type: Literal["eq", "ineq"]
    fun: Callable[Concatenate[_Array_1d_f8, ...], AnyReal]
    jac: NotRequired[Callable[Concatenate[_Array_1d_f8, ...], _ArrayLikeFloat_co]]
    args: NotRequired[tuple[object, ...]]

Constraint: TypeAlias = LinearConstraint | NonlinearConstraint | ConstraintDict
Constraints: TypeAlias = Constraint | Sequence[Constraint]

Brack: TypeAlias = tuple[AnyReal, AnyReal] | tuple[AnyReal, AnyReal, AnyReal]

Solver: TypeAlias = Literal["minimize", "minimize_scalar", "root", "root_salar", "linprog", "quadratic_assignment"]

MethodJac: TypeAlias = Literal["2-point", "3-point", "cs"]
MethodMimimize: TypeAlias = Literal[
    "NELDER-MEAD", "Nelder-Mead", "nelder-mead",
    "POWELL", "Powell", "powell",
    "CG", "Cg", "cg",
    "BFGS", "Bfgs", "bfgs",
    "NEWTON-CG", "Newton-CG", "newton-cg",
    "L-BFGS-B", "L-Bfgs-B", "l-bfgs-b",
    "TNC", "Tnc", "tnc",
    "COBYLA", "Cobyla", "cobyla",
    "COBYQA", "Cobyqa", "cobyqa",
    "SLSQP", "Slsqp", "slsqp",
    "TRUST-CONSTR", "Trust-Constr", "trust-constr",
    "DOGLEG", "Dogleg", "dogleg",
    "TRUST-NCG", "Trust-NCG", "trust-ncg",
    "TRUST-EXACT", "Trust-Exact", "trust-exact",
    "TRUST-KRYLOV", "Trust-Krylov", "trust-krylov",
]  # fmt: skip
MethodRoot: TypeAlias = Literal[
    "hybr",
    "lm",
    "broyden1",
    "broyden2",
    "anderson",
    "linearmixing",
    "diagbroyden",
    "excitingmixing",
    "krylov",
    "df-sane",
]
MethodMinimizeScalar: TypeAlias = Literal["brent", "golden", "bounded"]
MethodRootScalar: TypeAlias = Literal["bisect", "brentq", "brenth", "ridder", "toms748", "newton", "secant", "halley"]
MethodLinprog: TypeAlias = Literal["highs", "highs-ds", "highs-ipm"]  # Literal["interior-point", "revised simplex", "simplex"]
MethodQuadraticAssignment: TypeAlias = Literal["faq", "2opt"]
MethodAll: TypeAlias = Literal[
    MethodMimimize,
    MethodRoot,
    MethodMinimizeScalar,
    MethodRootScalar,
    MethodLinprog,
    MethodQuadraticAssignment,
]

SolverLSQ: TypeAlias = Literal["exact", "lsmr", None]
MethodLSQ: TypeAlias = Literal["trf", "dogbox", "lm"]

@type_check_only
class OptimizeResult_minimize_scalar(_OptimizeResult):
    x: float | np.float64
    fun: float | np.float64

    success: bool
    message: str
    nit: int
    nfev: int

@type_check_only
class OptimizeResult_minimize(_OptimizeResult):
    x: _Array_1d_f8
    fun: float | np.float64
    jac: _Array_1d_f8  # requires `jac`
    hess: _Array_2d_f8  # requires `hess` or `hessp`
    hess_inv: _Array_2d_f8 | LinearOperator  # requires `hess` or `hessp`, depends on solver

    success: bool
    status: int
    message: str
    nit: int
    nfev: int
    njev: int  # requires `jac`
    nhev: int  # requires `hess` or `hessp`
    maxcv: float  # requires `bounds`

@type_check_only
class OptimizeResult_linprog(OptimizeResult_minimize):
    slack: _Array_1d_f8
    con: _Array_1d_f8

@type_check_only
class OptimizeResult_lsq(OptimizeResult_minimize):
    cost: float | np.float64
    optimality: float | np.float64
    grad: _Array_1d_f8
    active_mask: onpt.Array[tuple[int], np.bool_]
