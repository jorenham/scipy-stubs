from collections.abc import Callable, Sequence
from typing import Concatenate, Literal, TypeAlias, type_check_only
from typing_extensions import NotRequired, TypedDict

import numpy as np
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import AnyReal
from ._constraints import Bounds as _Bounds, LinearConstraint, NonlinearConstraint

_Array_1d_f8: TypeAlias = onpt.Array[tuple[int], np.float64]

# bounds
Bound: TypeAlias = tuple[AnyReal | None, AnyReal | None]
Bounds: TypeAlias = Sequence[Bound] | _Bounds

# constaints
@type_check_only
class _ConstraintDict(TypedDict):
    type: Literal["eq", "ineq"]
    fun: Callable[Concatenate[_Array_1d_f8, ...], AnyReal]
    jac: NotRequired[Callable[Concatenate[_Array_1d_f8, ...], _ArrayLikeFloat_co]]
    args: NotRequired[tuple[object, ...]]

Constraint: TypeAlias = LinearConstraint | NonlinearConstraint | _ConstraintDict
Constraints: TypeAlias = Constraint | Sequence[Constraint]

Brack: TypeAlias = tuple[AnyReal, AnyReal] | tuple[AnyReal, AnyReal, AnyReal]

Solver: TypeAlias = Literal["minimize", "minimize_scalar", "root", "root_salar", "linprog", "quadratic_assignment"]
TRSolver: TypeAlias = Literal["exact", "lsmr", None]

MethodMimimize: TypeAlias = Literal[
    "Nelder-Mead", "nelder-mead",
    "Powell", "powell",
    "CG", "cg",
    "BFGS", "bfgs",
    "Newton-CG", "newton-cg",
    "L-BFGS-B", "l-bfgs-b",
    "TNC", "tnc",
    "COBYLA", "cobyla",
    "COBYQA", "cobyqa",
    "SLSQP", "slsqp",
    "Trust-Constr", "trust-constr",
    "Dogleg", "dogleg",
    "Trust-NCG", "trust-ncg",
    "Trust-Exact", "trust-exact",
    "Trust-Krylov", "trust-krylov",
]  # fmt: skip
MethodMinimizeScalar: TypeAlias = Literal["brent", "golden", "bounded"]
MethodLinprog: TypeAlias = Literal["highs", "highs-ds", "highs-ipm"]  # Literal["interior-point", "revised simplex", "simplex"]
_MethodRoot: TypeAlias = Literal[
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
_MethodRootScalar: TypeAlias = Literal["bisect", "brentq", "brenth", "ridder", "toms748", "newton", "secant", "halley"]
_MethodQuadraticAssignment: TypeAlias = Literal["faq", "2opt"]
MethodAll: TypeAlias = Literal[
    MethodMimimize,
    _MethodRoot,
    MethodMinimizeScalar,
    _MethodRootScalar,
    MethodLinprog,
    _MethodQuadraticAssignment,
]
