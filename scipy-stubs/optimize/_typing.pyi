from collections.abc import Sequence
from typing import Any, Final, Generic, Literal, Protocol, TypeAlias, type_check_only
from typing_extensions import NotRequired, TypedDict, TypeVar, TypeVarTuple, Unpack

import numpy as np
import optype.numpy as onpt
from ._constraints import Bounds as _Bounds, LinearConstraint, NonlinearConstraint
from ._optimize import OptimizeResult as _OptimizeResult

_Ts = TypeVarTuple("_Ts")
_Ts0 = TypeVarTuple("_Ts0", default=Unpack[tuple[()]])
_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)

_SCT_f = TypeVar("_SCT_f", bound=np.floating[Any], default=np.float32 | np.float64)
_ScalarLike_f: TypeAlias = float | np.floating[Any]
_ScalarLike_f_co: TypeAlias = float | np.floating[Any] | np.integer[Any]

# any objective function
@type_check_only
class ObjectiveFunc(Protocol[_T_contra, Unpack[_Ts], _T_co]):
    def __call__(self, x: _T_contra, /, *args: Unpack[_Ts]) -> _T_co: ...

# bounds
Bound: TypeAlias = tuple[_ScalarLike_f_co | None, _ScalarLike_f_co | None]
Bounds: TypeAlias = Sequence[Bound] | _Bounds

# constaints
@type_check_only
class ConstraintDict(TypedDict, Generic[Unpack[_Ts0]]):
    type: Literal["eq", "ineq"]
    fun: ObjectiveFunc[onpt.Array[tuple[int], np.floating[Any]], Unpack[_Ts0], _ScalarLike_f_co]
    jac: NotRequired[
        ObjectiveFunc[
            onpt.Array[tuple[int], np.floating[Any]],
            Unpack[_Ts0],
            Sequence[_ScalarLike_f] | onpt.CanArray[tuple[int, ...], np.dtype[np.floating[Any]]],
        ]
    ]
    args: NotRequired[tuple[Unpack[_Ts0]]]

Constraint: TypeAlias = LinearConstraint | NonlinearConstraint | ConstraintDict
Constraints: TypeAlias = Constraint | Sequence[Constraint]

Brack: TypeAlias = tuple[_ScalarLike_f_co, _ScalarLike_f_co] | tuple[_ScalarLike_f_co, _ScalarLike_f_co, _ScalarLike_f_co]

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

@type_check_only
class OptimizeResultMinimize(_OptimizeResult[_SCT_f], Generic[_SCT_f]):
    jac: onpt.Array[tuple[int], _SCT_f]
    nfev: Final[int]
    njev: Final[int]
    nhev: Final[int]

@type_check_only
class OptimizeResultMinimizeHess(OptimizeResultMinimize[_SCT_f], Generic[_SCT_f]):
    hess: onpt.Array[tuple[int, int], _SCT_f]
    hess_inv: onpt.Array[tuple[int, int], _SCT_f]

@type_check_only
class OptimizeResultMinimizeConstr(OptimizeResultMinimize[_SCT_f], Generic[_SCT_f]):
    maxcv: Final[float]

@type_check_only
class OptimizeResultMinimizeHessConstr(OptimizeResultMinimize[_SCT_f], Generic[_SCT_f]):
    hess: onpt.Array[tuple[int, int], _SCT_f]
    hess_inv: onpt.Array[tuple[int, int], _SCT_f]
    maxcv: Final[float]

@type_check_only
class OptimizeResultLinprog(Generic[_SCT_f]):
    slack: onpt.Array[tuple[int], _SCT_f]
    con: onpt.Array[tuple[int], _SCT_f]
