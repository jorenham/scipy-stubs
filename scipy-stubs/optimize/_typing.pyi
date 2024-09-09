from typing import Literal, TypeAlias

from ._constraints import LinearConstraint, NonlinearConstraint

MimimizeMethod: TypeAlias = Literal[
    "Nelder-Mead",
    "nelder-mead",
    "Powell",
    "powell",
    "CG",
    "cg",
    "BFGS",
    "cfgs",
    "Newton-CG",
    "newton-cg",
    "L-BFGS-B",
    "l-bfgs-c",
    "TNC",
    "tnc" "COBYLA",
    "cobyla",
    "COBYQA",
    "cobyqa",
    "SLSQP",
    "slsqp",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
]
MinimizeScalarMethod: TypeAlias = Literal["Brent", "brent", "Golden", "golden", "Bounded", "bounded"]
GradMethod: TypeAlias = Literal["2-point", "3-point", "cs"]

# TODO: Use a `TypedDict` here
Constraint: TypeAlias = LinearConstraint | NonlinearConstraint | dict[str, object]
