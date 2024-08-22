from .framework import TrustRegion as TrustRegion
from .problem import (
    BoundConstraints as BoundConstraints,
    LinearConstraints as LinearConstraints,
    NonlinearConstraints as NonlinearConstraints,
    ObjectiveFunction as ObjectiveFunction,
    Problem as Problem,
)
from .settings import (
    DEFAULT_CONSTANTS as DEFAULT_CONSTANTS,
    DEFAULT_OPTIONS as DEFAULT_OPTIONS,
    PRINT_OPTIONS as PRINT_OPTIONS,
    Constants as Constants,
    ExitStatus as ExitStatus,
    Options as Options,
)
from .utils import (
    CallbackSuccess as CallbackSuccess,
    FeasibleSuccess as FeasibleSuccess,
    MaxEvalError as MaxEvalError,
    TargetSuccess as TargetSuccess,
    exact_1d_array as exact_1d_array,
)
from scipy._typing import Untyped
from scipy.optimize import (
    Bounds as Bounds,
    LinearConstraint as LinearConstraint,
    NonlinearConstraint as NonlinearConstraint,
    OptimizeResult as OptimizeResult,
)

def minimize(
    fun,
    x0,
    args=(),
    bounds: Untyped | None = None,
    constraints=(),
    callback: Untyped | None = None,
    options: Untyped | None = None,
    **kwargs,
) -> Untyped: ...
