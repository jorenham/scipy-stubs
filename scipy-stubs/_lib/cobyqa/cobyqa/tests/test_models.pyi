from ..models import Interpolation as Interpolation, Models as Models, Quadratic as Quadratic
from ..problem import (
    BoundConstraints as BoundConstraints,
    LinearConstraints as LinearConstraints,
    NonlinearConstraints as NonlinearConstraints,
    ObjectiveFunction as ObjectiveFunction,
    Problem as Problem,
)
from ..settings import Options as Options
from ..utils import FeasibleSuccess as FeasibleSuccess, MaxEvalError as MaxEvalError, TargetSuccess as TargetSuccess
from scipy._typing import Untyped
from scipy.optimize import (
    Bounds as Bounds,
    LinearConstraint as LinearConstraint,
    NonlinearConstraint as NonlinearConstraint,
    rosen as rosen,
)

class TestInterpolation:
    def test_simple(self): ...
    def test_close(self): ...

class TestQuadratic:
    def test_simple(self): ...
    def test_exceptions(self): ...

class TestModels:
    def test_simple(self): ...
    def test_max_eval(self): ...
    def test_feasibility_problem(self): ...
    def test_target(self): ...

def get_problem(x0) -> Untyped: ...
