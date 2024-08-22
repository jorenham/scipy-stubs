from scipy._typing import Untyped
from scipy.optimize import (
    NonlinearConstraint as NonlinearConstraint,
    minimize as minimize,
    rosen as rosen,
    rosen_der as rosen_der,
)

def test_gh21193() -> Untyped: ...
