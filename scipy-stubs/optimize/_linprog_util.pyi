from typing import NamedTuple

from scipy._typing import Untyped
from ._optimize import OptimizeWarning as OptimizeWarning

class _LPProblem(NamedTuple):
    c: Untyped
    A_ub: Untyped
    b_ub: Untyped
    A_eq: Untyped
    b_eq: Untyped
    bounds: Untyped
    x0: Untyped
    integrality: Untyped
