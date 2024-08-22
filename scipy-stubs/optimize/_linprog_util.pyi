from typing import NamedTuple

from ._optimize import OptimizeWarning as OptimizeWarning
from scipy._typing import Untyped

class _LPProblem(NamedTuple):
    c: Untyped
    A_ub: Untyped
    b_ub: Untyped
    A_eq: Untyped
    b_eq: Untyped
    bounds: Untyped
    x0: Untyped
    integrality: Untyped
