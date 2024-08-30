from scipy._typing import Untyped
from ._optimize import OptimizeResult as OptimizeResult, OptimizeWarning as OptimizeWarning

__docformat__: str
LINPROG_METHODS: Untyped

def linprog_verbose_callback(res) -> Untyped: ...
def linprog_terse_callback(res): ...
def linprog(
    c,
    A_ub: Untyped | None = None,
    b_ub: Untyped | None = None,
    A_eq: Untyped | None = None,
    b_eq: Untyped | None = None,
    bounds=(0, None),
    method: str = "highs",
    callback: Untyped | None = None,
    options: Untyped | None = None,
    x0: Untyped | None = None,
    integrality: Untyped | None = None,
) -> Untyped: ...
