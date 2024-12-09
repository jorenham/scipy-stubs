from collections.abc import Callable, Mapping, Sequence
from typing import Final, Literal, type_check_only

import numpy as np
import optype.numpy as onp
from ._optimize import OptimizeResult
from ._typing import Bound, MethodLinprog

__all__ = ["linprog", "linprog_terse_callback", "linprog_verbose_callback"]

@type_check_only
class _OptimizeResult(OptimizeResult):
    x: onp.ArrayND[np.float64]
    fun: float
    slack: onp.ArrayND[np.float64]
    con: onp.ArrayND[np.float64]
    success: bool
    status: Literal[0, 1, 2, 3, 4]
    nit: int
    message: str

__docformat__: Final[str] = ...
LINPROG_METHODS: Final[Sequence[MethodLinprog]] = ...

def linprog_verbose_callback(res: _OptimizeResult) -> None: ...
def linprog_terse_callback(res: _OptimizeResult) -> None: ...

# TODO: Tighen these array-like types
def linprog(
    c: onp.ToScalar | onp.ToArrayND,
    A_ub: onp.ToScalar | onp.ToArrayND | None = None,
    b_ub: onp.ToScalar | onp.ToArrayND | None = None,
    A_eq: onp.ToScalar | onp.ToArrayND | None = None,
    b_eq: onp.ToScalar | onp.ToArrayND | None = None,
    bounds: Bound = (0, None),
    method: MethodLinprog = "highs",
    callback: Callable[[_OptimizeResult], None] | None = None,
    # TODO: `TypedDict`
    options: Mapping[str, object] | None = None,
    x0: onp.ToScalar | onp.ToArrayND | None = None,
    integrality: onp.ToScalar | onp.ToArrayND | None = None,
) -> _OptimizeResult: ...
