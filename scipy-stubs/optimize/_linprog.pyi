from collections.abc import Callable, Mapping, Sequence
from typing import Final

import numpy.typing as npt
from ._optimize import OptimizeResult
from ._typing import Bound, MethodLinprog

__all__ = ["linprog", "linprog_terse_callback", "linprog_verbose_callback"]

__docformat__: Final[str] = ...
LINPROG_METHODS: Final[Sequence[MethodLinprog]] = ...

def linprog_verbose_callback(res: OptimizeResult) -> None: ...
def linprog_terse_callback(res: OptimizeResult) -> None: ...

# TODO: Tighen these array-like types
def linprog(
    c: npt.ArrayLike,
    A_ub: npt.ArrayLike | None = None,
    b_ub: npt.ArrayLike | None = None,
    A_eq: npt.ArrayLike | None = None,
    b_eq: npt.ArrayLike | None = None,
    bounds: Bound = (0, None),
    method: MethodLinprog = "highs",
    callback: Callable[[OptimizeResult], None] | None = None,
    # TODO: `TypedDict`
    options: Mapping[str, object] | None = None,
    x0: npt.ArrayLike | None = None,
    integrality: npt.ArrayLike | None = None,
) -> OptimizeResult: ...
