from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, type_check_only

import numpy as np
import numpy.typing as npt
from scipy._typing import AnyReal, UntypedCallable
from ._optimize import OptimizeResult

__all__ = ["root"]

_RootMethod: TypeAlias = Literal[
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

@type_check_only
class _OptimizeResult(OptimizeResult):
    x: npt.NDArray[np.number[Any]]
    success: bool
    message: str
    nfev: int

###

def root(
    fun: UntypedCallable,
    x0: npt.NDArray[np.number[Any]],
    args: tuple[object, ...] = (),
    method: _RootMethod = "hybr",
    jac: bool | np.bool_ | UntypedCallable | None = None,
    tol: AnyReal | None = None,
    callback: UntypedCallable | None = None,
    options: Mapping[str, object] | None = None,
) -> _OptimizeResult: ...
