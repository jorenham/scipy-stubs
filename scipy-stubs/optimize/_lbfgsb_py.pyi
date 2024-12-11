from collections.abc import Callable, Sequence
from typing import Concatenate, Final, Literal, TypeAlias, TypedDict, type_check_only

import numpy as np
import optype.numpy as onp
from scipy.sparse.linalg import LinearOperator

__all__ = ["LbfgsInvHessProduct", "fmin_l_bfgs_b"]

_Ignored: TypeAlias = object

@type_check_only
class _InfoDict(TypedDict):
    grad: onp.Array1D[np.float64]
    task: str  # undocumented
    funcalls: int
    nit: int
    warnflag: Literal[0, 1, 2]

###

class LbfgsInvHessProduct(LinearOperator[np.float64]):
    sk: Final[onp.Array2D[np.float64]]
    yk: Final[onp.Array2D[np.float64]]
    n_corrs: Final[int]
    rho: Final[float | np.float64]

    def __init__(self, /, sk: onp.ToFloat2D, yk: onp.ToFloat2D) -> None: ...
    def todense(self, /) -> onp.Array2D[np.float64]: ...

def fmin_l_bfgs_b(
    func: Callable[Concatenate[onp.Array1D[np.float64], ...], onp.ToFloat],
    x0: onp.ToFloat | onp.ToFloat1D,
    fprime: Callable[Concatenate[onp.Array1D[np.float64], ...], onp.ToFloat1D] | None = None,
    args: tuple[object, ...] = (),
    approx_grad: onp.ToBool = 0,
    bounds: Sequence[tuple[onp.ToFloat, onp.ToFloat]] | None = None,
    m: onp.ToJustInt = 10,
    factr: onp.ToFloat = 1e7,
    pgtol: onp.ToFloat = 1e-5,
    epsilon: onp.ToFloat = 1e-8,
    iprint: onp.ToJustInt = -1,
    maxfun: onp.ToJustInt = 15_000,
    maxiter: onp.ToJustInt = 15_000,
    disp: onp.ToJustInt | None = None,
    callback: Callable[[onp.Array1D[np.float64]], _Ignored] | None = None,
    maxls: onp.ToJustInt = 20,
) -> tuple[onp.Array1D[np.float64], float | np.float64, _InfoDict]: ...
