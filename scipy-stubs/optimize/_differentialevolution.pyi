from collections.abc import Sequence
from typing import Literal, type_check_only

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from scipy._typing import Seed, Untyped, UntypedCallable
from scipy.optimize import OptimizeResult

__all__ = ["differential_evolution"]

@type_check_only
class _OptimizeResult(OptimizeResult):
    message: str
    success: bool
    fun: float
    x: onp.ArrayND[np.float64]  # 1d
    nit: int
    nfev: int
    population: onp.ArrayND[np.float64]  # 2d
    population_energies: onp.ArrayND[np.float64]  # 1d
    jac: onp.ArrayND[np.float64]  # 1d

###

def differential_evolution(
    func: UntypedCallable,
    bounds: Untyped,
    args: tuple[object, ...] = (),
    strategy: str | UntypedCallable = "best1bin",
    maxiter: int = 1000,
    popsize: int = 15,
    tol: onp.ToFloat = 0.01,
    mutation: onp.ToFloat | tuple[onp.ToFloat, onp.ToFloat] = (0.5, 1),
    recombination: onp.ToFloat = 0.7,
    seed: Seed | None = None,
    callback: UntypedCallable | None = None,
    disp: bool = False,
    polish: bool = True,
    init: str | npt.ArrayLike = "latinhypercube",
    atol: onp.ToFloat = 0,
    updating: Literal["immediate", "deferred"] = "immediate",
    workers: int | UntypedCallable = 1,
    constraints: Untyped = (),
    x0: npt.ArrayLike | None = None,
    *,
    integrality: Sequence[bool] | onp.ArrayND[np.bool_] | None = None,
    vectorized: bool = False,
) -> _OptimizeResult: ...
