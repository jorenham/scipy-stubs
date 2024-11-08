from collections.abc import Sequence
from typing import Literal, type_check_only
from typing_extensions import Self

import numpy as np
import numpy.typing as npt
from scipy._typing import AnyReal, EnterSelfMixin, Seed, Untyped, UntypedCallable
from scipy.optimize import OptimizeResult

__all__ = ["differential_evolution"]

@type_check_only
class _OptimizeResult(OptimizeResult):
    message: str
    success: bool
    fun: float
    x: npt.NDArray[np.float64]  # 1d
    nit: int
    nfev: int
    population: npt.NDArray[np.float64]  # 2d
    population_energies: npt.NDArray[np.float64]  # 1d
    jac: npt.NDArray[np.float64]  # 1d

###

def differential_evolution(
    func: UntypedCallable,
    bounds: Untyped,
    args: tuple[object, ...] = (),
    strategy: str | UntypedCallable = "best1bin",
    maxiter: int = 1000,
    popsize: int = 15,
    tol: AnyReal = 0.01,
    mutation: AnyReal | tuple[AnyReal, AnyReal] = (0.5, 1),
    recombination: AnyReal = 0.7,
    seed: Seed | None = None,
    callback: UntypedCallable | None = None,
    disp: bool = False,
    polish: bool = True,
    init: str | npt.ArrayLike = "latinhypercube",
    atol: AnyReal = 0,
    updating: Literal["immediate", "deferred"] = "immediate",
    workers: int | UntypedCallable = 1,
    constraints: Untyped = (),
    x0: npt.ArrayLike | None = None,
    *,
    integrality: Sequence[bool] | npt.NDArray[np.bool_] | None = None,
    vectorized: bool = False,
) -> _OptimizeResult: ...

# undocumented
class DifferentialEvolutionSolver(EnterSelfMixin):
    mutation_func: Untyped
    strategy: Untyped
    callback: Untyped
    polish: Untyped
    vectorized: Untyped
    scale: Untyped
    dither: Untyped
    cross_over_probability: Untyped
    func: Untyped
    args: Untyped
    limits: Untyped
    maxiter: Untyped
    maxfun: Untyped
    parameter_count: Untyped
    random_number_generator: Untyped
    integrality: Untyped
    num_population_members: Untyped
    population_shape: Untyped
    constraints: Untyped
    total_constraints: Untyped
    constraint_violation: Untyped
    feasible: Untyped
    disp: Untyped
    def __init__(
        self,
        func: Untyped,
        bounds: Untyped,
        args: Untyped = (),
        strategy: str = "best1bin",
        maxiter: int = 1000,
        popsize: int = 15,
        tol: float = 0.01,
        mutation: Untyped = (0.5, 1),
        recombination: float = 0.7,
        seed: Untyped | None = None,
        maxfun: Untyped = ...,
        callback: Untyped | None = None,
        disp: bool = False,
        polish: bool = True,
        init: str = "latinhypercube",
        atol: int = 0,
        updating: str = "immediate",
        workers: int = 1,
        constraints: Untyped = (),
        x0: Untyped | None = None,
        *,
        integrality: Untyped | None = None,
        vectorized: bool = False,
    ) -> None: ...
    population: Untyped
    population_energies: Untyped
    def init_population_lhs(self) -> None: ...
    def init_population_qmc(self, qmc_engine: Untyped) -> None: ...
    def init_population_random(self) -> None: ...
    def init_population_array(self, init: Untyped) -> None: ...
    @property
    def x(self) -> Untyped: ...
    @property
    def convergence(self) -> Untyped: ...
    def converged(self) -> Untyped: ...
    def solve(self) -> Untyped: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Untyped: ...

# undocumented
class _ConstraintWrapper:
    constraint: Untyped
    fun: Untyped
    num_constr: Untyped
    parameter_count: Untyped
    bounds: Untyped
    def __init__(self, constraint: Untyped, x0: Untyped) -> None: ...
    def __call__(self, x: Untyped) -> Untyped: ...
    def violation(self, x: Untyped) -> Untyped: ...
