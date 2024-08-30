from scipy._lib._util import MapWrapper as MapWrapper, check_random_state as check_random_state, rng_integers as rng_integers
from scipy._typing import Untyped
from scipy.optimize import OptimizeResult as OptimizeResult, minimize as minimize
from scipy.optimize._constraints import (
    Bounds as Bounds,
    LinearConstraint as LinearConstraint,
    NonlinearConstraint as NonlinearConstraint,
    new_bounds_to_old as new_bounds_to_old,
)
from scipy.sparse import issparse as issparse

def differential_evolution(
    func,
    bounds,
    args=(),
    strategy: str = "best1bin",
    maxiter: int = 1000,
    popsize: int = 15,
    tol: float = 0.01,
    mutation=(0.5, 1),
    recombination: float = 0.7,
    seed: Untyped | None = None,
    callback: Untyped | None = None,
    disp: bool = False,
    polish: bool = True,
    init: str = "latinhypercube",
    atol: int = 0,
    updating: str = "immediate",
    workers: int = 1,
    constraints=(),
    x0: Untyped | None = None,
    *,
    integrality: Untyped | None = None,
    vectorized: bool = False,
) -> Untyped: ...

class DifferentialEvolutionSolver:
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
        func,
        bounds,
        args=(),
        strategy: str = "best1bin",
        maxiter: int = 1000,
        popsize: int = 15,
        tol: float = 0.01,
        mutation=(0.5, 1),
        recombination: float = 0.7,
        seed: Untyped | None = None,
        maxfun=...,
        callback: Untyped | None = None,
        disp: bool = False,
        polish: bool = True,
        init: str = "latinhypercube",
        atol: int = 0,
        updating: str = "immediate",
        workers: int = 1,
        constraints=(),
        x0: Untyped | None = None,
        *,
        integrality: Untyped | None = None,
        vectorized: bool = False,
    ): ...
    population: Untyped
    population_energies: Untyped
    def init_population_lhs(self): ...
    def init_population_qmc(self, qmc_engine): ...
    def init_population_random(self): ...
    def init_population_array(self, init): ...
    @property
    def x(self) -> Untyped: ...
    @property
    def convergence(self) -> Untyped: ...
    def converged(self) -> Untyped: ...
    def solve(self) -> Untyped: ...
    def __iter__(self) -> Untyped: ...
    def __enter__(self) -> Untyped: ...
    def __exit__(self, *args) -> Untyped: ...
    def __next__(self) -> Untyped: ...

class _ConstraintWrapper:
    constraint: Untyped
    fun: Untyped
    num_constr: Untyped
    parameter_count: Untyped
    bounds: Untyped
    def __init__(self, constraint, x0) -> None: ...
    def __call__(self, x) -> Untyped: ...
    def violation(self, x) -> Untyped: ...
