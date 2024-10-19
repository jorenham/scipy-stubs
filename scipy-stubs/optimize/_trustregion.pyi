import abc

import numpy as np
from scipy._typing import Untyped
from scipy.optimize._differentiable_functions import FD_METHODS as FD_METHODS
from scipy.optimize._hessian_update_strategy import HessianUpdateStrategy as HessianUpdateStrategy

__all__: list[str] = []

class BaseQuadraticSubproblem:
    def __init__(
        self,
        x: Untyped,
        fun: Untyped,
        jac: Untyped,
        hess: Untyped | None = None,
        hessp: Untyped | None = None,
    ) -> None: ...
    def __call__(self, p: Untyped) -> Untyped: ...
    @property
    def fun(self) -> Untyped: ...
    @property
    def jac(self) -> Untyped: ...
    @property
    def hess(self) -> Untyped: ...
    def hessp(self, p: Untyped) -> Untyped: ...
    @property
    def jac_mag(self) -> Untyped: ...
    def get_boundaries_intersections(self, z: Untyped, d: Untyped, trust_radius: Untyped) -> Untyped: ...
    @abc.abstractmethod
    def solve(self, trust_radius: float | np.float64) -> Untyped: ...
