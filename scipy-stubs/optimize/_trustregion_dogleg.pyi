from typing_extensions import override

import numpy as np
from scipy._typing import Untyped
from ._trustregion import BaseQuadraticSubproblem

__all__: list[str] = []

class DoglegSubproblem(BaseQuadraticSubproblem):
    def cauchy_point(self) -> Untyped: ...
    def newton_point(self) -> Untyped: ...
    @override
    def solve(self, trust_radius: float | np.float64) -> Untyped: ...
