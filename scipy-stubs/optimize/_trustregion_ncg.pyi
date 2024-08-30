from scipy._typing import Untyped
from ._trustregion import BaseQuadraticSubproblem as BaseQuadraticSubproblem

class CGSteihaugSubproblem(BaseQuadraticSubproblem):
    def solve(self, trust_radius) -> Untyped: ...
