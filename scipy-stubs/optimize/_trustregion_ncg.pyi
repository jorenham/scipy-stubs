from ._trustregion import BaseQuadraticSubproblem as BaseQuadraticSubproblem
from scipy._typing import Untyped

class CGSteihaugSubproblem(BaseQuadraticSubproblem):
    def solve(self, trust_radius) -> Untyped: ...
