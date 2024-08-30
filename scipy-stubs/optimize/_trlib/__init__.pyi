from scipy._typing import Untyped
from ._trlib import TRLIBQuadraticSubproblem

__all__ = ["TRLIBQuadraticSubproblem", "get_trlib_quadratic_subproblem"]

def get_trlib_quadratic_subproblem(tol_rel_i: float = -2.0, tol_rel_b: float = -3.0, disp: bool = False) -> Untyped: ...
