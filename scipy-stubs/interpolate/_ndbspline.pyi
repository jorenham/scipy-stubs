from scipy._typing import Untyped
from scipy.sparse import csr_array as csr_array

class NdBSpline:
    k: Untyped
    t: Untyped
    c: Untyped
    extrapolate: Untyped
    def __init__(self, t, c, k, *, extrapolate: Untyped | None = None): ...
    def __call__(self, xi, *, nu: Untyped | None = None, extrapolate: Untyped | None = None) -> Untyped: ...
    @classmethod
    def design_matrix(cls, xvals, t, k, extrapolate: bool = True) -> Untyped: ...

def make_ndbspl(points, values, k: int = 3, *, solver=..., **solver_args) -> Untyped: ...
