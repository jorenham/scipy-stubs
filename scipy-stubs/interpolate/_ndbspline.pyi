from scipy._typing import Untyped

__all__ = ["NdBSpline"]

class NdBSpline:
    k: Untyped
    t: Untyped
    c: Untyped
    extrapolate: Untyped

    def __init__(self, t: Untyped, c: Untyped, k: Untyped, *, extrapolate: bool | None = None) -> None: ...
    def __call__(self, xi: Untyped, *, nu: Untyped | None = None, extrapolate: bool | None = None) -> Untyped: ...
    @classmethod
    def design_matrix(cls, xvals: Untyped, t: Untyped, k: Untyped, extrapolate: bool = True) -> Untyped: ...

def make_ndbspl(
    points: Untyped,
    values: Untyped,
    k: int = 3,
    *,
    solver: Untyped = ...,
    **solver_args: Untyped,
) -> Untyped: ...  # undocumented
