from scipy._typing import Untyped

__all__ = ["RegularGridInterpolator", "interpn"]

class RegularGridInterpolator:
    method: Untyped
    bounds_error: Untyped
    values: Untyped
    fill_value: Untyped
    def __init__(
        self,
        points: Untyped,
        values: Untyped,
        method: str = "linear",
        bounds_error: bool = True,
        fill_value: Untyped = ...,
        *,
        solver: Untyped | None = None,
        solver_args: Untyped | None = None,
    ) -> None: ...
    def __call__(self, xi: Untyped, method: Untyped | None = None, *, nu: Untyped | None = None) -> Untyped: ...

def interpn(
    points: Untyped,
    values: Untyped,
    xi: Untyped,
    method: str = "linear",
    bounds_error: bool = True,
    fill_value: Untyped = ...,
) -> Untyped: ...
