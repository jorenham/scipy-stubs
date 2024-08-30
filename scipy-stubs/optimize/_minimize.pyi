from scipy._typing import Untyped

__all__ = ["minimize", "minimize_scalar"]

def minimize(
    fun,
    x0,
    args=(),
    method: Untyped | None = None,
    jac: Untyped | None = None,
    hess: Untyped | None = None,
    hessp: Untyped | None = None,
    bounds: Untyped | None = None,
    constraints=(),
    tol: Untyped | None = None,
    callback: Untyped | None = None,
    options: Untyped | None = None,
) -> Untyped: ...
def minimize_scalar(
    fun,
    bracket: Untyped | None = None,
    bounds: Untyped | None = None,
    args=(),
    method: Untyped | None = None,
    tol: Untyped | None = None,
    options: Untyped | None = None,
) -> Untyped: ...
def standardize_bounds(bounds, x0, meth) -> Untyped: ...
def standardize_constraints(constraints, x0, meth) -> Untyped: ...
