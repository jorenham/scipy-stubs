from scipy._typing import Untyped

def find_root(
    f, init, /, *, args=(), tolerances: Untyped | None = None, maxiter: Untyped | None = None, callback: Untyped | None = None
) -> Untyped: ...
def find_minimum(
    f, init, /, *, args=(), tolerances: Untyped | None = None, maxiter: int = 100, callback: Untyped | None = None
) -> Untyped: ...
def bracket_root(
    f,
    xl0,
    xr0: Untyped | None = None,
    *,
    xmin: Untyped | None = None,
    xmax: Untyped | None = None,
    factor: Untyped | None = None,
    args=(),
    maxiter: int = 1000,
) -> Untyped: ...
def bracket_minimum(
    f,
    xm0,
    *,
    xl0: Untyped | None = None,
    xr0: Untyped | None = None,
    xmin: Untyped | None = None,
    xmax: Untyped | None = None,
    factor: Untyped | None = None,
    args=(),
    maxiter: int = 1000,
) -> Untyped: ...
