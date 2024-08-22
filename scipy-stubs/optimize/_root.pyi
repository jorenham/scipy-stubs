from ._minpack_py import leastsq as leastsq
from ._optimize import MemoizeJac as MemoizeJac, OptimizeResult as OptimizeResult
from scipy._typing import Untyped

ROOT_METHODS: Untyped

def root(
    fun,
    x0,
    args=(),
    method: str = "hybr",
    jac: Untyped | None = None,
    tol: Untyped | None = None,
    callback: Untyped | None = None,
    options: Untyped | None = None,
) -> Untyped: ...
