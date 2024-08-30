from scipy._typing import Untyped
from ._optimize import OptimizeResult

__all__ = ["root"]

def root(
    fun,
    x0,
    args=(),
    method: str = "hybr",
    jac: Untyped | None = None,
    tol: Untyped | None = None,
    callback: Untyped | None = None,
    options: Untyped | None = None,
) -> OptimizeResult: ...
