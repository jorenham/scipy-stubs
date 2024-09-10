__all__ = ["_minimize_trust_krylov"]

from scipy._typing import Untyped, UntypedCallable, UntypedTuple

def _minimize_trust_krylov(
    fun: UntypedCallable,
    x0: Untyped,
    args: UntypedTuple = (),
    jac: Untyped | None = None,
    hess: Untyped | None = None,
    hessp: Untyped | None = None,
    inexact: bool = True,
    **trust_region_options: Untyped,
) -> Untyped: ...
