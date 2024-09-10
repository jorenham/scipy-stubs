from scipy._typing import Untyped, UntypedCallable, UntypedTuple

def group_columns(A: Untyped, order: int = 0) -> Untyped: ...
def approx_derivative(
    fun: UntypedCallable,
    x0: Untyped,
    method: str = "3-point",
    rel_step: Untyped | None = None,
    abs_step: Untyped | None = None,
    f0: Untyped | None = None,
    bounds: Untyped = ...,
    sparsity: Untyped | None = None,
    as_linear_operator: bool = False,
    args: UntypedTuple = (),
    kwargs: Untyped = ...,
) -> Untyped: ...
def check_derivative(
    fun: UntypedCallable,
    jac: UntypedCallable,
    x0: Untyped,
    bounds: Untyped = ...,
    args: UntypedTuple = (),
    kwargs: Untyped = ...,
) -> Untyped: ...
