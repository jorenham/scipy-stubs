from scipy._typing import Untyped, UntypedCallable, UntypedTuple

def _nsum(
    f: UntypedCallable,
    a: Untyped,
    b: Untyped,
    step: int = 1,
    args: UntypedTuple = (),
    log: bool = False,
    maxterms: int = ...,
    atol: float | None = None,
    rtol: float | None = None,
) -> Untyped: ...
