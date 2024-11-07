from scipy._typing import AnyBool, Untyped, UntypedCallable

__all__ = ["fmin_cobyla"]

izip = zip

def synchronized(func: UntypedCallable) -> UntypedCallable: ...  # undocumented
def fmin_cobyla(
    func: UntypedCallable,
    x0: Untyped,
    cons: Untyped,
    args: tuple[object, ...] = (),
    consargs: Untyped | None = None,
    rhobeg: float = 1.0,
    rhoend: float = 0.0001,
    maxfun: int = 1000,
    disp: AnyBool | None = None,
    catol: float = 0.0002,
    *,
    callback: Untyped | None = None,
) -> Untyped: ...
