from scipy._typing import Untyped

__all__ = ["gcrotmk"]

def gcrotmk(
    A: Untyped,
    b: Untyped,
    x0: Untyped | None = None,
    *,
    rtol: float = 1e-05,
    atol: float = 0.0,
    maxiter: int = 1000,
    M: Untyped | None = None,
    callback: Untyped | None = None,
    m: int = 20,
    k: Untyped | None = None,
    CU: Untyped | None = None,
    discard_C: bool = False,
    truncate: str = "oldest",
) -> Untyped: ...
