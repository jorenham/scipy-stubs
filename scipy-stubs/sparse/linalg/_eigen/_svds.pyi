from scipy._typing import Untyped

__all__ = ["svds"]

def svds(
    A: Untyped,
    k: int = 6,
    ncv: Untyped | None = None,
    tol: int = 0,
    which: str = "LM",
    v0: Untyped | None = None,
    maxiter: Untyped | None = None,
    return_singular_vectors: bool = True,
    solver: str = "arpack",
    random_state: Untyped | None = None,
    options: Untyped | None = None,
) -> Untyped: ...
