from scipy._typing import Untyped

__all__ = ["kmeans", "kmeans2", "vq", "whiten"]

class ClusterError(Exception): ...

def whiten(obs, check_finite: bool = True) -> Untyped: ...
def vq(obs, code_book, check_finite: bool = True) -> Untyped: ...
def py_vq(obs, code_book, check_finite: bool = True) -> Untyped: ...
def kmeans(
    obs,
    k_or_guess,
    iter: int = 20,
    thresh: float = 1e-05,
    check_finite: bool = True,
    *,
    seed: Untyped | None = None,
) -> Untyped: ...
def kmeans2(
    data,
    k,
    iter: int = 10,
    thresh: float = 1e-05,
    minit: str = "random",
    missing: str = "warn",
    check_finite: bool = True,
    *,
    seed: Untyped | None = None,
) -> Untyped: ...
