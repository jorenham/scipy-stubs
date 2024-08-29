from scipy._typing import Untyped

__all__ = ["MatrixRankWarning", "factorized", "spilu", "splu", "spsolve", "spsolve_triangular", "use_solver"]

class MatrixRankWarning(UserWarning): ...

def use_solver(**kwargs): ...
def spsolve(A, b, permc_spec: Untyped | None = None, use_umfpack: bool = True) -> Untyped: ...
def splu(
    A,
    permc_spec: Untyped | None = None,
    diag_pivot_thresh: Untyped | None = None,
    relax: Untyped | None = None,
    panel_size: Untyped | None = None,
    options: Untyped | None = None,
) -> Untyped: ...
def spilu(
    A,
    drop_tol: Untyped | None = None,
    fill_factor: Untyped | None = None,
    drop_rule: Untyped | None = None,
    permc_spec: Untyped | None = None,
    diag_pivot_thresh: Untyped | None = None,
    relax: Untyped | None = None,
    panel_size: Untyped | None = None,
    options: Untyped | None = None,
) -> Untyped: ...
def factorized(A) -> Untyped: ...
def spsolve_triangular(
    A,
    b,
    lower: bool = True,
    overwrite_A: bool = False,
    overwrite_b: bool = False,
    unit_diagonal: bool = False,
) -> Untyped: ...
