from scipy._typing import Untyped

__all__ = ["MatrixRankWarning", "factorized", "spilu", "splu", "spsolve", "spsolve_triangular", "use_solver"]

class MatrixRankWarning(UserWarning): ...

def use_solver(*, useUmfpack: bool = ..., assumeSortedIndices: bool = ...) -> None: ...
def spsolve(A: Untyped, b: Untyped, permc_spec: Untyped | None = None, use_umfpack: bool = True) -> Untyped: ...
def splu(
    A: Untyped,
    permc_spec: Untyped | None = None,
    diag_pivot_thresh: Untyped | None = None,
    relax: Untyped | None = None,
    panel_size: Untyped | None = None,
    options: Untyped | None = {},
) -> Untyped: ...
def spilu(
    A: Untyped,
    drop_tol: Untyped | None = None,
    fill_factor: Untyped | None = None,
    drop_rule: Untyped | None = None,
    permc_spec: Untyped | None = None,
    diag_pivot_thresh: Untyped | None = None,
    relax: Untyped | None = None,
    panel_size: Untyped | None = None,
    options: Untyped | None = None,
) -> Untyped: ...
def factorized(A: Untyped) -> Untyped: ...
def spsolve_triangular(
    A: Untyped,
    b: Untyped,
    lower: bool = True,
    overwrite_A: bool = False,
    overwrite_b: bool = False,
    unit_diagonal: bool = False,
) -> Untyped: ...
