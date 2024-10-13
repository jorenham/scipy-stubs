# This module is not meant for public use and will be removed in SciPy v2.0.0.
from types import ModuleType
from typing import final
from typing_extensions import deprecated

__all__ = [
    "MatrixRankWarning",
    "SuperLU",
    "factorized",
    "spilu",
    "splu",
    "spsolve",
    "spsolve_triangular",
    "test",
    "use_solver",
]

test: ModuleType

@deprecated("will be removed in SciPy v2.0.0")
class MatrixRankWarning(UserWarning): ...

@final
@deprecated("will be removed in SciPy v2.0.0")
class SuperLU:
    L: object
    U: object
    nnz: object
    perm_r: object
    perm_c: object
    shape: object

    def solve(self, /, rhs: object) -> None: ...

@deprecated("will be removed in SciPy v2.0.0")
def use_solver(**kwargs: object) -> None: ...
@deprecated("will be removed in SciPy v2.0.0")
def spsolve(A: object, b: object, permc_spec: object = ..., use_umfpack: object = ...) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def splu(
    A: object,
    permc_spec: object = ...,
    diag_pivot_thresh: object = ...,
    relax: object = ...,
    panel_size: object = ...,
    options: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def spilu(
    A: object,
    drop_tol: object = ...,
    fill_factor: object = ...,
    drop_rule: object = ...,
    permc_spec: object = ...,
    diag_pivot_thresh: object = ...,
    relax: object = ...,
    panel_size: object = ...,
    options: object = ...,
) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def factorized(A: object) -> object: ...
@deprecated("will be removed in SciPy v2.0.0")
def spsolve_triangular(
    A: object,
    b: object,
    lower: object = ...,
    overwrite_A: object = ...,
    overwrite_b: object = ...,
    unit_diagonal: object = ...,
) -> object: ...
