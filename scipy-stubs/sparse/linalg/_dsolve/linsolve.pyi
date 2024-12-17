from collections.abc import Mapping
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, overload, type_check_only
from typing_extensions import deprecated

import numpy as np
import optype.numpy as onp
from scipy.sparse._base import SparseEfficiencyWarning, _spbase
from scipy.sparse._bsr import _bsr_base
from scipy.sparse._lil import _lil_base
from ._superlu import SuperLU

__all__ = [
    "MatrixRankWarning",
    "factorized",
    "is_sptriangular",
    "spbandwidth",
    "spilu",
    "splu",
    "spsolve",
    "spsolve_triangular",
    "use_solver",
]

_SparseT = TypeVar("_SparseT", bound=_spbase)

_PermcSpec: TypeAlias = Literal["COLAMD", "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A"]
_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]
_Complex1D: TypeAlias = onp.Array1D[np.complex128]
_Complex2D: TypeAlias = onp.Array2D[np.complex128]

@type_check_only
class _Solve(Protocol):
    @overload
    def __call__(self, b: onp.Array1D[np.integer[Any] | np.floating[Any]], /) -> _Float1D: ...
    @overload
    def __call__(self, b: onp.Array1D[np.complexfloating[Any, Any]], /) -> _Complex1D: ...
    @overload
    def __call__(self, b: onp.Array2D[np.integer[Any] | np.floating[Any]], /) -> _Float2D: ...
    @overload
    def __call__(self, b: onp.Array2D[np.complexfloating[Any, Any]], /) -> _Complex2D: ...
    @overload
    def __call__(self, b: onp.ArrayND[np.integer[Any] | np.floating[Any]], /) -> _Float1D | _Float2D: ...
    @overload
    def __call__(self, b: onp.ArrayND[np.complexfloating[Any, Any]], /) -> _Complex1D | _Complex2D: ...
    @overload
    def __call__(self, b: onp.ArrayND[np.number[Any]], /) -> _Float1D | _Complex1D | _Float2D | _Complex2D: ...

###

class MatrixRankWarning(UserWarning): ...

def use_solver(*, useUmfpack: bool = ..., assumeSortedIndices: bool = ...) -> None: ...
def factorized(A: _spbase | onp.ToComplex2D) -> _Solve: ...

#
@overload
def spsolve(
    A: _spbase | onp.ToComplex2D,
    b: _SparseT,
    permc_spec: _PermcSpec | None = None,
    use_umfpack: bool = True,
) -> _SparseT: ...
@overload
def spsolve(
    A: _spbase | onp.ToComplex2D,
    b: onp.ToComplex2D | onp.ToComplex1D,
    permc_spec: _PermcSpec | None = None,
    use_umfpack: bool = True,
) -> _Float1D | _Complex1D | _Float2D | _Complex2D: ...

#
def spsolve_triangular(
    A: _spbase | onp.ToComplex2D,
    b: _spbase | onp.ToComplex2D | onp.ToComplex1D,
    lower: bool = True,
    overwrite_A: bool = False,
    overwrite_b: bool = False,
    unit_diagonal: bool = False,
) -> _Float1D | _Complex1D | _Float2D | _Complex2D: ...

#
def splu(
    A: _spbase | onp.ToComplex2D,
    permc_spec: _PermcSpec | None = None,
    diag_pivot_thresh: onp.ToFloat | None = None,
    relax: int | None = None,
    panel_size: int | None = None,
    options: Mapping[str, object] | None = None,
) -> SuperLU: ...

#
def spilu(
    A: _spbase | onp.ToComplex2D,
    drop_tol: onp.ToFloat | None = None,
    fill_factor: onp.ToFloat | None = None,
    drop_rule: str | None = None,
    permc_spec: _PermcSpec | None = None,
    diag_pivot_thresh: onp.ToFloat | None = None,
    relax: int | None = None,
    panel_size: int | None = None,
    options: Mapping[str, object] | None = None,
) -> SuperLU: ...

#
@overload
@deprecated("is_sptriangular needs sparse and not BSR format. Converting to CSR.", category=SparseEfficiencyWarning)
def is_sptriangular(A: _bsr_base) -> tuple[bool, bool]: ...
@overload
def is_sptriangular(A: _spbase) -> tuple[bool, bool]: ...

#
@overload
@deprecated("spbandwidth needs sparse format not LIL and BSR. Converting to CSR.", category=SparseEfficiencyWarning)
def spbandwidth(A: _bsr_base | _lil_base) -> tuple[int, int]: ...
@overload
def spbandwidth(A: _spbase) -> tuple[int, int]: ...
