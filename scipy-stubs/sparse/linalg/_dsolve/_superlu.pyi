from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeAlias, final, overload

import numpy as np
import optype as op
import optype.numpy as onp
from scipy.sparse import csc_array, csc_matrix, csr_matrix

_Int1D: TypeAlias = onp.Array1D[np.int32]
_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]
_Complex1D: TypeAlias = onp.Array1D[np.complex128]
_Complex2D: TypeAlias = onp.Array2D[np.complex128]
_Inexact2D: TypeAlias = onp.Array2D[np.float32 | np.float64 | np.complex64 | np.complex128]

###

@final
class SuperLU:
    shape: tuple[int, int]
    nnz: int
    perm_r: onp.Array1D[np.intp]
    perm_c: onp.Array1D[np.intp]
    L: csc_array[np.float64 | np.complex128]
    U: csc_array[np.float64 | np.complex128]

    @overload
    def solve(self, /, rhs: onp.Array1D[np.integer[Any] | np.floating[Any]]) -> _Float1D: ...
    @overload
    def solve(self, /, rhs: onp.Array1D[np.complexfloating[Any, Any]]) -> _Complex1D: ...
    @overload
    def solve(self, /, rhs: onp.Array2D[np.integer[Any] | np.floating[Any]]) -> _Float2D: ...
    @overload
    def solve(self, /, rhs: onp.Array2D[np.complexfloating[Any, Any]]) -> _Complex2D: ...
    @overload
    def solve(self, /, rhs: onp.ArrayND[np.integer[Any] | np.floating[Any]]) -> _Float1D | _Float2D: ...
    @overload
    def solve(self, /, rhs: onp.ArrayND[np.complexfloating[Any, Any]]) -> _Complex1D | _Complex2D: ...
    @overload
    def solve(self, /, rhs: onp.ArrayND[np.number[Any]]) -> _Float1D | _Complex1D | _Float2D | _Complex2D: ...

def gssv(
    N: op.CanIndex,
    nnz: op.CanIndex,
    nzvals: _Inexact2D,
    colind: _Int1D,
    rowptr: _Int1D,
    B: _Inexact2D,
    csc: onp.ToBool = 0,
    options: Mapping[str, object] = ...,
) -> tuple[csc_matrix | csr_matrix, int]: ...

#
def gstrf(
    N: op.CanIndex,
    nnz: op.CanIndex,
    nzvals: _Inexact2D,
    colind: _Int1D,
    rowptr: _Int1D,
    csc_construct_func: type[csc_array] | Callable[..., csc_array],
    ilu: onp.ToBool = 0,
    options: Mapping[str, object] = ...,
) -> SuperLU: ...

#
def gstrs(
    trans: Literal["N", "T"],
    L_n: op.CanIndex,
    L_nnz: op.CanIndex,
    L_nzvals: _Inexact2D,
    L_rowind: _Int1D,
    L_colptr: _Int1D,
    U_n: op.CanIndex,
    U_nnz: op.CanIndex,
    U_nzvals: _Inexact2D,
    U_rowind: _Int1D,
    U_colptr: _Int1D,
    B: _Inexact2D,
) -> tuple[_Float1D | _Complex1D | _Float2D | _Complex2D, int]: ...
