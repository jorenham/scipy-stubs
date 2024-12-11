from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator

_LaplacianMatrix: TypeAlias = onp.Array2D[np.number[Any]] | sparray | spmatrix | LinearOperator
_LaplacianDiag: TypeAlias = onp.Array1D[np.number[Any]]
_ToCSGraph: TypeAlias = onp.ToComplex2D | sparray | spmatrix

@overload
def laplacian(
    csgraph: _ToCSGraph,
    normed: bool = False,
    return_diag: Literal[False] = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array", "function", "lo"] = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> _LaplacianMatrix: ...
@overload
def laplacian(
    csgraph: _ToCSGraph,
    normed: bool,
    return_diag: Literal[True],
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: Literal["array", "function", "lo"] = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianMatrix, _LaplacianDiag]: ...
@overload
def laplacian(
    csgraph: _ToCSGraph,
    normed: bool = False,
    *,
    return_diag: Literal[True],
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array", "function", "lo"] = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianMatrix, _LaplacianDiag]: ...
