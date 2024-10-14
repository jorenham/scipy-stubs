from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator

_LaplacianMatrix: TypeAlias = onpt.Array[tuple[int, int], np.number[Any]] | sparray | spmatrix | LinearOperator
_LaplacianDiag: TypeAlias = onpt.Array[tuple[int], np.number[Any]]

@overload
def laplacian(
    csgraph: npt.ArrayLike | spmatrix | sparray,
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
    csgraph: npt.ArrayLike | spmatrix | sparray,
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
    csgraph: npt.ArrayLike | spmatrix | sparray,
    normed: bool = False,
    *,
    return_diag: Literal[True],
    use_out_degree: bool = False,
    copy: bool = True,
    form: Literal["array", "function", "lo"] = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianMatrix, _LaplacianDiag]: ...
