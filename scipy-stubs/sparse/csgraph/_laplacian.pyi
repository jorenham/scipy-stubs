from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from scipy._typing import Falsy, Truthy
from scipy.sparse._base import _spbase
from scipy.sparse.linalg import LinearOperator

_LaplacianMatrix: TypeAlias = onp.Array2D[np.number[Any]] | _spbase | LinearOperator
_LaplacianDiag: TypeAlias = onp.Array1D[np.number[Any]]
_ToCSGraph: TypeAlias = onp.ToComplex2D | _spbase
_Form: TypeAlias = Literal["array", "function", "lo"]

###

@overload
def laplacian(
    csgraph: _ToCSGraph,
    normed: bool = False,
    return_diag: Falsy = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: _Form = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> _LaplacianMatrix: ...
@overload
def laplacian(
    csgraph: _ToCSGraph,
    normed: bool,
    return_diag: Truthy,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: _Form = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianMatrix, _LaplacianDiag]: ...
@overload
def laplacian(
    csgraph: _ToCSGraph,
    normed: bool = False,
    *,
    return_diag: Truthy,
    use_out_degree: bool = False,
    copy: bool = True,
    form: _Form = "array",
    dtype: npt.DTypeLike | None = None,
    symmetrized: bool = False,
) -> tuple[_LaplacianMatrix, _LaplacianDiag]: ...
