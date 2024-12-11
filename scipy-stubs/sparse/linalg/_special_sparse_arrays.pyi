from typing import Any, Final, Generic, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy.sparse import bsr_array, coo_array, csc_array, csr_array, dia_array, dok_array, lil_array
from scipy.sparse.linalg import LinearOperator

__all__ = ["LaplacianNd"]

_SCT = TypeVar("_SCT", bound=np.number[Any])
_SCT_co = TypeVar("_SCT_co", bound=np.number[Any], default=np.int8, covariant=True)

_BoundaryConditions: TypeAlias = Literal["dirichlet", "neumann", "periodic"]
_ToDType: TypeAlias = type[_SCT] | np.dtype[_SCT] | onp.HasDType[np.dtype[_SCT]]

# because `scipy.sparse.sparray` does not implement anything :(
_SpArray: TypeAlias = bsr_array | coo_array | csc_array | csr_array | dia_array | dok_array | lil_array

###

class LaplacianNd(LinearOperator[_SCT_co], Generic[_SCT_co]):
    grid_shape: Final[onp.AtLeast1D]
    boundary_conditions: Final[_BoundaryConditions]

    @overload  # default dtype (int8)
    def __init__(
        self: LaplacianNd[np.int8],
        /,
        grid_shape: onp.AtLeast1D,
        *,
        boundary_conditions: _BoundaryConditions = "neumann",
        dtype: _ToDType[np.int8] = ...,  # default: np.int8
    ) -> None: ...
    @overload  # know dtype
    def __init__(
        self,
        /,
        grid_shape: onp.AtLeast1D,
        *,
        boundary_conditions: _BoundaryConditions = "neumann",
        dtype: _ToDType[_SCT_co] = ...,  # default: np.int8
    ) -> None: ...

    #
    def eigenvalues(self, /, m: onp.ToJustInt | None = None) -> onp.Array1D[np.float64]: ...
    def eigenvectors(self, /, m: onp.ToJustInt | None = None) -> onp.Array2D[np.float64]: ...
    def toarray(self, /) -> onp.Array2D[_SCT_co]: ...
    def tosparse(self, /) -> _SpArray: ...
