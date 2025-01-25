from typing import Final, TypeAlias
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, lil_array, lil_matrix
from scipy.sparse._base import _spbase
from scipy.sparse._typing import Floating, Integer

_Real: TypeAlias = Integer | Floating
_RealT = TypeVar("_RealT", bound=_Real, default=_Real)

_SparseGraph: TypeAlias = (
    csr_array[_RealT] | csr_matrix[_RealT]
    | csc_array[_RealT] | csc_matrix[_RealT]
    | lil_array[_RealT] | lil_matrix[_RealT]
)  # fmt: skip

_ToGraph: TypeAlias = onp.ToFloat2D | _spbase[_Real, tuple[int, int]]

###

DTYPE: Final[type[np.float64]] = ...
ITYPE: Final[type[np.int32]] = ...

def csgraph_from_masked(graph: onp.MArray2D[_Real]) -> csr_array[np.float64 | np.int32]: ...

#
def csgraph_masked_from_dense(
    graph: onp.ToFloat2D,
    null_value: float | None = 0,
    nan_null: bool = True,
    infinity_null: bool = True,
    copy: bool = True,
) -> onp.MArray2D[np.float64 | np.int32]: ...

#
def csgraph_from_dense(
    graph: onp.ToFloat2D,
    null_value: float | None = 0,
    nan_null: bool = True,
    infinity_null: bool = True,
) -> csr_array[np.float64 | np.int32]: ...

#
def csgraph_to_dense(csgraph: _SparseGraph[_Real], null_value: float | None = 0) -> onp.Array2D[np.float64 | np.int32]: ...
def csgraph_to_masked(csgraph: _SparseGraph[_Real]) -> onp.MArray2D[np.float64 | np.int32]: ...

#
def reconstruct_path(
    csgraph: _ToGraph,
    predecessors: onp.ToFloatND,
    directed: bool = True,
) -> csr_matrix[np.float64 | np.int32]: ...

#
def construct_dist_matrix(
    graph: _ToGraph,
    predecessors: onp.ToFloatND,
    directed: bool = True,
    null_value: float = ...,
) -> onp.Array2D[np.float64 | np.int32]: ...
