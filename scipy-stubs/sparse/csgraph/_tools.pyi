from typing import Any, Final, TypeAlias
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, lil_array, lil_matrix, sparray, spmatrix

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...], default=tuple[int, ...])
_SCT = TypeVar("_SCT", bound=np.floating[Any] | np.integer[Any], default=np.floating[Any] | np.integer[Any])

_MaskedArray: TypeAlias = np.ma.MaskedArray[_ShapeT, np.dtype[_SCT]]
_SparseGraph: TypeAlias = csr_matrix | csc_matrix | lil_matrix | csr_array | csc_array | lil_array
_GraphLike: TypeAlias = onp.ToFloat2D | sparray | spmatrix

DTYPE: Final[type[np.float64]] = ...
ITYPE: Final[type[np.int32]] = ...

def csgraph_from_masked(graph: _MaskedArray) -> csr_matrix: ...
def csgraph_masked_from_dense(
    graph: onp.ToFloat2D,
    null_value: float | None = 0,
    nan_null: bool = True,
    infinity_null: bool = True,
    copy: bool = True,
) -> _MaskedArray[tuple[int, int], np.floating[Any]]: ...
def csgraph_from_dense(
    graph: onp.ToFloat2D,
    null_value: float | None = 0,
    nan_null: bool = True,
    infinity_null: bool = True,
) -> csr_matrix: ...
def csgraph_to_dense(csgraph: _SparseGraph, null_value: float | None = 0) -> onp.Array2D[np.floating[Any]]: ...
def csgraph_to_masked(csgraph: _SparseGraph) -> _MaskedArray[tuple[int, int], np.floating[Any]]: ...
def reconstruct_path(csgraph: _GraphLike, predecessors: onp.ToFloatND, directed: bool = True) -> csr_matrix: ...
def construct_dist_matrix(
    graph: _GraphLike,
    predecessors: onp.ToFloatND,
    directed: bool = True,
    null_value: float = ...,
) -> onp.Array2D[np.floating[Any]]: ...
