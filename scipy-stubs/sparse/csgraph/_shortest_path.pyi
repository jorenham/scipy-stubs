from typing import Final, Literal, TypeAlias, overload

import numpy as np
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt
from scipy.sparse import sparray, spmatrix

_GraphArrayLike: TypeAlias = _ArrayLikeFloat_co | sparray | spmatrix

_DVector: TypeAlias = onpt.Array[tuple[int], np.float64]
_DMatrix: TypeAlias = onpt.Array[tuple[int, int], np.float64]
_IVector: TypeAlias = onpt.Array[tuple[int], np.int32]
_IMatrix: TypeAlias = onpt.Array[tuple[int, int], np.int32]

DTYPE: Final[type[np.float64]] = ...
ITYPE: Final[type[np.int32]] = ...

class NegativeCycleError(Exception):
    def __init__(self, /, message: str = "") -> None: ...

@overload
def shortest_path(
    csgraph: _GraphArrayLike,
    method: Literal["auto", "FW", "D", "BF", "J"] = "auto",
    directed: bool = True,
    return_predecessors: Literal[False] = False,
    unweighted: bool = False,
    overwrite: bool = False,
    indices: _ArrayLikeInt | None = None,
) -> _DMatrix: ...
@overload
def shortest_path(
    csgraph: _GraphArrayLike,
    method: Literal["auto", "FW", "D", "BF", "J"],
    directed: bool,
    return_predecessors: Literal[True],
    unweighted: bool = False,
    overwrite: bool = False,
    indices: _ArrayLikeInt | None = None,
) -> tuple[_DMatrix, _IMatrix]: ...
@overload
def shortest_path(
    csgraph: _GraphArrayLike,
    method: Literal["auto", "FW", "D", "BF", "J"] = "auto",
    directed: bool = True,
    *,
    return_predecessors: Literal[True],
    unweighted: bool = False,
    overwrite: bool = False,
    indices: _ArrayLikeInt | None = None,
) -> tuple[_DMatrix, _IMatrix]: ...

#
@overload
def floyd_warshall(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    return_predecessors: Literal[False] = False,
    unweighted: bool = False,
    overwrite: bool = False,
) -> _DMatrix: ...
@overload
def floyd_warshall(
    csgraph: _GraphArrayLike,
    directed: bool,
    return_predecessors: Literal[True],
    unweighted: bool = False,
    overwrite: bool = False,
) -> tuple[_DMatrix, _IMatrix]: ...
@overload
def floyd_warshall(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    *,
    return_predecessors: Literal[True],
    unweighted: bool = False,
    overwrite: bool = False,
) -> tuple[_DMatrix, _IMatrix]: ...

#
@overload
def dijkstra(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: _ArrayLikeInt | None = None,
    return_predecessors: Literal[False] = False,
    unweighted: bool = False,
    limit: float = ...,
    min_only: Literal[False] = False,
) -> _DMatrix: ...
@overload
def dijkstra(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: _ArrayLikeInt | None = None,
    return_predecessors: Literal[False] = False,
    unweighted: bool = False,
    limit: float = ...,
    *,
    min_only: Literal[True],
) -> _DVector: ...
@overload
def dijkstra(
    csgraph: _GraphArrayLike,
    directed: bool,
    indices: _ArrayLikeInt | None,
    return_predecessors: Literal[True],
    unweighted: bool = False,
    limit: float = ...,
    min_only: Literal[False] = False,
) -> tuple[_DMatrix, _IMatrix]: ...
@overload
def dijkstra(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: _ArrayLikeInt | None = None,
    *,
    return_predecessors: Literal[True],
    unweighted: bool = False,
    limit: float = ...,
    min_only: Literal[False] = False,
) -> tuple[_DMatrix, _IMatrix]: ...
@overload
def dijkstra(
    csgraph: _GraphArrayLike,
    directed: bool,
    indices: _ArrayLikeInt | None,
    return_predecessors: Literal[True],
    unweighted: bool = False,
    limit: float = ...,
    *,
    min_only: Literal[True],
) -> tuple[_DVector, _IVector, _IVector]: ...
@overload
def dijkstra(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: _ArrayLikeInt | None = None,
    *,
    return_predecessors: Literal[True],
    unweighted: bool = False,
    limit: float = ...,
    min_only: Literal[True],
) -> tuple[_DVector, _IVector, _IVector]: ...

#
@overload
def bellman_ford(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: _ArrayLikeInt | None = None,
    return_predecessors: Literal[False] = False,
    unweighted: bool = False,
) -> _DMatrix: ...
@overload
def bellman_ford(
    csgraph: _GraphArrayLike,
    directed: bool,
    indices: _ArrayLikeInt | None,
    return_predecessors: Literal[True],
    unweighted: bool = False,
) -> tuple[_DMatrix, _IMatrix]: ...
@overload
def bellman_ford(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: _ArrayLikeInt | None = None,
    *,
    return_predecessors: Literal[True],
    unweighted: bool = False,
) -> tuple[_DMatrix, _IMatrix]: ...

#
@overload
def johnson(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: _ArrayLikeInt | None = None,
    return_predecessors: Literal[False] = False,
    unweighted: bool = False,
) -> _DMatrix: ...
@overload
def johnson(
    csgraph: _GraphArrayLike,
    directed: bool,
    indices: _ArrayLikeInt | None,
    return_predecessors: Literal[True],
    unweighted: bool = False,
) -> tuple[_DMatrix, _IMatrix]: ...
@overload
def johnson(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: _ArrayLikeInt | None = None,
    *,
    return_predecessors: Literal[True],
    unweighted: bool = False,
) -> tuple[_DMatrix, _IMatrix]: ...

#
@overload
def yen(
    csgraph: _GraphArrayLike,
    source: int,
    sink: int,
    K: int,
    *,
    directed: bool = True,
    return_predecessors: Literal[False] = False,
    unweighted: bool = False,
) -> _DVector: ...
@overload
def yen(
    csgraph: _GraphArrayLike,
    source: int,
    sink: int,
    K: int,
    *,
    directed: bool = True,
    return_predecessors: Literal[True],
    unweighted: bool = False,
) -> tuple[_DVector, _IMatrix]: ...
