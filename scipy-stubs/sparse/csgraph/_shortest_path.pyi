from typing import Final, Literal, TypeAlias, overload

import numpy as np
import optype.numpy as onp
from scipy.sparse import sparray, spmatrix

_GraphArrayLike: TypeAlias = onp.ToFloat2D | sparray | spmatrix

_Int1D: TypeAlias = onp.Array1D[np.int32]
_Int2D: TypeAlias = onp.Array2D[np.int32]
_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]

###

DTYPE: Final[type[np.float64]] = ...
ITYPE: Final[type[np.int32]] = ...

class NegativeCycleError(Exception):
    def __init__(self, /, message: str = "") -> None: ...

@overload
def shortest_path(
    csgraph: _GraphArrayLike,
    method: Literal["auto", "FW", "D", "BF", "J"] = "auto",
    directed: bool = True,
    return_predecessors: Literal[False, 0] = False,
    unweighted: bool = False,
    overwrite: bool = False,
    indices: onp.ToInt | onp.ToIntND | None = None,
) -> _Float2D: ...
@overload
def shortest_path(
    csgraph: _GraphArrayLike,
    method: Literal["auto", "FW", "D", "BF", "J"],
    directed: bool,
    return_predecessors: Literal[True, 1],
    unweighted: bool = False,
    overwrite: bool = False,
    indices: onp.ToInt | onp.ToIntND | None = None,
) -> tuple[_Float2D, _Int2D]: ...
@overload
def shortest_path(
    csgraph: _GraphArrayLike,
    method: Literal["auto", "FW", "D", "BF", "J"] = "auto",
    directed: bool = True,
    *,
    return_predecessors: Literal[True, 1],
    unweighted: bool = False,
    overwrite: bool = False,
    indices: onp.ToInt | onp.ToIntND | None = None,
) -> tuple[_Float2D, _Int2D]: ...

#
@overload
def floyd_warshall(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    return_predecessors: Literal[False, 0] = False,
    unweighted: bool = False,
    overwrite: bool = False,
) -> _Float2D: ...
@overload
def floyd_warshall(
    csgraph: _GraphArrayLike,
    directed: bool,
    return_predecessors: Literal[True, 1],
    unweighted: bool = False,
    overwrite: bool = False,
) -> tuple[_Float2D, _Int2D]: ...
@overload
def floyd_warshall(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    *,
    return_predecessors: Literal[True, 1],
    unweighted: bool = False,
    overwrite: bool = False,
) -> tuple[_Float2D, _Int2D]: ...

#
@overload
def dijkstra(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    return_predecessors: Literal[False, 0] = False,
    unweighted: bool = False,
    limit: float = ...,
    min_only: Literal[False, 0] = False,
) -> _Float2D: ...
@overload
def dijkstra(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    return_predecessors: Literal[False, 0] = False,
    unweighted: bool = False,
    limit: float = ...,
    *,
    min_only: Literal[True, 1],
) -> _Float1D: ...
@overload
def dijkstra(
    csgraph: _GraphArrayLike,
    directed: bool,
    indices: onp.ToIntND | None,
    return_predecessors: Literal[True, 1],
    unweighted: bool = False,
    limit: float = ...,
    min_only: Literal[False, 0] = False,
) -> tuple[_Float2D, _Int2D]: ...
@overload
def dijkstra(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    *,
    return_predecessors: Literal[True, 1],
    unweighted: bool = False,
    limit: float = ...,
    min_only: Literal[False, 0] = False,
) -> tuple[_Float2D, _Int2D]: ...
@overload
def dijkstra(
    csgraph: _GraphArrayLike,
    directed: bool,
    indices: onp.ToIntND | None,
    return_predecessors: Literal[True, 1],
    unweighted: bool = False,
    limit: float = ...,
    *,
    min_only: Literal[True, 1],
) -> tuple[_Float1D, _Int1D, _Int1D]: ...
@overload
def dijkstra(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    *,
    return_predecessors: Literal[True, 1],
    unweighted: bool = False,
    limit: float = ...,
    min_only: Literal[True, 1],
) -> tuple[_Float1D, _Int1D, _Int1D]: ...

#
@overload
def bellman_ford(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    return_predecessors: Literal[False, 0] = False,
    unweighted: bool = False,
) -> _Float2D: ...
@overload
def bellman_ford(
    csgraph: _GraphArrayLike,
    directed: bool,
    indices: onp.ToIntND | None,
    return_predecessors: Literal[True, 1],
    unweighted: bool = False,
) -> tuple[_Float2D, _Int2D]: ...
@overload
def bellman_ford(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    *,
    return_predecessors: Literal[True, 1],
    unweighted: bool = False,
) -> tuple[_Float2D, _Int2D]: ...

#
@overload
def johnson(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    return_predecessors: Literal[False, 0] = False,
    unweighted: bool = False,
) -> _Float2D: ...
@overload
def johnson(
    csgraph: _GraphArrayLike,
    directed: bool,
    indices: onp.ToIntND | None,
    return_predecessors: Literal[True, 1],
    unweighted: bool = False,
) -> tuple[_Float2D, _Int2D]: ...
@overload
def johnson(
    csgraph: _GraphArrayLike,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    *,
    return_predecessors: Literal[True, 1],
    unweighted: bool = False,
) -> tuple[_Float2D, _Int2D]: ...

#
@overload
def yen(
    csgraph: _GraphArrayLike,
    source: int,
    sink: int,
    K: int,
    *,
    directed: bool = True,
    return_predecessors: Literal[False, 0] = False,
    unweighted: bool = False,
) -> _Float1D: ...
@overload
def yen(
    csgraph: _GraphArrayLike,
    source: int,
    sink: int,
    K: int,
    *,
    directed: bool = True,
    return_predecessors: Literal[True, 1],
    unweighted: bool = False,
) -> tuple[_Float1D, _Int2D]: ...
