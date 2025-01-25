from typing import Final, Literal, TypeAlias, overload

import numpy as np
import optype.numpy as onp
from scipy._typing import Falsy, Truthy
from scipy.sparse._base import _spbase
from scipy.sparse._typing import Floating, Integer

_Int1D: TypeAlias = onp.Array1D[np.int32]
_Int2D: TypeAlias = onp.Array2D[np.int32]
_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]

_ToGraphArray: TypeAlias = onp.ToFloat2D | _spbase[np.bool_ | Integer | Floating]

_ShortestPathMethod: TypeAlias = Literal["auto", "FW", "D", "BF", "J"]

###

DTYPE: Final[type[np.float64]] = ...
ITYPE: Final[type[np.int32]] = ...

class NegativeCycleError(Exception):
    def __init__(self, /, message: str = "") -> None: ...

@overload
def shortest_path(
    csgraph: _ToGraphArray,
    method: _ShortestPathMethod = "auto",
    directed: bool = True,
    return_predecessors: Falsy = False,
    unweighted: bool = False,
    overwrite: bool = False,
    indices: onp.ToInt | onp.ToIntND | None = None,
) -> _Float2D: ...
@overload
def shortest_path(
    csgraph: _ToGraphArray,
    method: _ShortestPathMethod,
    directed: bool,
    return_predecessors: Truthy,
    unweighted: bool = False,
    overwrite: bool = False,
    indices: onp.ToInt | onp.ToIntND | None = None,
) -> tuple[_Float2D, _Int2D]: ...
@overload
def shortest_path(
    csgraph: _ToGraphArray,
    method: _ShortestPathMethod = "auto",
    directed: bool = True,
    *,
    return_predecessors: Truthy,
    unweighted: bool = False,
    overwrite: bool = False,
    indices: onp.ToInt | onp.ToIntND | None = None,
) -> tuple[_Float2D, _Int2D]: ...

#
@overload
def floyd_warshall(
    csgraph: _ToGraphArray,
    directed: bool = True,
    return_predecessors: Falsy = False,
    unweighted: bool = False,
    overwrite: bool = False,
) -> _Float2D: ...
@overload
def floyd_warshall(
    csgraph: _ToGraphArray,
    directed: bool,
    return_predecessors: Truthy,
    unweighted: bool = False,
    overwrite: bool = False,
) -> tuple[_Float2D, _Int2D]: ...
@overload
def floyd_warshall(
    csgraph: _ToGraphArray,
    directed: bool = True,
    *,
    return_predecessors: Truthy,
    unweighted: bool = False,
    overwrite: bool = False,
) -> tuple[_Float2D, _Int2D]: ...

#
@overload
def dijkstra(
    csgraph: _ToGraphArray,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    return_predecessors: Falsy = False,
    unweighted: bool = False,
    limit: float = ...,
    min_only: Falsy = False,
) -> _Float2D: ...
@overload
def dijkstra(
    csgraph: _ToGraphArray,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    return_predecessors: Falsy = False,
    unweighted: bool = False,
    limit: float = ...,
    *,
    min_only: Truthy,
) -> _Float1D: ...
@overload
def dijkstra(
    csgraph: _ToGraphArray,
    directed: bool,
    indices: onp.ToIntND | None,
    return_predecessors: Truthy,
    unweighted: bool = False,
    limit: float = ...,
    min_only: Falsy = False,
) -> tuple[_Float2D, _Int2D]: ...
@overload
def dijkstra(
    csgraph: _ToGraphArray,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    *,
    return_predecessors: Truthy,
    unweighted: bool = False,
    limit: float = ...,
    min_only: Falsy = False,
) -> tuple[_Float2D, _Int2D]: ...
@overload
def dijkstra(
    csgraph: _ToGraphArray,
    directed: bool,
    indices: onp.ToIntND | None,
    return_predecessors: Truthy,
    unweighted: bool = False,
    limit: float = ...,
    *,
    min_only: Truthy,
) -> tuple[_Float1D, _Int1D, _Int1D]: ...
@overload
def dijkstra(
    csgraph: _ToGraphArray,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    *,
    return_predecessors: Truthy,
    unweighted: bool = False,
    limit: float = ...,
    min_only: Truthy,
) -> tuple[_Float1D, _Int1D, _Int1D]: ...

#
@overload
def bellman_ford(
    csgraph: _ToGraphArray,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    return_predecessors: Falsy = False,
    unweighted: bool = False,
) -> _Float2D: ...
@overload
def bellman_ford(
    csgraph: _ToGraphArray,
    directed: bool,
    indices: onp.ToIntND | None,
    return_predecessors: Truthy,
    unweighted: bool = False,
) -> tuple[_Float2D, _Int2D]: ...
@overload
def bellman_ford(
    csgraph: _ToGraphArray,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    *,
    return_predecessors: Truthy,
    unweighted: bool = False,
) -> tuple[_Float2D, _Int2D]: ...

#
@overload
def johnson(
    csgraph: _ToGraphArray,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    return_predecessors: Falsy = False,
    unweighted: bool = False,
) -> _Float2D: ...
@overload
def johnson(
    csgraph: _ToGraphArray,
    directed: bool,
    indices: onp.ToIntND | None,
    return_predecessors: Truthy,
    unweighted: bool = False,
) -> tuple[_Float2D, _Int2D]: ...
@overload
def johnson(
    csgraph: _ToGraphArray,
    directed: bool = True,
    indices: onp.ToIntND | None = None,
    *,
    return_predecessors: Truthy,
    unweighted: bool = False,
) -> tuple[_Float2D, _Int2D]: ...

#
@overload
def yen(
    csgraph: _ToGraphArray,
    source: int,
    sink: int,
    K: int,
    *,
    directed: bool = True,
    return_predecessors: Falsy = False,
    unweighted: bool = False,
) -> _Float1D: ...
@overload
def yen(
    csgraph: _ToGraphArray,
    source: int,
    sink: int,
    K: int,
    *,
    directed: bool = True,
    return_predecessors: Truthy,
    unweighted: bool = False,
) -> tuple[_Float1D, _Int2D]: ...
