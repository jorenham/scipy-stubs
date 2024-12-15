from typing import Final, Literal, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp
from scipy.sparse import csr_matrix
from scipy.sparse._base import _spbase
from scipy.sparse._typing import Float, Int

_T = TypeVar("_T")
_Pair: TypeAlias = tuple[_T, _T]

_Falsy: TypeAlias = Literal[False, 0]
_Truthy: TypeAlias = Literal[True, 1]

_Real: TypeAlias = Int | Float
_Int1D: TypeAlias = onp.Array1D[np.int32]

_ToGraph: TypeAlias = onp.ToFloat2D | _spbase[_Real, tuple[int, int]]

###

DTYPE: Final[type[np.float64]] = ...
ITYPE: Final[type[np.int32]] = ...

def connected_components(
    csgraph: _ToGraph,
    directed: bool = True,
    connection: Literal["weak", "strong"] = "weak",
    return_labels: bool = True,
) -> tuple[int, _Int1D]: ...
def breadth_first_tree(csgraph: _ToGraph, i_start: int, directed: bool = True) -> csr_matrix[_Real]: ...
def depth_first_tree(csgraph: _ToGraph, i_start: int, directed: bool = True) -> csr_matrix[_Real]: ...

#
@overload
def breadth_first_order(
    csgraph: _ToGraph,
    i_start: int,
    directed: bool = True,
    return_predecessors: _Truthy = True,
) -> _Pair[_Int1D]: ...
@overload
def breadth_first_order(csgraph: _ToGraph, i_start: int, directed: bool, return_predecessors: _Falsy) -> _Int1D: ...
@overload
def breadth_first_order(csgraph: _ToGraph, i_start: int, directed: bool = True, *, return_predecessors: _Falsy) -> _Int1D: ...

#
@overload
def depth_first_order(
    csgraph: _ToGraph,
    i_start: int,
    directed: bool = True,
    return_predecessors: _Truthy = True,
) -> _Pair[_Int1D]: ...
@overload
def depth_first_order(csgraph: _ToGraph, i_start: int, directed: bool, return_predecessors: _Falsy) -> _Int1D: ...
@overload
def depth_first_order(csgraph: _ToGraph, i_start: int, directed: bool = True, *, return_predecessors: _Falsy) -> _Int1D: ...
