from typing import Final, overload
from typing_extensions import Self, override

import numpy as np
import optype as op
import optype.numpy as onp
from ._ckdtree import cKDTree, cKDTreeNode

__all__ = ["KDTree", "Rectangle", "distance_matrix", "minkowski_distance", "minkowski_distance_p"]

class Rectangle:
    maxes: Final[onp.Array1D[np.float64]]
    mins: Final[onp.Array1D[np.float64]]
    def __init__(self, /, maxes: onp.ToFloatND, mins: onp.ToFloatND) -> None: ...
    def volume(self, /) -> np.float64: ...
    def split(self, /, d: op.CanIndex, split: onp.ToFloat) -> tuple[Self, Self]: ...
    def min_distance_point(self, /, x: onp.ToFloat | onp.ToFloatND, p: onp.ToFloat = 2.0) -> onp.ArrayND[np.float64]: ...
    def max_distance_point(self, /, x: onp.ToFloat | onp.ToFloatND, p: onp.ToFloat = 2.0) -> onp.ArrayND[np.float64]: ...
    def min_distance_rectangle(self, /, other: Rectangle, p: onp.ToFloat = 2.0) -> onp.ArrayND[np.float64]: ...
    def max_distance_rectangle(self, /, other: Rectangle, p: onp.ToFloat = 2.0) -> onp.ArrayND[np.float64]: ...

class KDTree(cKDTree):
    class node:
        @staticmethod
        def _create(ckdtree_node: cKDTreeNode | None = None) -> KDTree.leafnode | KDTree.innernode: ...
        def __init__(self, /, ckdtree_node: cKDTreeNode | None = None) -> None: ...
        def __lt__(self, other: object, /) -> bool: ...
        def __gt__(self, other: object, /) -> bool: ...
        def __le__(self, other: object, /) -> bool: ...
        def __ge__(self, other: object, /) -> bool: ...

    class leafnode(node):
        @property
        def idx(self, /) -> onp.ArrayND[np.intp]: ...
        @property
        def children(self, /) -> int: ...

    class innernode(node):
        less: Final[KDTree.innernode | KDTree.leafnode]
        greater: Final[KDTree.innernode | KDTree.leafnode]

        def __init__(self, /, ckdtreenode: cKDTreeNode) -> None: ...
        @property
        def split_dim(self, /) -> int: ...
        @property
        def split(self, /) -> float: ...
        @property
        def children(self, /) -> int: ...

    def __init__(
        self,
        /,
        data: onp.ToComplexND,
        leafsize: onp.ToInt = 10,
        compact_nodes: bool = True,
        copy_data: bool = False,
        balanced_tree: bool = True,
        boxsize: onp.ToFloat | onp.ToFloatND | None = None,
    ) -> None: ...
    @override
    def query(
        self,
        /,
        x: onp.ToFloat1D,
        k: onp.ToInt | onp.ToInt1D = 1,
        eps: onp.ToFloat = 0,
        p: onp.ToFloat = 2.0,
        distance_upper_bound: float = ...,  # inf
        workers: int | None = None,
    ) -> tuple[float, np.intp] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.intp]]: ...
    @override
    def query_ball_point(
        self,
        /,
        x: onp.ToFloatND,
        r: onp.ToFloat | onp.ToFloatND,
        p: onp.ToFloat = 2.0,
        eps: onp.ToFloat = 0,
        workers: int | None = None,
        return_sorted: bool | None = None,
        return_length: bool = False,
    ) -> list[int] | onp.ArrayND[np.object_]: ...
    @override
    def query_ball_tree(
        self,
        /,
        other: cKDTree,
        r: onp.ToFloat,
        p: onp.ToFloat = 2.0,
        eps: onp.ToFloat = 0,
    ) -> list[list[int]]: ...

@overload
def minkowski_distance_p(x: onp.ToFloatND, y: onp.ToFloatND, p: int = 2) -> onp.ArrayND[np.float64]: ...
@overload
def minkowski_distance_p(x: onp.ToComplexND, y: onp.ToComplexND, p: int = 2) -> onp.ArrayND[np.float64 | np.complex128]: ...

#
@overload
def minkowski_distance(x: onp.ToFloatND, y: onp.ToFloatND, p: int = 2) -> onp.ArrayND[np.float64]: ...
@overload
def minkowski_distance(x: onp.ToComplexND, y: onp.ToComplexND, p: int = 2) -> onp.ArrayND[np.float64 | np.complex128]: ...

#
@overload
def distance_matrix(x: onp.ToFloatND, y: onp.ToFloatND, p: int = 2, threshold: int = 1_000_000) -> onp.Array2D[np.float64]: ...
@overload
def distance_matrix(
    x: onp.ToComplexND,
    y: onp.ToComplexND,
    p: int = 2,
    threshold: int = 1_000_000,
) -> onp.Array2D[np.float64 | np.complex128]: ...
