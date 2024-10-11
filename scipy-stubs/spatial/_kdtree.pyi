from typing_extensions import override

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeInt
from scipy._typing import Untyped
from ._ckdtree import cKDTree, cKDTreeNode

__all__ = ["KDTree", "Rectangle", "distance_matrix", "minkowski_distance", "minkowski_distance_p"]

class Rectangle:
    maxes: Untyped
    mins: Untyped
    def __init__(self, /, maxes: Untyped, mins: Untyped) -> None: ...
    def volume(self, /) -> Untyped: ...
    def split(self, /, d: Untyped, split: Untyped) -> Untyped: ...
    def min_distance_point(self, /, x: Untyped, p: float = 2.0) -> Untyped: ...
    def max_distance_point(self, /, x: Untyped, p: float = 2.0) -> Untyped: ...
    def min_distance_rectangle(self, /, other: Rectangle, p: float = 2.0) -> Untyped: ...
    def max_distance_rectangle(self, /, other: Rectangle, p: float = 2.0) -> Untyped: ...

class KDTree(cKDTree):
    class node:
        def __init__(self, /, ckdtree_node: Untyped | None = None) -> None: ...
        @override
        def __eq__(self, other: object, /) -> bool: ...
        def __lt__(self, other: object, /) -> bool: ...
        def __gt__(self, other: object, /) -> bool: ...
        def __le__(self, other: object, /) -> bool: ...
        def __ge__(self, other: object, /) -> bool: ...

    class leafnode(node):
        @property
        def idx(self, /) -> Untyped: ...
        @property
        def children(self, /) -> Untyped: ...

    class innernode(node):
        less: Untyped
        greater: Untyped
        def __init__(self, /, ckdtreenode: cKDTreeNode) -> None: ...
        @property
        def split_dim(self, /) -> Untyped: ...
        @property
        def split(self, /) -> Untyped: ...
        @property
        def children(self, /) -> Untyped: ...

    def __init__(
        self,
        /,
        data: npt.ArrayLike,
        leafsize: int = 10,
        compact_nodes: bool = True,
        copy_data: bool = False,
        balanced_tree: bool = True,
        boxsize: npt.ArrayLike | None = None,
    ) -> None: ...
    @override
    def query(
        self,
        /,
        x: npt.ArrayLike,
        k: _ArrayLikeInt = 1,
        eps: float = 0,
        p: float = 2.0,
        distance_upper_bound: float = ...,  # inf
        workers: int | None = None,
    ) -> tuple[float, float] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    @override
    def query_ball_point(
        self,
        /,
        x: npt.ArrayLike,
        r: npt.ArrayLike,
        p: float = 2.0,
        eps: float = 0,
        workers: int | None = None,
        return_sorted: bool | None = None,
        return_length: bool = False,
    ) -> list[int] | npt.NDArray[np.object_]: ...
    @override
    def query_ball_tree(
        self,
        /,
        other: cKDTree,
        r: float,
        p: float = 2.0,
        eps: float = 0,
    ) -> list[list[int]]: ...

def minkowski_distance_p(x: npt.ArrayLike, y: npt.ArrayLike, p: int = 2) -> Untyped: ...
def minkowski_distance(x: npt.ArrayLike, y: npt.ArrayLike, p: int = 2) -> Untyped: ...
def distance_matrix(x: npt.ArrayLike, y: npt.ArrayLike, p: int = 2, threshold: int = 1000000) -> Untyped: ...
