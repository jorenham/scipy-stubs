from typing import Final
from typing_extensions import Self, override

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt, _ArrayLikeNumber_co
from optype import CanIndex
from scipy._typing import AnyInt, AnyReal
from ._ckdtree import cKDTree, cKDTreeNode

__all__ = ["KDTree", "Rectangle", "distance_matrix", "minkowski_distance", "minkowski_distance_p"]

class Rectangle:
    maxes: Final[onpt.Array[tuple[int], np.float64]]
    mins: Final[onpt.Array[tuple[int], np.float64]]
    def __init__(self, /, maxes: _ArrayLikeFloat_co, mins: _ArrayLikeFloat_co) -> None: ...
    def volume(self, /) -> np.float64: ...
    def split(self, /, d: CanIndex, split: AnyReal) -> tuple[Self, Self]: ...
    def min_distance_point(self, /, x: _ArrayLikeFloat_co, p: AnyReal = 2.0) -> npt.NDArray[np.float64]: ...
    def max_distance_point(self, /, x: _ArrayLikeFloat_co, p: AnyReal = 2.0) -> npt.NDArray[np.float64]: ...
    def min_distance_rectangle(self, /, other: Rectangle, p: AnyReal = 2.0) -> npt.NDArray[np.float64]: ...
    def max_distance_rectangle(self, /, other: Rectangle, p: AnyReal = 2.0) -> npt.NDArray[np.float64]: ...

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
        def idx(self, /) -> npt.NDArray[np.intp]: ...
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
        data: _ArrayLikeFloat_co,
        leafsize: AnyInt = 10,
        compact_nodes: bool = True,
        copy_data: bool = False,
        balanced_tree: bool = True,
        boxsize: npt.ArrayLike | None = None,
    ) -> None: ...
    @override
    def query(
        self,
        /,
        x: _ArrayLikeFloat_co,
        k: _ArrayLikeInt = 1,
        eps: AnyReal = 0,
        p: AnyReal = 2.0,
        distance_upper_bound: float = ...,  # inf
        workers: int | None = None,
    ) -> tuple[float, np.intp] | tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]: ...
    @override
    def query_ball_point(
        self,
        /,
        x: _ArrayLikeFloat_co,
        r: npt.ArrayLike,
        p: AnyReal = 2.0,
        eps: AnyReal = 0,
        workers: int | None = None,
        return_sorted: bool | None = None,
        return_length: bool = False,
    ) -> list[int] | npt.NDArray[np.object_]: ...
    @override
    def query_ball_tree(
        self,
        /,
        other: cKDTree,
        r: AnyReal,
        p: AnyReal = 2.0,
        eps: AnyReal = 0,
    ) -> list[list[int]]: ...

def minkowski_distance_p(
    x: _ArrayLikeNumber_co,
    y: _ArrayLikeNumber_co,
    p: int = 2,
) -> npt.NDArray[np.float64 | np.complex128]: ...
def minkowski_distance(
    x: _ArrayLikeNumber_co,
    y: _ArrayLikeNumber_co,
    p: int = 2,
) -> npt.NDArray[np.float64 | np.complex128]: ...
def distance_matrix(
    x: _ArrayLikeNumber_co,
    y: _ArrayLikeNumber_co,
    p: int = 2,
    threshold: int = 1_000_000,
) -> onpt.Array[tuple[int, int], np.float64 | np.complex128]: ...
