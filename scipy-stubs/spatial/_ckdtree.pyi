from typing import Literal as L, TypeAlias, overload, type_check_only

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt
from scipy._typing import AnyReal, AnyScalar
from scipy.sparse import coo_matrix, dok_matrix

__all__ = ["cKDTree"]

_Weights: TypeAlias = npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike]

@type_check_only
class _CythonMixin:
    def __setstate_cython__(self, pyx_state: object, /) -> None: ...
    def __reduce_cython__(self, /) -> None: ...

class cKDTreeNode(_CythonMixin):
    @property
    def data_points(self) -> npt.NDArray[np.float64]: ...
    @property
    def indices(self) -> npt.NDArray[np.intp]: ...

    # These are read-only attributes in cython, which behave like properties
    @property
    def level(self) -> int: ...
    @property
    def split_dim(self) -> int: ...
    @property
    def children(self) -> int: ...
    @property
    def start_idx(self) -> int: ...
    @property
    def end_idx(self) -> int: ...
    @property
    def split(self) -> float: ...
    @property
    def lesser(self) -> cKDTreeNode | None: ...
    @property
    def greater(self) -> cKDTreeNode | None: ...

class cKDTree(_CythonMixin):
    @property
    def n(self) -> int: ...
    @property
    def m(self) -> int: ...
    @property
    def leafsize(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def tree(self) -> cKDTreeNode: ...

    # These are read-only attributes in cython, which behave like properties
    @property
    def data(self) -> npt.NDArray[np.float64]: ...
    @property
    def maxes(self) -> npt.NDArray[np.float64]: ...
    @property
    def mins(self) -> npt.NDArray[np.float64]: ...
    @property
    def indices(self) -> npt.NDArray[np.float64]: ...
    @property
    def boxsize(self) -> npt.NDArray[np.float64] | None: ...

    #
    def __init__(
        self,
        /,
        data: npt.ArrayLike,
        leafsize: int = ...,
        compact_nodes: bool = ...,
        copy_data: bool = ...,
        balanced_tree: bool = ...,
        boxsize: npt.ArrayLike | None = ...,
    ) -> None: ...

    #
    def query(
        self,
        /,
        x: _ArrayLikeFloat_co,
        k: _ArrayLikeInt = 1,
        eps: AnyReal = 0.0,
        p: AnyReal = 2.0,
        distance_upper_bound: float = ...,  # inf
        workers: int | None = None,
    ) -> tuple[float, np.intp] | tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]: ...

    #
    def query_ball_point(
        self,
        /,
        x: _ArrayLikeFloat_co,
        r: npt.ArrayLike,
        p: AnyReal = 2.0,
        eps: AnyReal = 0.0,
        workers: int | None = None,
        return_sorted: bool | None = None,
        return_length: bool = False,
    ) -> list[int] | npt.NDArray[np.object_]: ...

    #
    def query_ball_tree(self, /, other: cKDTree, r: AnyReal, p: AnyReal = 2.0, eps: AnyReal = 0.0) -> list[list[int]]: ...

    #
    @overload
    def query_pairs(
        self,
        /,
        r: AnyReal,
        p: AnyReal = 2.0,
        eps: AnyReal = 0,
        output_type: L["set"] = "set",
    ) -> set[tuple[int, int]]: ...
    @overload
    def query_pairs(
        self,
        /,
        r: AnyReal,
        p: AnyReal,
        eps: AnyReal,
        output_type: L["ndarray"],
    ) -> npt.NDArray[np.intp]: ...
    @overload
    def query_pairs(
        self,
        /,
        r: AnyReal,
        p: AnyReal = 2.0,
        eps: AnyReal = 0,
        *,
        output_type: L["ndarray"],
    ) -> npt.NDArray[np.intp]: ...

    #
    @overload
    def count_neighbors(
        self,
        /,
        other: cKDTree,
        r: AnyScalar,
        p: AnyReal = 2.0,
        weights: tuple[None, None] | None = None,
        cumulative: bool = True,
    ) -> np.intp: ...
    @overload
    def count_neighbors(
        self,
        /,
        other: cKDTree,
        r: AnyScalar,
        p: AnyReal,
        weights: _Weights,
        cumulative: bool = True,
    ) -> np.float64: ...
    @overload
    def count_neighbors(
        self,
        /,
        other: cKDTree,
        r: AnyScalar,
        p: AnyReal = 2.0,
        *,
        weights: _Weights,
        cumulative: bool = True,
    ) -> np.float64: ...
    @overload
    def count_neighbors(
        self,
        /,
        other: cKDTree,
        r: npt.ArrayLike,
        p: AnyReal = 2.0,
        weights: None | tuple[None, None] = ...,
        cumulative: bool = True,
    ) -> np.float64 | np.intp | npt.NDArray[np.intp]: ...
    @overload
    def count_neighbors(
        self,
        /,
        other: cKDTree,
        r: npt.ArrayLike,
        p: AnyReal,
        weights: _Weights,
        cumulative: bool = True,
    ) -> np.float64 | np.intp | npt.NDArray[np.float64]: ...
    @overload
    def count_neighbors(
        self,
        /,
        other: cKDTree,
        r: npt.ArrayLike,
        p: AnyReal = 2.0,
        *,
        weights: _Weights,
        cumulative: bool = True,
    ) -> np.float64 | np.intp | npt.NDArray[np.float64]: ...

    #
    @overload
    def sparse_distance_matrix(
        self,
        /,
        other: cKDTree,
        max_distance: AnyReal,
        p: AnyReal = 2.0,
        output_type: L["dok_matrix"] = ...,
    ) -> dok_matrix: ...
    @overload
    def sparse_distance_matrix(
        self,
        /,
        other: cKDTree,
        max_distance: AnyReal,
        p: AnyReal = 2.0,
        *,
        output_type: L["coo_matrix"],
    ) -> coo_matrix: ...
    @overload
    def sparse_distance_matrix(
        self,
        /,
        other: cKDTree,
        max_distance: AnyReal,
        p: AnyReal = 2.0,
        *,
        output_type: L["dict"],
    ) -> dict[tuple[int, int], float]: ...
    @overload
    def sparse_distance_matrix(
        self,
        /,
        other: cKDTree,
        max_distance: AnyReal,
        p: AnyReal = 2.0,
        *,
        output_type: L["ndarray"],
    ) -> npt.NDArray[np.void]: ...
