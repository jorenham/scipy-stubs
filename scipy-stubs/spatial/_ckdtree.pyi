from typing import Literal as L, TypeAlias, overload, type_check_only

import numpy as np
import optype.numpy as onp
from scipy.sparse import coo_matrix, dok_matrix

__all__ = ["cKDTree"]

_Weights: TypeAlias = onp.ToFloatND | tuple[onp.ToFloatND, onp.ToFloatND]

@type_check_only
class _CythonMixin:
    def __setstate_cython__(self, pyx_state: object, /) -> None: ...
    def __reduce_cython__(self, /) -> None: ...

class cKDTreeNode(_CythonMixin):
    @property
    def data_points(self, /) -> onp.ArrayND[np.float64]: ...
    @property
    def indices(self, /) -> onp.ArrayND[np.intp]: ...

    # These are read-only attributes in cython, which behave like properties
    @property
    def level(self, /) -> int: ...
    @property
    def split_dim(self, /) -> int: ...
    @property
    def children(self, /) -> int: ...
    @property
    def start_idx(self, /) -> int: ...
    @property
    def end_idx(self, /) -> int: ...
    @property
    def split(self, /) -> float: ...
    @property
    def lesser(self, /) -> cKDTreeNode | None: ...
    @property
    def greater(self, /) -> cKDTreeNode | None: ...

class cKDTree(_CythonMixin):
    @property
    def n(self, /) -> int: ...
    @property
    def m(self, /) -> int: ...
    @property
    def leafsize(self, /) -> int: ...
    @property
    def size(self, /) -> int: ...
    @property
    def tree(self, /) -> cKDTreeNode: ...

    # These are read-only attributes in cython, which behave like properties
    @property
    def data(self, /) -> onp.ArrayND[np.float64]: ...
    @property
    def maxes(self, /) -> onp.ArrayND[np.float64]: ...
    @property
    def mins(self, /) -> onp.ArrayND[np.float64]: ...
    @property
    def indices(self, /) -> onp.ArrayND[np.float64]: ...
    @property
    def boxsize(self, /) -> onp.ArrayND[np.float64] | None: ...

    #
    def __init__(
        self,
        /,
        data: onp.ToComplexND,
        leafsize: int = ...,
        compact_nodes: bool = ...,
        copy_data: bool = ...,
        balanced_tree: bool = ...,
        boxsize: onp.ToFloat2D | None = ...,
    ) -> None: ...

    #
    def query(
        self,
        /,
        x: onp.ToFloat1D,
        k: onp.ToInt | onp.ToInt1D = 1,
        eps: onp.ToFloat = 0.0,
        p: onp.ToFloat = 2.0,
        distance_upper_bound: float = ...,  # inf
        workers: int | None = None,
    ) -> tuple[float, np.intp] | tuple[onp.ArrayND[np.float64], onp.ArrayND[np.intp]]: ...

    #
    def query_ball_point(
        self,
        /,
        x: onp.ToFloatND,
        r: onp.ToFloat | onp.ToFloatND,
        p: onp.ToFloat = 2.0,
        eps: onp.ToFloat = 0.0,
        workers: int | None = None,
        return_sorted: bool | None = None,
        return_length: bool = False,
    ) -> list[int] | onp.ArrayND[np.object_]: ...

    #
    def query_ball_tree(
        self,
        /,
        other: cKDTree,
        r: onp.ToFloat,
        p: onp.ToFloat = 2.0,
        eps: onp.ToFloat = 0.0,
    ) -> list[list[int]]: ...

    #
    @overload
    def query_pairs(
        self,
        /,
        r: onp.ToFloat,
        p: onp.ToFloat = 2.0,
        eps: onp.ToFloat = 0,
        output_type: L["set"] = "set",
    ) -> set[tuple[int, int]]: ...
    @overload
    def query_pairs(
        self,
        /,
        r: onp.ToFloat,
        p: onp.ToFloat,
        eps: onp.ToFloat,
        output_type: L["ndarray"],
    ) -> onp.ArrayND[np.intp]: ...
    @overload
    def query_pairs(
        self,
        /,
        r: onp.ToFloat,
        p: onp.ToFloat = 2.0,
        eps: onp.ToFloat = 0,
        *,
        output_type: L["ndarray"],
    ) -> onp.ArrayND[np.intp]: ...

    #
    @overload
    def count_neighbors(
        self,
        /,
        other: cKDTree,
        r: onp.ToScalar,
        p: onp.ToFloat = 2.0,
        weights: tuple[None, None] | None = None,
        cumulative: bool = True,
    ) -> np.intp: ...
    @overload
    def count_neighbors(
        self,
        /,
        other: cKDTree,
        r: onp.ToScalar,
        p: onp.ToFloat,
        weights: _Weights,
        cumulative: bool = True,
    ) -> np.float64: ...
    @overload
    def count_neighbors(
        self,
        /,
        other: cKDTree,
        r: onp.ToScalar,
        p: onp.ToFloat = 2.0,
        *,
        weights: _Weights,
        cumulative: bool = True,
    ) -> np.float64: ...
    @overload
    def count_neighbors(
        self,
        /,
        other: cKDTree,
        r: onp.ToFloat | onp.ToFloatND,
        p: onp.ToFloat = 2.0,
        weights: tuple[None, None] | None = ...,
        cumulative: bool = True,
    ) -> np.float64 | np.intp | onp.ArrayND[np.intp]: ...
    @overload
    def count_neighbors(
        self,
        /,
        other: cKDTree,
        r: onp.ToFloat | onp.ToFloatND,
        p: onp.ToFloat,
        weights: _Weights,
        cumulative: bool = True,
    ) -> np.float64 | np.intp | onp.ArrayND[np.float64]: ...
    @overload
    def count_neighbors(
        self,
        /,
        other: cKDTree,
        r: onp.ToFloat | onp.ToFloatND,
        p: onp.ToFloat = 2.0,
        *,
        weights: _Weights,
        cumulative: bool = True,
    ) -> np.float64 | np.intp | onp.ArrayND[np.float64]: ...

    #
    @overload
    def sparse_distance_matrix(
        self,
        /,
        other: cKDTree,
        max_distance: onp.ToFloat,
        p: onp.ToFloat = 2.0,
        output_type: L["dok_matrix"] = ...,
    ) -> dok_matrix: ...
    @overload
    def sparse_distance_matrix(
        self,
        /,
        other: cKDTree,
        max_distance: onp.ToFloat,
        p: onp.ToFloat = 2.0,
        *,
        output_type: L["coo_matrix"],
    ) -> coo_matrix: ...
    @overload
    def sparse_distance_matrix(
        self,
        /,
        other: cKDTree,
        max_distance: onp.ToFloat,
        p: onp.ToFloat = 2.0,
        *,
        output_type: L["dict"],
    ) -> dict[tuple[int, int], float]: ...
    @overload
    def sparse_distance_matrix(
        self,
        /,
        other: cKDTree,
        max_distance: onp.ToFloat,
        p: onp.ToFloat = 2.0,
        *,
        output_type: L["ndarray"],
    ) -> onp.ArrayND[np.void]: ...
