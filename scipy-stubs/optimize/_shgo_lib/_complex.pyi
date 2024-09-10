from collections.abc import Generator, Sequence
from typing import Any, TypeAlias

import numpy as np
import optype as op
import optype.numpy as onpt
from scipy._typing import Untyped, UntypedCallable, UntypedTuple
from ._vertex import VertexBase, VertexCacheBase

_Location: TypeAlias = Sequence[float]
_Bounds: TypeAlias = Sequence[tuple[float, float]]
_Constraints: TypeAlias = dict[str, object] | Sequence[dict[str, object]]  # TODO: TypedDict
_Symmetry: TypeAlias = op.CanGetitem[int, op.CanIndex]
_CyclicProduct: TypeAlias = Generator[tuple[float, ...], None, tuple[float, ...]]

class Complex:
    dim: int
    domain: _Bounds
    bounds: _Bounds
    symmetry: _Symmetry
    sfield: UntypedCallable
    sfield_args: UntypedTuple
    min_cons: _Constraints  # only set if `constraints` is not None
    g_cons: UntypedTuple | None
    g_args: UntypedTuple | None
    gen: int
    perm_cycle: int

    H: list[Untyped]
    V: VertexCacheBase
    V_non_symm: list[VertexBase]
    origin: list[float]
    supremum: list[float]
    cp: _CyclicProduct
    triangulated_vectors: list[tuple[tuple[float, ...], tuple[float, ...]]]
    rls: Untyped
    def __init__(
        self,
        /,
        dim: int,
        domain: _Bounds | None = None,
        sfield: UntypedCallable | None = None,
        sfield_args: UntypedTuple = (),
        symmetry: _Symmetry | None = None,
        constraints: _Constraints | None = None,
        workers: int = 1,
    ) -> None: ...
    def __call__(self, /) -> Untyped: ...
    def cyclic_product(
        self,
        /,
        bounds: _Bounds,
        origin: _Location,
        supremum: _Location,
        centroid: bool = True,
    ) -> _CyclicProduct: ...
    def triangulate(
        self,
        /,
        n: int | None = None,
        symmetry: _Symmetry | None = None,
        centroid: bool = True,
        printout: bool = False,
    ) -> None: ...
    def refine(self, /, n: int = 1) -> None: ...
    def refine_all(self, /, centroids: bool = True) -> None: ...
    def refine_local_space(
        self,
        /,
        origin: _Location,
        supremum: _Location,
        bounds: _Bounds,
        centroid: int = 1,
    ) -> Generator[VertexBase | tuple[float, ...], None, None]: ...
    def refine_star(self, /, v: VertexBase) -> None: ...
    def split_edge(self, /, v1: VertexBase, v2: VertexBase) -> VertexBase: ...
    def vpool(self, /, origin: _Location, supremum: _Location) -> set[VertexBase]: ...
    def vf_to_vv(self, /, vertices: Sequence[VertexBase], simplices: Sequence[Untyped]) -> None: ...
    def connect_vertex_non_symm(
        self,
        /,
        v_x: tuple[float | np.floating[Any]] | onpt.Array[tuple[int], np.floating[Any]],
        near: set[VertexBase] | list[VertexBase] | None = None,
    ) -> bool | None: ...
    def in_simplex(
        self,
        /,
        S: Sequence[float | np.floating[Any]] | onpt.Array[tuple[int, ...], np.floating[Any]],
        v_x: onpt.Array[tuple[int], np.floating[Any]],
        A_j0: onpt.Array[tuple[int, ...], np.floating[Any]] | None = None,
    ) -> Untyped: ...
    def deg_simplex(
        self,
        /,
        S: onpt.Array[tuple[int, ...], np.floating[Any]],
        proj: onpt.Array[tuple[int, ...], np.floating[Any]] | None = None,
    ) -> Untyped: ...
