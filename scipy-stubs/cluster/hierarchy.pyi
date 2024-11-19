from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, Literal, TypeAlias, TypedDict, overload
from typing_extensions import TypeVar, override

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from scipy._lib._disjoint_set import DisjointSet
from scipy.spatial.distance import _MetricCallback, _MetricKind

__all__ = [
    "ClusterNode",
    "DisjointSet",
    "average",
    "centroid",
    "complete",
    "cophenet",
    "correspond",
    "cut_tree",
    "dendrogram",
    "fcluster",
    "fclusterdata",
    "from_mlab_linkage",
    "inconsistent",
    "is_isomorphic",
    "is_monotonic",
    "is_valid_im",
    "is_valid_linkage",
    "leaders",
    "leaves_list",
    "linkage",
    "maxRstat",
    "maxdists",
    "maxinconsts",
    "median",
    "num_obs_linkage",
    "optimal_leaf_ordering",
    "set_link_color_palette",
    "single",
    "to_mlab_linkage",
    "to_tree",
    "ward",
    "weighted",
]

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=np.number[Any], default=np.float64)
_LinkageMethod: TypeAlias = Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"]
_LinkageArray: TypeAlias = onp.Array[tuple[int, int], _SCT]
_ClusterCriterion: TypeAlias = Literal["inconsistent", "distance", "maxclust", "monocrit", "maxclust_monocrit"]
_SortOrder: TypeAlias = Literal["ascending", "descending"]

# for the lack of a better type
_MatplotlibAxes: TypeAlias = object

class _DendrogramResult(TypedDict):
    color_list: list[str]
    icoord: list[list[int]]
    dcoord: list[list[int]]
    ivl: list[str]
    leaves: list[int] | None
    leaves_color_list: list[str]

class ClusterWarning(UserWarning): ...

def int_floor(arr: onp.AnyArray, xp: ModuleType) -> int: ...
def single(y: onp.AnyArray) -> _LinkageArray: ...
def complete(y: onp.AnyArray) -> _LinkageArray: ...
def average(y: onp.AnyArray) -> _LinkageArray: ...
def weighted(y: onp.AnyArray) -> _LinkageArray: ...
def centroid(y: onp.AnyArray) -> _LinkageArray: ...
def median(y: onp.AnyArray) -> _LinkageArray: ...
def ward(y: onp.AnyArray) -> _LinkageArray: ...
def linkage(
    y: onp.AnyArray,
    method: _LinkageMethod = "single",
    metric: _MetricKind | _MetricCallback = "euclidean",
    optimal_ordering: bool = False,
) -> _LinkageArray[np.int_ | np.float64 | np.complex128]: ...

class ClusterNode:
    id: int
    left: ClusterNode | None
    right: ClusterNode | None
    dist: float
    count: int
    def __init__(
        self,
        /,
        id: int,
        left: ClusterNode | None = None,
        right: ClusterNode | None = None,
        dist: float = 0,
        count: int = 1,
    ) -> None: ...
    def __lt__(self, node: ClusterNode, /) -> bool: ...
    def __gt__(self, node: ClusterNode, /) -> bool: ...
    @override
    def __eq__(self, node: ClusterNode, /) -> bool: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def get_id(self, /) -> int: ...
    def get_count(self, /) -> int: ...
    def get_left(self, /) -> ClusterNode: ...
    def get_right(self, /) -> ClusterNode: ...
    def is_leaf(self, /) -> bool: ...
    @overload
    def pre_order(self, /, func: Callable[[ClusterNode], int] = ...) -> list[int]: ...
    @overload
    def pre_order(self, /, func: Callable[[ClusterNode], _T]) -> list[_T]: ...

def cut_tree(
    Z: onp.AnyArray,
    n_clusters: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    height: Sequence[float] | npt.NDArray[np.integer[Any] | np.floating[Any]] | None = None,
) -> onp.Array[tuple[int, int], np.int64]: ...
@overload
def to_tree(Z: onp.AnyArray, rd: Literal[False] = False) -> ClusterNode: ...
@overload
def to_tree(Z: onp.AnyArray, rd: Literal[True]) -> tuple[ClusterNode, list[ClusterNode]]: ...
def optimal_leaf_ordering(
    Z: onp.AnyArray,
    y: onp.AnyArray,
    metric: _MetricKind | _MetricCallback = "euclidean",
) -> _LinkageArray: ...
@overload
def cophenet(Z: onp.AnyArray, Y: None = None) -> onp.Array[tuple[int], np.float64]: ...
@overload
def cophenet(
    Z: onp.AnyArray,
    Y: onp.AnyArray,
) -> tuple[onp.Array[tuple[int], np.float64], onp.Array[tuple[int], np.float64]]: ...
def inconsistent(Z: onp.AnyArray, d: int = 2) -> _LinkageArray: ...
def from_mlab_linkage(Z: onp.AnyArray) -> _LinkageArray: ...
def to_mlab_linkage(Z: onp.AnyArray) -> _LinkageArray: ...
def is_monotonic(Z: onp.AnyArray) -> bool: ...
def is_valid_im(R: onp.AnyArray, warning: bool = False, throw: bool = False, name: str | None = None) -> bool: ...
def is_valid_linkage(Z: onp.AnyArray, warning: bool = False, throw: bool = False, name: str | None = None) -> bool: ...
def num_obs_linkage(Z: onp.AnyArray) -> int: ...
def correspond(Z: onp.AnyArray, Y: onp.AnyArray) -> bool: ...
def fcluster(
    Z: onp.AnyArray,
    t: float | np.floating[Any] | np.integer[Any],
    criterion: _ClusterCriterion = "inconsistent",
    depth: int = 2,
    R: onp.AnyArray | None = None,
    monocrit: onp.AnyArray | None = None,
) -> onp.Array[tuple[int], np.int32]: ...
def fclusterdata(
    X: onp.AnyArray,
    t: float | np.floating[Any] | np.integer[Any],
    criterion: _ClusterCriterion = "inconsistent",
    metric: _MetricKind | _MetricCallback = "euclidean",
    depth: int = 2,
    method: _LinkageMethod = "single",
    R: onp.AnyArray | None = None,
) -> onp.Array[tuple[int], np.int32]: ...
def leaves_list(Z: onp.AnyArray) -> onp.Array[tuple[int], np.int32]: ...
def set_link_color_palette(palette: list[str] | tuple[str, ...] | None) -> None: ...
def dendrogram(
    Z: onp.AnyArray,
    p: int = 30,
    truncate_mode: Literal["lastp", "level"] | None = None,
    color_threshold: float | np.floating[Any] | None = None,
    get_leaves: bool = True,
    orientation: Literal["top", "bottom", "left", "right"] = "top",
    labels: onp.AnyArray | None = None,
    count_sort: _SortOrder | bool = False,
    distance_sort: _SortOrder | bool = False,
    show_leaf_counts: bool = True,
    no_plot: bool = False,
    no_labels: bool = False,
    leaf_font_size: float | np.floating[Any] | None = None,
    leaf_rotation: float | np.floating[Any] | None = None,
    leaf_label_func: Callable[[int], str] | None = None,
    show_contracted: bool = False,
    link_color_func: Callable[[int], str] | None = None,
    ax: _MatplotlibAxes | None = None,
    above_threshold_color: str = "C0",
) -> _DendrogramResult: ...
def is_isomorphic(T1: onp.AnyArray, T2: onp.AnyArray) -> bool: ...
def maxdists(Z: onp.AnyArray) -> onp.Array[tuple[int], np.float64]: ...
def maxinconsts(Z: onp.AnyArray, R: onp.AnyArray) -> onp.Array[tuple[int], np.float64]: ...
def maxRstat(Z: onp.AnyArray, R: onp.AnyArray, i: int) -> onp.Array[tuple[int], np.float64]: ...
def leaders(Z: onp.AnyArray, T: onp.AnyArray) -> tuple[onp.Array[tuple[int], np.int32], onp.Array[tuple[int], np.int32]]: ...
