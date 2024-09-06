from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, Literal, TypeAlias, TypedDict, overload
from typing_extensions import TypeVar, override

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
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
_LinkageArray: TypeAlias = onpt.Array[tuple[int, int], _SCT]
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

def int_floor(arr: onpt.AnyArray, xp: ModuleType) -> int: ...
def single(y: onpt.AnyArray) -> _LinkageArray: ...
def complete(y: onpt.AnyArray) -> _LinkageArray: ...
def average(y: onpt.AnyArray) -> _LinkageArray: ...
def weighted(y: onpt.AnyArray) -> _LinkageArray: ...
def centroid(y: onpt.AnyArray) -> _LinkageArray: ...
def median(y: onpt.AnyArray) -> _LinkageArray: ...
def ward(y: onpt.AnyArray) -> _LinkageArray: ...
def linkage(
    y: onpt.AnyArray,
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
    def __eq__(self, node: ClusterNode, /) -> bool: ...  # type: ignore[override]
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
    Z: onpt.AnyArray,
    n_clusters: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    height: Sequence[float] | npt.NDArray[np.integer[Any] | np.floating[Any]] | None = None,
) -> onpt.Array[tuple[int, int], np.int64]: ...
@overload
def to_tree(Z: onpt.AnyArray, rd: Literal[False] = False) -> ClusterNode: ...
@overload
def to_tree(Z: onpt.AnyArray, rd: Literal[True]) -> tuple[ClusterNode, list[ClusterNode]]: ...
def optimal_leaf_ordering(
    Z: onpt.AnyArray,
    y: onpt.AnyArray,
    metric: _MetricKind | _MetricCallback = "euclidean",
) -> _LinkageArray: ...
@overload
def cophenet(Z: onpt.AnyArray, Y: None = None) -> onpt.Array[tuple[int], np.float64]: ...
@overload
def cophenet(
    Z: onpt.AnyArray,
    Y: onpt.AnyArray,
) -> tuple[onpt.Array[tuple[int], np.float64], onpt.Array[tuple[int], np.float64]]: ...
def inconsistent(Z: onpt.AnyArray, d: int = 2) -> _LinkageArray: ...
def from_mlab_linkage(Z: onpt.AnyArray) -> _LinkageArray: ...
def to_mlab_linkage(Z: onpt.AnyArray) -> _LinkageArray: ...
def is_monotonic(Z: onpt.AnyArray) -> bool: ...
def is_valid_im(R: onpt.AnyArray, warning: bool = False, throw: bool = False, name: str | None = None) -> bool: ...
def is_valid_linkage(Z: onpt.AnyArray, warning: bool = False, throw: bool = False, name: str | None = None) -> bool: ...
def num_obs_linkage(Z: onpt.AnyArray) -> int: ...
def correspond(Z: onpt.AnyArray, Y: onpt.AnyArray) -> bool: ...
def fcluster(
    Z: onpt.AnyArray,
    t: float | np.floating[Any] | np.integer[Any],
    criterion: _ClusterCriterion = "inconsistent",
    depth: int = 2,
    R: onpt.AnyArray | None = None,
    monocrit: onpt.AnyArray | None = None,
) -> onpt.Array[tuple[int], np.int32]: ...
def fclusterdata(
    X: onpt.AnyArray,
    t: float | np.floating[Any] | np.integer[Any],
    criterion: _ClusterCriterion = "inconsistent",
    metric: _MetricKind | _MetricCallback = "euclidean",
    depth: int = 2,
    method: _LinkageMethod = "single",
    R: onpt.AnyArray | None = None,
) -> onpt.Array[tuple[int], np.int32]: ...
def leaves_list(Z: onpt.AnyArray) -> onpt.Array[tuple[int], np.int32]: ...
def set_link_color_palette(palette: list[str] | tuple[str, ...] | None) -> None: ...
def dendrogram(
    Z: onpt.AnyArray,
    p: int = 30,
    truncate_mode: Literal["lastp", "level"] | None = None,
    color_threshold: float | np.floating[Any] | None = None,
    get_leaves: bool = True,
    orientation: Literal["top", "bottom", "left", "right"] = "top",
    labels: onpt.AnyArray | None = None,
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
def is_isomorphic(T1: onpt.AnyArray, T2: onpt.AnyArray) -> bool: ...
def maxdists(Z: onpt.AnyArray) -> onpt.Array[tuple[int], np.float64]: ...
def maxinconsts(Z: onpt.AnyArray, R: onpt.AnyArray) -> onpt.Array[tuple[int], np.float64]: ...
def maxRstat(Z: onpt.AnyArray, R: onpt.AnyArray, i: int) -> onpt.Array[tuple[int], np.float64]: ...
def leaders(Z: onpt.AnyArray, T: onpt.AnyArray) -> tuple[onpt.Array[tuple[int], np.int32], onpt.Array[tuple[int], np.int32]]: ...
