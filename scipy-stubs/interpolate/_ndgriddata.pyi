from typing import Generic, Literal, TypeAlias, TypedDict, overload
from typing_extensions import TypeVar, Unpack, override

import numpy as np
import optype.numpy as onp
from scipy.spatial._ckdtree import cKDTree
from .interpnd import CloughTocher2DInterpolator, LinearNDInterpolator, NDInterpolatorBase

__all__ = ["CloughTocher2DInterpolator", "LinearNDInterpolator", "NearestNDInterpolator", "griddata"]

_SCT_co = TypeVar("_SCT_co", bound=np.float64 | np.complex128, default=np.float64 | np.complex128, covariant=True)

class _TreeOptions(TypedDict, total=False):
    leafsize: onp.ToJustInt
    compact_nodes: onp.ToBool
    copy_data: onp.ToBool
    balanced_tree: onp.ToBool
    boxsize: onp.ToFloatND | None

class _QueryOptions(TypedDict, total=False):
    eps: onp.ToFloat
    p: onp.ToFloat
    distance_upper_bound: onp.ToFloat
    workers: int

_Method: TypeAlias = Literal["nearest", "linear", "cubic"]

###

class NearestNDInterpolator(NDInterpolatorBase[_SCT_co], Generic[_SCT_co]):
    tree: cKDTree

    @overload
    def __init__(
        self: NearestNDInterpolator[np.float64],
        /,
        x: onp.ToFloat2D,
        y: onp.ToFloat1D,
        rescale: bool = False,
        tree_options: _TreeOptions | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        x: onp.ToFloat2D,
        y: onp.ToComplex1D,
        rescale: bool = False,
        tree_options: _TreeOptions | None = None,
    ) -> None: ...
    @override
    def __call__(self, /, *args: onp.ToFloatND, **query_options: Unpack[_QueryOptions]) -> onp.Array[onp.AtLeast1D, _SCT_co]: ...

@overload
def griddata(
    points: onp.ToFloat1D | onp.ToFloat2D,
    values: onp.ToFloat1D,
    xi: onp.ToFloat2D | tuple[onp.ToFloat1D | onp.ToFloat2D, ...],
    method: _Method = "linear",
    fill_value: onp.ToFloat = ...,  # np.nan
    rescale: onp.ToBool = False,
) -> onp.Array[onp.AtLeast1D, np.float64]: ...
@overload
def griddata(
    points: onp.ToFloat1D | onp.ToFloat2D,
    values: onp.ToComplex1D,
    xi: onp.ToFloat2D | tuple[onp.ToFloat1D | onp.ToFloat2D, ...],
    method: _Method = "linear",
    fill_value: onp.ToComplex = ...,  # np.nan
    rescale: onp.ToBool = False,
) -> onp.Array[onp.AtLeast1D, np.float64 | np.complex128]: ...
