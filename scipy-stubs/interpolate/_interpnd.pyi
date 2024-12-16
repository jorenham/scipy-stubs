# scipy/interpolate/interpnd.pyx

from typing import Generic, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy.spatial._qhull import Delaunay, DelaunayInfo_t

_SCT_co = TypeVar("_SCT_co", bound=np.float64 | np.complex128, default=np.float64 | np.complex128, covariant=True)

###

class GradientEstimationWarning(Warning): ...

class NDInterpolatorBase(Generic[_SCT_co]):
    points: onp.Array2D[np.float64]
    values: onp.ArrayND[_SCT_co] | None
    is_complex: bool
    scale: onp.Array1D[np.float64] | None
    offset: onp.Array1D[np.float64]  # only if rescale=True

    @overload
    def __init__(
        self: NDInterpolatorBase[np.float64],
        /,
        points: onp.ToFloat2D | Delaunay,
        values: onp.ToFloatND,
        fill_value: onp.ToFloat = ...,  # np.nan
        ndim: int | None = None,
        rescale: bool = False,
        need_contiguous: bool = True,
        need_values: bool = True,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        points: onp.ToFloat2D | Delaunay,
        values: onp.ToComplexND | None,
        fill_value: onp.ToComplex = ...,  # np.nan
        ndim: int | None = None,
        rescale: bool = False,
        need_contiguous: bool = True,
        need_values: bool = True,
    ) -> None: ...
    def __call__(self, /, *args: onp.ToFloatND) -> onp.Array[onp.AtLeast1D, _SCT_co]: ...

class LinearNDInterpolator(NDInterpolatorBase[_SCT_co], Generic[_SCT_co]):
    @overload
    def __init__(
        self: LinearNDInterpolator[np.float64],
        /,
        points: onp.ToFloat2D | Delaunay,
        values: onp.ToFloatND,
        fill_value: onp.ToFloat = ...,  # np.nan
        rescale: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        points: onp.ToFloat2D | Delaunay,
        values: onp.ToComplexND,
        fill_value: onp.ToComplex = ...,  # np.nan
        rescale: bool = False,
    ) -> None: ...

class CloughTocher2DInterpolator(NDInterpolatorBase[_SCT_co], Generic[_SCT_co]):
    @overload
    def __init__(
        self: CloughTocher2DInterpolator[np.float64],
        /,
        points: onp.ToFloat2D | Delaunay,
        values: onp.ToFloatND,
        fill_value: onp.ToFloat = ...,  # np.nan
        tol: onp.ToFloat = 1e-06,
        maxiter: int = 400,
        rescale: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        points: onp.ToFloat2D | Delaunay,
        values: onp.ToComplexND,
        fill_value: onp.ToComplex = ...,  # np.nan
        tol: onp.ToFloat = 1e-06,
        maxiter: int = 400,
        rescale: bool = False,
    ) -> None: ...

@overload
def estimate_gradients_2d_global(
    tri: DelaunayInfo_t,
    y: onp.ToFloat1D | onp.ToFloat2D,
    maxiter: onp.ToJustInt = 400,
    tol: float = 1e-6,
) -> onp.Array3D[np.float64]: ...
@overload
def estimate_gradients_2d_global(
    tri: DelaunayInfo_t,
    y: onp.ToComplex1D | onp.ToComplex2D,
    maxiter: onp.ToJustInt = 400,
    tol: float = 1e-6,
) -> onp.Array3D[np.float64] | onp.Array3D[np.complex128]: ...
