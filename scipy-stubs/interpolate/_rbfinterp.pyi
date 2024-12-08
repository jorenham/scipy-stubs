from typing import Generic, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp

__all__ = ["RBFInterpolator"]

_Kernel: TypeAlias = Literal[
    "thin_plate_spline",
    "linear",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian",
]

_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int, ...], default=onp.AtLeast1D, covariant=True)
_SCT_co = TypeVar("_SCT_co", bound=np.float64 | np.complex128, default=np.float64 | np.complex128, covariant=True)

###

class RBFInterpolator(Generic[_ShapeT_co, _SCT_co]):
    y: onp.Array2D[np.float64]
    d: onp.Array[_ShapeT_co, np.float64]
    d_shape: _ShapeT_co
    d_dtype: type[float | complex]
    neighbors: int
    smoothing: onp.Array1D[np.float64]
    kernel: _Kernel
    epsilon: float
    powers: int

    @overload
    def __init__(
        self: RBFInterpolator[tuple[int], np.float64],
        /,
        y: onp.ToFloat2D,
        d: onp.ToFloatStrict1D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[tuple[int]],
        /,
        y: onp.ToFloat2D,
        d: onp.ToComplexStrict1D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[tuple[int, int], np.float64],
        /,
        y: onp.ToFloat2D,
        d: onp.ToFloatStrict2D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[tuple[int, int]],
        /,
        y: onp.ToFloat2D,
        d: onp.ToComplexStrict2D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[tuple[int, int, int], np.float64],
        /,
        y: onp.ToFloat2D,
        d: onp.ToFloatStrict3D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[tuple[int, int, int]],
        /,
        y: onp.ToFloat2D,
        d: onp.ToComplexStrict3D,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: RBFInterpolator[onp.AtLeast1D, np.float64],
        /,
        y: onp.ToFloat2D,
        d: onp.ToFloatND,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        y: onp.ToFloat2D,
        d: onp.ToComplexND,
        neighbors: onp.ToJustInt | None = None,
        smoothing: onp.ToFloat | onp.ToFloat1D = 0.0,
        kernel: _Kernel = "thin_plate_spline",
        epsilon: onp.ToFloat | None = None,
        degree: onp.ToJustInt | None = None,
    ) -> None: ...

    # TODO(jorenham): Return `onp.Array[tuple[int, Unpack[_ShapeT_co]], _SCT_co]` once mypy supports it (if ever)
    def __call__(self, /, x: onp.ToFloat2D) -> onp.ArrayND[_SCT_co]: ...
