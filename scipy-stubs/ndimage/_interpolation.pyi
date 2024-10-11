from collections.abc import Callable
from typing import Concatenate, Literal, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt
from ._typing import (
    _BoolValueIn,
    _ComplexArrayOut,
    _FloatArrayIn,
    _FloatArrayOut,
    _FloatMatrixIn,
    _FloatValueIn,
    _FloatVectorIn,
    _IntValueIn,
    _ScalarArrayIn,
    _ScalarValueIn,
)

__all__ = [
    "affine_transform",
    "geometric_transform",
    "map_coordinates",
    "rotate",
    "shift",
    "spline_filter",
    "spline_filter1d",
    "zoom",
]

_SCT = TypeVar("_SCT", bound=np.generic)
_Order: TypeAlias = Literal[0, 1, 2, 3, 4, 5]
_Mode: TypeAlias = Literal["reflect", "grid-mirror", "constant", "grid-constant", "nearest", "mirror", "wrap", "grid-wrap"]
_MappingFunc: TypeAlias = Callable[Concatenate[tuple[int, ...], ...], tuple[_FloatValueIn, ...]]

#
@overload
def spline_filter1d(
    input: npt.ArrayLike,
    order: _Order = 3,
    axis: _IntValueIn = -1,
    output: type[float | np.float64] = ...,
    mode: _Mode = "mirror",
) -> npt.NDArray[np.float64]: ...
@overload
def spline_filter1d(
    input: npt.ArrayLike,
    order: _Order = 3,
    axis: _IntValueIn = -1,
    output: type[complex] = ...,
    mode: _Mode = "mirror",
) -> npt.NDArray[np.complex128 | np.float64]: ...
@overload
def spline_filter1d(
    input: npt.ArrayLike,
    order: _Order,
    axis: _IntValueIn,
    output: npt.NDArray[_SCT] | type[_SCT],
    mode: _Mode = "mirror",
) -> npt.NDArray[_SCT]: ...
@overload
def spline_filter1d(
    input: npt.ArrayLike,
    order: _Order = 3,
    axis: _IntValueIn = -1,
    *,
    output: npt.NDArray[_SCT] | type[_SCT],
    mode: _Mode = "mirror",
) -> npt.NDArray[_SCT]: ...

#
@overload
def spline_filter(
    input: npt.ArrayLike,
    order: _Order = 3,
    output: type[float | np.float64] = ...,
    mode: _Mode = "mirror",
) -> npt.NDArray[np.float64]: ...
@overload
def spline_filter(
    input: npt.ArrayLike,
    order: _Order = 3,
    output: type[complex] = ...,
    mode: _Mode = "mirror",
) -> npt.NDArray[np.complex128 | np.float64]: ...
@overload
def spline_filter(
    input: npt.ArrayLike,
    order: _Order,
    output: npt.NDArray[_SCT] | type[_SCT],
    mode: _Mode = "mirror",
) -> npt.NDArray[_SCT]: ...
@overload
def spline_filter(
    input: npt.ArrayLike,
    order: _Order = 3,
    *,
    output: npt.NDArray[_SCT] | type[_SCT],
    mode: _Mode = "mirror",
) -> npt.NDArray[_SCT]: ...

#
@overload
def geometric_transform(
    input: _FloatArrayIn,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None = None,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
    extra_arguments: tuple[object, ...] = ...,
    extra_keywords: dict[str, object] | None = ...,
) -> _FloatArrayOut: ...
@overload
def geometric_transform(
    input: _ScalarArrayIn,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None = None,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
    extra_arguments: tuple[object, ...] = ...,
    extra_keywords: dict[str, object] | None = ...,
) -> _ComplexArrayOut: ...
@overload
def geometric_transform(
    input: npt.ArrayLike,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None,
    output: npt.NDArray[_SCT] | type[_SCT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
    extra_arguments: tuple[object, ...] = ...,
    extra_keywords: dict[str, object] | None = ...,
) -> npt.NDArray[_SCT]: ...
@overload
def geometric_transform(
    input: npt.ArrayLike,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: npt.NDArray[_SCT] | type[_SCT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
    extra_arguments: tuple[object, ...] = ...,
    extra_keywords: dict[str, object] | None = ...,
) -> npt.NDArray[_SCT]: ...
@overload
def geometric_transform(
    input: npt.ArrayLike,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: type[int],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
    extra_arguments: tuple[object, ...] = ...,
    extra_keywords: dict[str, object] | None = ...,
) -> npt.NDArray[np.int_]: ...
@overload
def geometric_transform(
    input: npt.ArrayLike,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: type[float],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
    extra_arguments: tuple[object, ...] = ...,
    extra_keywords: dict[str, object] | None = ...,
) -> npt.NDArray[np.float64 | np.int_]: ...
@overload
def geometric_transform(
    input: npt.ArrayLike,
    mapping: _MappingFunc,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: type[complex],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
    extra_arguments: tuple[object, ...] = ...,
    extra_keywords: dict[str, object] | None = ...,
) -> npt.NDArray[np.complex128 | np.float64 | np.int_]: ...

#
@overload
def map_coordinates(
    input: _FloatArrayIn,
    coordinates: _FloatArrayIn,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> _FloatArrayOut: ...
@overload
def map_coordinates(
    input: _ScalarArrayIn,
    coordinates: _FloatArrayIn,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> _ComplexArrayOut: ...
@overload
def map_coordinates(
    input: npt.ArrayLike,
    coordinates: _FloatArrayIn,
    output: npt.NDArray[_SCT] | type[_SCT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[_SCT]: ...
@overload
def map_coordinates(
    input: npt.ArrayLike,
    coordinates: _FloatArrayIn,
    output: type[bool],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[np.bool_]: ...
@overload
def map_coordinates(
    input: npt.ArrayLike,
    coordinates: _FloatArrayIn,
    output: type[int],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[np.int_ | np.bool_]: ...
@overload
def map_coordinates(
    input: npt.ArrayLike,
    coordinates: _FloatArrayIn,
    output: type[float],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[np.float64 | np.int_ | np.bool_]: ...
@overload
def map_coordinates(
    input: npt.ArrayLike,
    coordinates: _FloatArrayIn,
    output: type[complex],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[np.complex128 | np.float64 | np.int_ | np.bool_]: ...

#
@overload
def affine_transform(
    input: _FloatArrayIn,
    matrix: _FloatMatrixIn,
    offset: _FloatVectorIn = 0.0,
    output_shape: tuple[int, ...] | None = None,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> _FloatArrayOut: ...
@overload
def affine_transform(
    input: _ScalarArrayIn,
    matrix: _FloatMatrixIn,
    offset: _FloatVectorIn = 0.0,
    output_shape: tuple[int, ...] | None = None,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> _ComplexArrayOut: ...
@overload
def affine_transform(
    input: npt.ArrayLike,
    matrix: _FloatMatrixIn,
    offset: _FloatVectorIn = 0.0,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: npt.NDArray[_SCT] | type[_SCT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[_SCT]: ...
@overload
def affine_transform(
    input: npt.ArrayLike,
    matrix: _FloatMatrixIn,
    offset: _FloatVectorIn = 0.0,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: type[int],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[np.int_]: ...
@overload
def affine_transform(
    input: npt.ArrayLike,
    matrix: _FloatMatrixIn,
    offset: _FloatVectorIn = 0.0,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: type[float],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[np.float64 | np.int_]: ...
@overload
def affine_transform(
    input: npt.ArrayLike,
    matrix: _FloatMatrixIn,
    offset: _FloatVectorIn = 0.0,
    output_shape: tuple[int, ...] | None = None,
    *,
    output: type[complex],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[np.complex128 | np.float64 | np.int_]: ...

#
@overload
def shift(
    input: _FloatArrayIn,
    shift: _FloatArrayIn,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> _FloatArrayOut: ...
@overload
def shift(
    input: _ScalarArrayIn,
    shift: _FloatArrayIn,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> _ComplexArrayOut: ...
@overload
def shift(
    input: npt.ArrayLike,
    shift: _FloatArrayIn,
    output: npt.NDArray[_SCT] | type[_SCT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[_SCT]: ...
@overload
def shift(
    input: npt.ArrayLike,
    shift: _FloatArrayIn,
    output: type[int],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[np.int_]: ...
@overload
def shift(
    input: npt.ArrayLike,
    shift: _FloatArrayIn,
    output: type[float],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[np.float64 | np.int_]: ...
@overload
def shift(
    input: npt.ArrayLike,
    shift: _FloatArrayIn,
    output: type[complex],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[np.complex128 | np.float64 | np.int_]: ...

#
@overload
def zoom(
    input: _FloatArrayIn,
    zoom: _FloatArrayIn,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
    *,
    grid_mode: bool = False,
) -> _FloatArrayOut: ...
@overload
def zoom(
    input: _ScalarArrayIn,
    zoom: _FloatArrayIn,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
    *,
    grid_mode: bool = False,
) -> _ComplexArrayOut: ...
@overload
def zoom(
    input: npt.ArrayLike,
    zoom: _FloatArrayIn,
    output: npt.NDArray[_SCT] | type[_SCT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
    *,
    grid_mode: bool = False,
) -> npt.NDArray[_SCT]: ...
@overload
def zoom(
    input: npt.ArrayLike,
    zoom: _FloatArrayIn,
    output: type[int],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
    *,
    grid_mode: bool = False,
) -> npt.NDArray[np.int_]: ...
@overload
def zoom(
    input: npt.ArrayLike,
    zoom: _FloatArrayIn,
    output: type[float],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
    *,
    grid_mode: bool = False,
) -> npt.NDArray[np.float64 | np.int_]: ...
@overload
def zoom(
    input: npt.ArrayLike,
    zoom: _FloatArrayIn,
    output: type[complex],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
    *,
    grid_mode: bool = False,
) -> npt.NDArray[np.complex128 | np.float64 | np.int_]: ...

#
@overload
def rotate(
    input: _FloatArrayIn,
    angle: _FloatValueIn,
    axes: tuple[_IntValueIn, _IntValueIn] = (1, 0),
    reshape: bool = True,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _FloatValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> _FloatArrayOut: ...
@overload
def rotate(
    input: _ScalarArrayIn,
    angle: _FloatValueIn,
    axes: tuple[_IntValueIn, _IntValueIn] = (1, 0),
    reshape: bool = True,
    output: None = None,
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> _ComplexArrayOut: ...
@overload
def rotate(
    input: npt.ArrayLike,
    angle: _FloatValueIn,
    axes: tuple[_IntValueIn, _IntValueIn] = (1, 0),
    reshape: bool = True,
    *,
    output: npt.NDArray[_SCT] | type[_SCT],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[_SCT]: ...
@overload
def rotate(
    input: npt.ArrayLike,
    angle: _FloatValueIn,
    axes: tuple[_IntValueIn, _IntValueIn] = (1, 0),
    reshape: bool = True,
    *,
    output: type[int],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[np.int_]: ...
@overload
def rotate(
    input: npt.ArrayLike,
    angle: _FloatValueIn,
    axes: tuple[_IntValueIn, _IntValueIn] = (1, 0),
    reshape: bool = True,
    *,
    output: type[float],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[np.float64 | np.int_]: ...
@overload
def rotate(
    input: npt.ArrayLike,
    angle: _FloatValueIn,
    axes: tuple[_IntValueIn, _IntValueIn] = (1, 0),
    reshape: bool = True,
    *,
    output: type[complex],
    order: _Order = 3,
    mode: _Mode = "constant",
    cval: _ScalarValueIn = 0.0,
    prefilter: _BoolValueIn = True,
) -> npt.NDArray[np.complex128 | np.float64 | np.int_]: ...
