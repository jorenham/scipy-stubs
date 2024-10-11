from collections.abc import Callable, Sequence
from typing import Concatenate, Literal, TypeAlias, TypedDict, type_check_only
from typing_extensions import Unpack

import numpy as np
import numpy.typing as npt
from scipy._lib._ccallback import LowLevelCallable
from ._typing import (
    _FloatArrayIn,
    _FloatArrayOut,
    _FloatValueIn,
    _FloatVectorIn,
    _IntArrayIn,
    _ScalarArrayIn,
    _ScalarArrayOut,
    _ScalarValueIn,
    _ScalarValueOut,
)

__all__ = [
    "convolve",
    "convolve1d",
    "correlate",
    "correlate1d",
    "gaussian_filter",
    "gaussian_filter1d",
    "gaussian_gradient_magnitude",
    "gaussian_laplace",
    "generic_filter",
    "generic_filter1d",
    "generic_gradient_magnitude",
    "generic_laplace",
    "laplace",
    "maximum_filter",
    "maximum_filter1d",
    "median_filter",
    "minimum_filter",
    "minimum_filter1d",
    "percentile_filter",
    "prewitt",
    "rank_filter",
    "sobel",
    "uniform_filter",
    "uniform_filter1d",
]

_Mode: TypeAlias = Literal["reflect", "constant", "nearest", "mirror", "wrap", "grid-constant", "grid-mirror", "grid-wrap"]
_Modes: TypeAlias = _Mode | Sequence[_Mode]
_Ints: TypeAlias = int | Sequence[int]

# TODO: allow passing dtype-likes to `output`

#
def laplace(
    input: _ScalarArrayIn,
    output: _ScalarArrayOut | None = None,
    mode: _Modes = "reflect",
    cval: _ScalarValueIn = 0.0,
) -> _ScalarArrayOut: ...

#
def prewitt(
    input: _ScalarArrayIn,
    axis: int = -1,
    output: _ScalarArrayOut | None = None,
    mode: _Modes = "reflect",
    cval: _ScalarValueIn = 0.0,
) -> _ScalarArrayOut: ...
def sobel(
    input: _ScalarArrayIn,
    axis: int = -1,
    output: _ScalarArrayOut | None = None,
    mode: _Modes = "reflect",
    cval: _ScalarValueIn = 0.0,
) -> _ScalarArrayOut: ...

#
def correlate1d(
    input: _ScalarArrayIn,
    weights: _FloatVectorIn,
    axis: int = -1,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: int = 0,
) -> _ScalarArrayOut: ...
def convolve1d(
    input: _ScalarArrayIn,
    weights: _FloatVectorIn,
    axis: int = -1,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: int = 0,
) -> _ScalarArrayOut: ...

#
def correlate(
    input: _ScalarArrayIn,
    weights: _FloatArrayIn,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Ints = 0,
) -> _ScalarArrayOut: ...
def convolve(
    input: _ScalarArrayIn,
    weights: _FloatArrayIn,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Ints = 0,
) -> _ScalarArrayOut: ...

#

@type_check_only
class _GaussianKwargs(TypedDict, total=False):
    truncate: float
    radius: _Ints
    axes: tuple[int, ...]

def gaussian_laplace(
    input: _ScalarArrayIn,
    sigma: _FloatArrayIn,
    output: _ScalarArrayOut | None = None,
    mode: _Modes = "reflect",
    cval: _ScalarValueIn = 0.0,
    **kwargs: Unpack[_GaussianKwargs],
) -> _ScalarArrayOut: ...
def gaussian_gradient_magnitude(
    input: _ScalarArrayIn,
    sigma: _FloatArrayIn,
    output: _ScalarArrayOut | None = None,
    mode: _Modes = "reflect",
    cval: _ScalarValueIn = 0.0,
    **kwargs: Unpack[_GaussianKwargs],
) -> _ScalarArrayOut: ...

#
def gaussian_filter1d(
    input: _ScalarArrayIn,
    sigma: _FloatValueIn,
    axis: int = -1,
    order: int = 0,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    truncate: float = 4.0,
    *,
    radius: int | None = None,
) -> _ScalarArrayOut: ...

#
def gaussian_filter(
    input: _ScalarArrayIn,
    sigma: _FloatArrayIn,
    order: _Ints = 0,
    output: _ScalarArrayOut | None = None,
    mode: _Modes = "reflect",
    cval: _ScalarValueIn = 0.0,
    truncate: float = 4.0,
    *,
    radius: _Ints | None = None,
    axes: tuple[int, ...] | None = None,
) -> _ScalarArrayOut: ...

#
_Derivative: TypeAlias = Callable[
    # (input, axis, output, mode, cval, *extra_arguments, **extra_keywords)
    Concatenate[_ScalarArrayOut, int, np.dtype[_ScalarValueOut], _Mode, _ScalarValueIn, ...],
    _ScalarArrayOut,
]

def generic_laplace(
    input: _ScalarArrayIn,
    derivative2: _Derivative,
    output: _ScalarArrayOut | None = None,
    mode: _Modes = "reflect",
    cval: _ScalarValueIn = 0.0,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> _ScalarArrayOut: ...
def generic_gradient_magnitude(
    input: _ScalarArrayIn,
    derivative: _Derivative,
    output: _ScalarArrayOut | None = None,
    mode: _Modes = "reflect",
    cval: _ScalarValueIn = 0.0,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> _ScalarArrayOut: ...

#
_FilterFunc1D: TypeAlias = Callable[Concatenate[npt.NDArray[np.float64], npt.NDArray[np.float64], ...], None]

def generic_filter1d(
    input: _FloatArrayIn,
    function: _FilterFunc1D | LowLevelCallable,
    filter_size: float,
    axis: int = -1,
    output: _FloatArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: int = 0,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> _FloatArrayOut: ...

#
_FilterFuncND: TypeAlias = Callable[
    Concatenate[npt.NDArray[np.float64], ...],
    _ScalarValueIn | _ScalarValueOut | _ScalarArrayOut,
]

def generic_filter(
    input: _FloatArrayIn,
    function: _FilterFuncND | LowLevelCallable,
    size: int | tuple[int, ...] | None = None,
    footprint: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Modes = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Ints = 0,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> _ScalarArrayOut: ...

#
def uniform_filter1d(
    input: _ScalarArrayIn,
    size: int,
    axis: int = -1,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: int = 0,
) -> _ScalarArrayOut: ...
def minimum_filter1d(
    input: _ScalarArrayIn,
    size: int,
    axis: int = -1,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: int = 0,
) -> _ScalarArrayOut: ...
def maximum_filter1d(
    input: _ScalarArrayIn,
    size: int,
    axis: int = -1,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: int = 0,
) -> _ScalarArrayOut: ...

#
def uniform_filter(
    input: _ScalarArrayIn,
    size: int | tuple[int, ...] = 3,
    output: _ScalarArrayOut | None = None,
    mode: _Modes = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Ints = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> _ScalarArrayOut: ...

#
def minimum_filter(
    input: _ScalarArrayIn,
    size: int | tuple[int, ...] | None = None,
    footprint: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Modes = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Ints = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> _ScalarArrayOut: ...
def maximum_filter(
    input: _ScalarArrayIn,
    size: int | tuple[int, ...] | None = None,
    footprint: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Modes = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Ints = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> _ScalarArrayOut: ...

#
def median_filter(
    input: _ScalarArrayIn,
    size: int | tuple[int, ...] | None = None,
    footprint: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Ints = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> _ScalarArrayOut: ...

#
def rank_filter(
    input: _ScalarArrayIn,
    rank: int,
    size: int | tuple[int, ...] | None = None,
    footprint: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Ints = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> _ScalarArrayOut: ...
def percentile_filter(
    input: _ScalarArrayIn,
    percentile: _FloatValueIn,
    size: int | tuple[int, ...] | None = None,
    footprint: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Ints = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> _ScalarArrayOut: ...
