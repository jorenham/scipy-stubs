from scipy._lib._util import normalize_axis_index as normalize_axis_index
from scipy._typing import Untyped

def correlate1d(
    input, weights, axis: int = -1, output: Untyped | None = None, mode: str = "reflect", cval: float = 0.0, origin: int = 0
) -> Untyped: ...
def convolve1d(
    input, weights, axis: int = -1, output: Untyped | None = None, mode: str = "reflect", cval: float = 0.0, origin: int = 0
) -> Untyped: ...
def gaussian_filter1d(
    input,
    sigma,
    axis: int = -1,
    order: int = 0,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
    *,
    radius: Untyped | None = None,
) -> Untyped: ...
def gaussian_filter(
    input,
    sigma,
    order: int = 0,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
    *,
    radius: Untyped | None = None,
    axes: Untyped | None = None,
) -> Untyped: ...
def prewitt(input, axis: int = -1, output: Untyped | None = None, mode: str = "reflect", cval: float = 0.0) -> Untyped: ...
def sobel(input, axis: int = -1, output: Untyped | None = None, mode: str = "reflect", cval: float = 0.0) -> Untyped: ...
def generic_laplace(
    input,
    derivative2,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    extra_arguments=(),
    extra_keywords: Untyped | None = None,
) -> Untyped: ...
def laplace(input, output: Untyped | None = None, mode: str = "reflect", cval: float = 0.0) -> Untyped: ...
def gaussian_laplace(
    input, sigma, output: Untyped | None = None, mode: str = "reflect", cval: float = 0.0, **kwargs
) -> Untyped: ...
def generic_gradient_magnitude(
    input,
    derivative,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    extra_arguments=(),
    extra_keywords: Untyped | None = None,
) -> Untyped: ...
def gaussian_gradient_magnitude(
    input, sigma, output: Untyped | None = None, mode: str = "reflect", cval: float = 0.0, **kwargs
) -> Untyped: ...
def correlate(
    input, weights, output: Untyped | None = None, mode: str = "reflect", cval: float = 0.0, origin: int = 0
) -> Untyped: ...
def convolve(
    input, weights, output: Untyped | None = None, mode: str = "reflect", cval: float = 0.0, origin: int = 0
) -> Untyped: ...
def uniform_filter1d(
    input, size, axis: int = -1, output: Untyped | None = None, mode: str = "reflect", cval: float = 0.0, origin: int = 0
) -> Untyped: ...
def uniform_filter(
    input,
    size: int = 3,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    *,
    axes: Untyped | None = None,
) -> Untyped: ...
def minimum_filter1d(
    input, size, axis: int = -1, output: Untyped | None = None, mode: str = "reflect", cval: float = 0.0, origin: int = 0
) -> Untyped: ...
def maximum_filter1d(
    input, size, axis: int = -1, output: Untyped | None = None, mode: str = "reflect", cval: float = 0.0, origin: int = 0
) -> Untyped: ...
def minimum_filter(
    input,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    *,
    axes: Untyped | None = None,
) -> Untyped: ...
def maximum_filter(
    input,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    *,
    axes: Untyped | None = None,
) -> Untyped: ...
def rank_filter(
    input,
    rank,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    *,
    axes: Untyped | None = None,
) -> Untyped: ...
def median_filter(
    input,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    *,
    axes: Untyped | None = None,
) -> Untyped: ...
def percentile_filter(
    input,
    percentile,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    *,
    axes: Untyped | None = None,
) -> Untyped: ...
def generic_filter1d(
    input,
    function,
    filter_size,
    axis: int = -1,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    extra_arguments=(),
    extra_keywords: Untyped | None = None,
) -> Untyped: ...
def generic_filter(
    input,
    function,
    size: Untyped | None = None,
    footprint: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    extra_arguments=(),
    extra_keywords: Untyped | None = None,
) -> Untyped: ...
