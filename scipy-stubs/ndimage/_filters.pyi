from scipy._typing import Untyped, UntypedCallable

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

def correlate1d(
    input: Untyped,
    weights: Untyped,
    axis: int = -1,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def convolve1d(
    input: Untyped,
    weights: Untyped,
    axis: int = -1,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def gaussian_filter1d(
    input: Untyped,
    sigma: Untyped,
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
    input: Untyped,
    sigma: Untyped,
    order: int = 0,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
    *,
    radius: Untyped | None = None,
    axes: Untyped | None = None,
) -> Untyped: ...
def prewitt(
    input: Untyped,
    axis: int = -1,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
) -> Untyped: ...
def sobel(
    input: Untyped,
    axis: int = -1,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
) -> Untyped: ...
def generic_laplace(
    input: Untyped,
    derivative2: Untyped,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> Untyped: ...
def laplace(
    input: Untyped,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
) -> Untyped: ...
def gaussian_laplace(
    input: Untyped,
    sigma: Untyped,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    **kwargs: float,
) -> Untyped: ...
def generic_gradient_magnitude(
    input: Untyped,
    derivative: Untyped,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> Untyped: ...
def gaussian_gradient_magnitude(
    input: Untyped,
    sigma: Untyped,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    **kwargs: Untyped,
) -> Untyped: ...
def correlate(
    input: Untyped,
    weights: Untyped,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def convolve(
    input: Untyped,
    weights: Untyped,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def uniform_filter1d(
    input: Untyped,
    size: int,
    axis: int = -1,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def uniform_filter(
    input: Untyped,
    size: int = 3,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    *,
    axes: Untyped | None = None,
) -> Untyped: ...
def minimum_filter1d(
    input: Untyped,
    size: int,
    axis: int = -1,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def maximum_filter1d(
    input: Untyped,
    size: int,
    axis: int = -1,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> Untyped: ...
def minimum_filter(
    input: Untyped,
    size: int | None = None,
    footprint: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    *,
    axes: Untyped | None = None,
) -> Untyped: ...
def maximum_filter(
    input: Untyped,
    size: int | None = None,
    footprint: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    *,
    axes: Untyped | None = None,
) -> Untyped: ...
def rank_filter(
    input: Untyped,
    rank: int,
    size: int | None = None,
    footprint: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    *,
    axes: Untyped | None = None,
) -> Untyped: ...
def median_filter(
    input: Untyped,
    size: int | None = None,
    footprint: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    *,
    axes: Untyped | None = None,
) -> Untyped: ...
def percentile_filter(
    input: Untyped,
    percentile: float,
    size: int | None = None,
    footprint: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    *,
    axes: Untyped | None = None,
) -> Untyped: ...
def generic_filter1d(
    input: Untyped,
    function: UntypedCallable,
    filter_size: float,
    axis: int = -1,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: Untyped | None = None,
) -> Untyped: ...
def generic_filter(
    input: Untyped,
    function: UntypedCallable,
    size: int | None = None,
    footprint: Untyped | None = None,
    output: Untyped | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    extra_arguments: tuple[object, ...] = (),
    extra_keywords: dict[str, object] | None = None,
) -> Untyped: ...
