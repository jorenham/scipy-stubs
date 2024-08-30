from scipy import special as special
from scipy._lib._util import normalize_axis_index as normalize_axis_index
from scipy._typing import Untyped
from ._ni_docstrings import docfiller as docfiller

def spline_filter1d(input, order: int = 3, axis: int = -1, output=..., mode: str = "mirror") -> Untyped: ...
def spline_filter(input, order: int = 3, output=..., mode: str = "mirror") -> Untyped: ...
def geometric_transform(
    input,
    mapping,
    output_shape: Untyped | None = None,
    output: Untyped | None = None,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
    extra_arguments=(),
    extra_keywords: Untyped | None = None,
) -> Untyped: ...
def map_coordinates(
    input,
    coordinates,
    output: Untyped | None = None,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
) -> Untyped: ...
def affine_transform(
    input,
    matrix,
    offset: float = 0.0,
    output_shape: Untyped | None = None,
    output: Untyped | None = None,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
) -> Untyped: ...
def shift(
    input, shift, output: Untyped | None = None, order: int = 3, mode: str = "constant", cval: float = 0.0, prefilter: bool = True
) -> Untyped: ...
def zoom(
    input,
    zoom,
    output: Untyped | None = None,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
    *,
    grid_mode: bool = False,
) -> Untyped: ...
def rotate(
    input,
    angle,
    axes=(1, 0),
    reshape: bool = True,
    output: Untyped | None = None,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
) -> Untyped: ...
