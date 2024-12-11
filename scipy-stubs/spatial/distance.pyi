# NOTE: Scipy already has a `distance.pyi` stub, but it has several errors, which are fixed here
import sys
from typing import Any, Literal, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import Buffer

import numpy as np
import optype.numpy as onp
import optype.typing as opt

__all__ = [
    "braycurtis",
    "canberra",
    "cdist",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "directed_hausdorff",
    "euclidean",
    "hamming",
    "is_valid_dm",
    "is_valid_y",
    "jaccard",
    "jensenshannon",
    "kulczynski1",
    "mahalanobis",
    "minkowski",
    "num_obs_dm",
    "num_obs_y",
    "pdist",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "squareform",
    "yule",
]

# Anything that can be parsed by `np.float64.__init__` and is thus
# compatible with `onp.ArrayND.__setitem__` (for a float64 array)
if sys.version_info >= (3, 12):
    _FloatValue: TypeAlias = str | Buffer | opt.AnyFloat
else:
    _FloatValue: TypeAlias = str | bytes | memoryview | bytearray | opt.AnyFloat

@type_check_only
class _MetricCallback1(Protocol):
    def __call__(self, xa: onp.ArrayND, xb: onp.ArrayND, /) -> _FloatValue | None: ...

@type_check_only
class _MetricCallback2(Protocol):
    def __call__(self, xa: onp.ArrayND, xb: onp.ArrayND, /, **kwargs: object) -> _FloatValue | None: ...

# NOTE(jorenham): PEP 612 won't work here, becayse it requires both `*args` and `**kwargs` to be used.
_MetricCallback: TypeAlias = _MetricCallback1 | _MetricCallback2

_MetricKind: TypeAlias = Literal[
    "braycurtis",
    "canberra",
    "chebychev",
    "chebyshev",
    "cheby",
    "cheb",
    "ch",
    "cityblock",
    "cblock",
    "cb",
    "c",
    "correlation",
    "co",
    "cosine",
    "cos",
    "dice",
    "euclidean",
    "euclid",
    "eu",
    "e",
    "hamming",
    "hamm",
    "ha",
    "h",
    "minkowski",
    "mi",
    "m",
    "pnorm",
    "jaccard",
    "jacc",
    "ja",
    "j",
    "jensenshannon",
    "js",
    "kulczynski1",
    "mahalanobis",
    "mahal",
    "mah",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "se",
    "s",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "sqe",
    "sqeuclid",
    "yule",
]

# Function annotations

def braycurtis(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...
def canberra(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...

# TODO: Add `metric`-specific overloads
# Returns a float64 or float128 array, depending on the input dtype
@overload
def cdist(
    XA: onp.ToFloat2D,
    XB: onp.ToFloat2D,
    metric: _MetricKind = "euclidean",
    *,
    out: onp.ArrayND[np.floating[Any]] | None = None,
    p: float = 2,
    w: onp.ToFloat1D | None = None,
    V: onp.ToFloat2D | None = None,
    VI: onp.ToFloat2D | None = None,
) -> onp.ArrayND[np.floating[Any]]: ...
@overload
def cdist(
    XA: onp.ToFloat2D,
    XB: onp.ToFloat2D,
    metric: _MetricCallback,
    *,
    out: onp.ArrayND[np.floating[Any]] | None = None,
    **kwargs: object,
) -> onp.ArrayND[np.floating[Any]]: ...
def chebyshev(u: onp.ToComplex1D, v: onp.ToComplex1D, w: onp.ToComplex1D | None = None) -> np.number[Any]: ...
def cityblock(u: onp.ToComplex1D, v: onp.ToComplex1D, w: onp.ToComplex1D | None = None) -> np.number[Any]: ...
def correlation(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None, centered: bool = True) -> np.float64: ...
def cosine(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...
def dice(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> float: ...
def directed_hausdorff(u: onp.ToFloat1D, v: onp.ToFloat1D, seed: int | None = 0) -> tuple[float, int, int]: ...
def euclidean(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> float: ...
def hamming(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...
def is_valid_dm(
    D: onp.ToFloat2D,
    tol: float = 0.0,
    throw: bool = False,
    name: str | None = "D",
    warning: bool = False,
) -> bool: ...
def is_valid_y(y: onp.ToFloat1D, warning: bool = False, throw: bool = False, name: str | None = None) -> bool: ...
def jaccard(u: onp.ToBool1D, v: onp.ToBool1D, w: onp.ToBool1D | None = None) -> np.float64: ...
def jensenshannon(
    p: onp.ToFloat1D,
    q: onp.ToFloat1D,
    base: float | None = None,
    *,
    axis: int = 0,
    keepdims: bool = False,
) -> np.float64: ...
def kulczynski1(u: onp.ToFloat1D, v: onp.ToFloat1D, *, w: onp.ToFloat1D | None = None) -> np.float64: ...
def mahalanobis(u: onp.ToFloat1D, v: onp.ToFloat1D, VI: onp.ToFloat2D) -> np.float64: ...
def minkowski(u: onp.ToFloat1D, v: onp.ToFloat1D, p: float = 2, w: onp.ToFloat1D | None = None) -> float: ...
def num_obs_dm(d: onp.ToFloat1D) -> int: ...
def num_obs_y(Y: onp.ToFloat1D) -> int: ...

# TODO: Add `metric`-specific overloads
@overload
def pdist(
    X: onp.ToFloat2D,
    metric: _MetricKind = "euclidean",
    *,
    out: onp.ArrayND[np.floating[Any]] | None = None,
    p: float = 2,
    w: onp.ToFloat1D | None = None,
    V: onp.ToFloat2D | None = None,
    VI: onp.ToFloat2D | None = None,
) -> onp.ArrayND[np.floating[Any]]: ...
@overload
def pdist(
    X: onp.ToFloat2D,
    metric: _MetricCallback,
    *,
    out: onp.ArrayND[np.floating[Any]] | None = None,
    **kwargs: object,
) -> onp.ArrayND[np.floating[Any]]: ...
def seuclidean(u: onp.ToFloat1D, v: onp.ToFloat1D, V: onp.ToFloat1D) -> float: ...
def sokalmichener(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> float: ...
def sokalsneath(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...
def sqeuclidean(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> np.float64: ...
def squareform(
    X: onp.ToFloat2D,
    force: Literal["no", "tomatrix", "tovector"] = "no",
    checks: bool = True,
) -> onp.Array1D[np.floating[Any]] | onp.Array2D[np.floating[Any]]: ...
def rogerstanimoto(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> float: ...
def russellrao(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> float: ...
def yule(u: onp.ToFloat1D, v: onp.ToFloat1D, w: onp.ToFloat1D | None = None) -> float: ...
