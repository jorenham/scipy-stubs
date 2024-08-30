from typing import Any, Literal, Protocol, SupportsFloat, SupportsIndex, TypeAlias, overload, type_check_only

import numpy as np
import numpy.typing as npt

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
# compatible with `npt.NDArray.__setitem__` (for a float64 array)
_FloatValue: TypeAlias = None | str | bytes | SupportsFloat | SupportsIndex

@type_check_only
class _MetricCallback1(Protocol):
    def __call__(self, xa: npt.NDArray[Any], xb: npt.NDArray[Any], /) -> _FloatValue: ...

@type_check_only
class _MetricCallback2(Protocol):
    def __call__(self, xa: npt.NDArray[Any], xb: npt.NDArray[Any], /, **kwargs: Any) -> _FloatValue: ...

# TODO: Use a single protocol with a parameter specification variable
# once available (PEP 612)
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

def braycurtis(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> np.float64: ...
def canberra(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> np.float64: ...

# TODO: Add `metric`-specific overloads
# Returns a float64 or float128 array, depending on the input dtype
@overload
def cdist(
    XA: npt.ArrayLike,
    XB: npt.ArrayLike,
    metric: _MetricKind = ...,
    *,
    out: None | npt.NDArray[np.floating[Any]] = ...,
    p: float = ...,
    w: npt.ArrayLike | None = ...,
    V: npt.ArrayLike | None = ...,
    VI: npt.ArrayLike | None = ...,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def cdist(
    XA: npt.ArrayLike,
    XB: npt.ArrayLike,
    metric: _MetricCallback,
    *,
    out: None | npt.NDArray[np.floating[Any]] = ...,
    **kwargs: Any,
) -> npt.NDArray[np.floating[Any]]: ...

# TODO: Wait for dtype support; the return type is
# dependent on the input arrays dtype
def chebyshev(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> Any: ...

# TODO: Wait for dtype support; the return type is
# dependent on the input arrays dtype
def cityblock(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> Any: ...
def correlation(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ..., centered: bool = ...) -> np.float64: ...
def cosine(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> np.float64: ...
def dice(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> float: ...
def directed_hausdorff(u: npt.ArrayLike, v: npt.ArrayLike, seed: int | None = ...) -> tuple[float, int, int]: ...
def euclidean(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> float: ...
def hamming(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> np.float64: ...
def is_valid_dm(D: npt.ArrayLike, tol: float = ..., throw: bool = ..., name: str | None = ..., warning: bool = ...) -> bool: ...
def is_valid_y(y: npt.ArrayLike, warning: bool = ..., throw: bool = ..., name: str | None = ...) -> bool: ...
def jaccard(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> np.float64: ...
def jensenshannon(p: npt.ArrayLike, q: npt.ArrayLike, base: float | None = ...) -> np.float64: ...
def kulczynski1(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> np.float64: ...
def mahalanobis(u: npt.ArrayLike, v: npt.ArrayLike, VI: npt.ArrayLike) -> np.float64: ...
def minkowski(u: npt.ArrayLike, v: npt.ArrayLike, p: float = ..., w: npt.ArrayLike | None = ...) -> float: ...
def num_obs_dm(d: npt.ArrayLike) -> int: ...
def num_obs_y(Y: npt.ArrayLike) -> int: ...

# TODO: Add `metric`-specific overloads
@overload
def pdist(
    X: npt.ArrayLike,
    metric: _MetricKind = ...,
    *,
    out: None | npt.NDArray[np.floating[Any]] = ...,
    p: float = ...,
    w: npt.ArrayLike | None = ...,
    V: npt.ArrayLike | None = ...,
    VI: npt.ArrayLike | None = ...,
) -> npt.NDArray[np.floating[Any]]: ...
@overload
def pdist(
    X: npt.ArrayLike, metric: _MetricCallback, *, out: None | npt.NDArray[np.floating[Any]] = ..., **kwargs: Any
) -> npt.NDArray[np.floating[Any]]: ...
def seuclidean(u: npt.ArrayLike, v: npt.ArrayLike, V: npt.ArrayLike) -> float: ...
def sokalmichener(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> float: ...
def sokalsneath(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> np.float64: ...
def sqeuclidean(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> np.float64: ...
def squareform(X: npt.ArrayLike, force: Literal["no", "tomatrix", "tovector"] = ..., checks: bool = ...) -> npt.NDArray[Any]: ...
def rogerstanimoto(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> float: ...
def russellrao(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> float: ...
def yule(u: npt.ArrayLike, v: npt.ArrayLike, w: npt.ArrayLike | None = ...) -> float: ...
