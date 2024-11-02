from collections.abc import Callable, Mapping, Sequence
from typing import Any, Concatenate, Generic, Literal, NamedTuple, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import TypeVarTuple, Unpack

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import AnyReal, Seed
from scipy.optimize import OptimizeResult
from ._distn_infrastructure import rv_continuous, rv_continuous_frozen, rv_discrete

_Ts = TypeVarTuple("_Ts", default=Unpack[tuple[AnyReal, ...]])

# matplotlib.lines.Axes
_MPL_Axes: TypeAlias = Any

_RealVectorLike: TypeAlias = Sequence[AnyReal] | onpt.CanArray[Any, np.dtype[np.floating[Any] | np.integer[Any] | np.bool_]]
_Bounds: TypeAlias = Mapping[str, tuple[AnyReal, AnyReal]] | Sequence[tuple[AnyReal, AnyReal]]
# TODO: make more specific
_Optimizer: TypeAlias = Callable[Concatenate[Callable[..., AnyReal], ...], OptimizeResult]

_GOFStatName: TypeAlias = Literal["ad", "ks", "cvm", "filliben"]
_GOFStatFunc: TypeAlias = Callable[[rv_continuous_frozen, npt.NDArray[np.float64]], float | np.float32 | np.float64]

@type_check_only
class _PXF(Protocol[Unpack[_Ts]]):
    def __call__(self, x: AnyReal, /, *params: Unpack[_Ts]) -> np.float64: ...

class FitResult(Generic[Unpack[_Ts]]):
    # tuple of at least size 1
    params: tuple[AnyReal, Unpack[tuple[AnyReal, ...]]]
    success: bool | None
    message: str | None
    discrete: bool
    pxf: _PXF[Unpack[_Ts]]

    def __init__(
        self,
        /,
        dist: rv_continuous | rv_discrete,
        data: _ArrayLikeFloat_co,
        discrete: bool,
        res: OptimizeResult,
    ) -> None: ...
    def nllf(self, params: tuple[AnyReal, ...] | None = None, data: _ArrayLikeFloat_co | None = None) -> np.float64: ...
    def plot(self, ax: _MPL_Axes | None = None, *, plot_type: Literal["hist", "qq", "pp", "cdf"] = "hist") -> _MPL_Axes: ...

class GoodnessOfFitResult(NamedTuple):
    fit_result: FitResult
    statistic: float | np.float64
    pvalue: float | np.float64
    null_distribution: onpt.Array[tuple[int], np.float64]

@overload
def fit(
    dist: rv_discrete,
    data: _RealVectorLike,
    bounds: _Bounds | None = None,
    *,
    guess: Mapping[str, AnyReal] | _RealVectorLike | None = None,
    method: Literal["mle", "mse"] = "mle",
    optimizer: _Optimizer = ...,
) -> FitResult[AnyReal, Unpack[tuple[AnyReal, ...]]]: ...
@overload
def fit(
    dist: rv_continuous,
    data: _RealVectorLike,
    bounds: _Bounds | None = None,
    *,
    guess: Mapping[str, AnyReal] | _RealVectorLike | None = None,
    method: Literal["mle", "mse"] = "mle",
    optimizer: _Optimizer = ...,
) -> FitResult[AnyReal, AnyReal, Unpack[tuple[AnyReal, ...]]]: ...

#
def goodness_of_fit(
    dist: rv_continuous,
    data: _RealVectorLike,
    *,
    known_params: Mapping[str, AnyReal] | None = None,
    fit_params: Mapping[str, AnyReal] | None = None,
    guessed_params: Mapping[str, AnyReal] | None = None,
    statistic: _GOFStatName | _GOFStatFunc = "ad",
    n_mc_samples: int = 9999,
    random_state: Seed | None = None,
) -> GoodnessOfFitResult: ...
