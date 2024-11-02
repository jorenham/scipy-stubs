from collections.abc import Callable
from types import ModuleType
from typing import Any, Generic, Literal, NamedTuple, Protocol, TypeAlias, final, overload, type_check_only
from typing_extensions import Self, TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co
from optype import CanIndex
from scipy._typing import Alternative, AnyBool, AnyInt, AnyReal, NanPolicy
from scipy.optimize import OptimizeResult
from ._distn_infrastructure import rv_continuous_frozen
from ._fit import FitResult
from ._resampling import PermutationMethod
from ._stats_py import SignificanceResult
from ._typing import BaseBunch

__all__ = [
    "anderson",
    "anderson_ksamp",
    "ansari",
    "bartlett",
    "bayes_mvs",
    "boxcox",
    "boxcox_llf",
    "boxcox_normmax",
    "boxcox_normplot",
    "circmean",
    "circstd",
    "circvar",
    "directional_stats",
    "false_discovery_control",
    "fligner",
    "kstat",
    "kstatvar",
    "levene",
    "median_test",
    "mood",
    "mvsdist",
    "ppcc_max",
    "ppcc_plot",
    "probplot",
    "shapiro",
    "wilcoxon",
    "yeojohnson",
    "yeojohnson_llf",
    "yeojohnson_normmax",
    "yeojohnson_normplot",
]

_T = TypeVar("_T")
_NDT_co = TypeVar(
    "_NDT_co",
    covariant=True,
    bound=np.float64 | npt.NDArray[np.float64],
    default=np.float64 | npt.NDArray[np.float64],
)

@type_check_only
class _TestResult(NamedTuple, Generic[_NDT_co]):
    statistic: _NDT_co
    pvalue: _NDT_co

@type_check_only
class _ConfidenceInterval(NamedTuple):
    statistic: float | np.float64
    minmax: tuple[float, float] | tuple[np.float64, np.float64]

# represents the e.g. `matplotlib.pyplot` module and a `matplotlib.axes.Axes` object with a `plot` and `text` method
@type_check_only
class _CanPlotText(Protocol):
    # NOTE: `Any` is required as return type because it's covariant, and not shouldn't be `Never`.
    def plot(self, /, *args: float | npt.ArrayLike | str, **kwargs: object) -> Any: ...  # noqa: ANN401
    def text(self, /, x: float, y: float, s: str, fontdict: dict[str, Any] | None = None, **kwargs: object) -> Any: ...  # noqa: ANN401

@type_check_only
class _CanPPF(Protocol):
    def ppf(self, q: npt.NDArray[np.float64], /) -> npt.NDArray[np.float64]: ...

@type_check_only
class _HasX(Protocol):
    x: float | np.floating[Any]

_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple3: TypeAlias = tuple[_T, _T, _T]
_VectorF8: TypeAlias = onpt.Array[tuple[int], np.float64]

_FuncObjective1D: TypeAlias = Callable[[float], float | np.floating[Any]]
_FuncMinimize1D: TypeAlias = Callable[[_FuncObjective1D], _HasX] | Callable[[_FuncObjective1D], OptimizeResult]

_KStatOrder: TypeAlias = Literal[1, 2, 3, 4]
_CenterMethod: TypeAlias = Literal["mean", "median", "trimmed"]
_RVCAnderson: TypeAlias = Literal["norm", "expon", "logistic", "extreme1", "gumbel", "gumbel_l", "gumbel_r", "weibull_min"]
_RVC0: TypeAlias = Literal[
    "anglit",
    "arcsine",
    "cauchy",
    "cosine",
    "expon",
    "gibrat",
    "gumbel_l",
    "gumbel_r",
    "halfcauchy",
    "halflogistic",
    "halfnorm",
    "hypsecant",
    "kstwobign",
    "laplace",
    "levy",
    "levy_l",
    "logistic",
    "maxwell",
    "moyal",
    "norm",
    "rayleigh",
    "semicircular",
    "uniform",
    "wald",
]
_RVC1: TypeAlias = Literal[
    "alpha",
    "argus",
    "bradford",
    "chi",
    "chi2",
    "erlang",
    "exponnorm",
    "exponpow",
    "fatiguelife",
    "fisk",
    "gamma",
    "genextreme",
    "genlogistic",
    "gennorm",
    "genpareto",
    "gompertz",
    "invgamma",
    "invgauss",
    "invweibull",
    "loggamma",
    "loglaplace",
    "lognorm",
    "lomax",
    "nakagami",
    "pareto",
    "pearson3",
    "powerlaw",
    "powernorm",
    "rdist",
    "recipinvgauss",
    "rel_breitwigner",
    "rice",
    "skewcauchy",
    "skewnorm",
    "t",
    "triang",
    "tukeylambda",
    "vonmises",
    "vonmises_line",
    "weibull_max",
    "weibull_min",
    "wrapcauchy",
]

###

@final
class _BigFloat: ...

class DirectionalStats:
    mean_direction: npt.NDArray[np.float64]
    mean_resultant_length: npt.NDArray[np.float64]
    def __init__(self, /, mean_direction: npt.NDArray[np.float64], mean_resultant_length: npt.NDArray[np.float64]) -> None: ...

class ShapiroResult(_TestResult[_NDT_co], Generic[_NDT_co]): ...
class AnsariResult(_TestResult[_NDT_co], Generic[_NDT_co]): ...
class BartlettResult(_TestResult[_NDT_co], Generic[_NDT_co]): ...
class LeveneResult(_TestResult[_NDT_co], Generic[_NDT_co]): ...
class FlignerResult(_TestResult[_NDT_co], Generic[_NDT_co]): ...

#
class Mean(_ConfidenceInterval): ...
class Variance(_ConfidenceInterval): ...
class Std_dev(_ConfidenceInterval): ...

class AndersonResult(BaseBunch[np.float64, _VectorF8, _VectorF8]):
    @property
    def statistic(self) -> np.float64: ...
    @property
    def critical_values(self) -> _VectorF8: ...
    @property
    def significance_level(self) -> _VectorF8: ...
    @property
    def fit_result(self) -> FitResult[np.float64, np.float64]: ...
    def __new__(
        _cls,
        statistic: np.float64,
        critical_values: _VectorF8,
        significance_level: _VectorF8,
        *,
        fit_result: FitResult[np.float64, np.float64],
    ) -> Self: ...
    def __init__(
        self,
        /,
        statistic: np.float64,
        critical_values: _VectorF8,
        significance_level: _VectorF8,
        *,
        fit_result: FitResult[np.float64, np.float64],
    ) -> None: ...

class Anderson_ksampResult(BaseBunch[np.float64, _VectorF8, np.float64]):
    @property
    def statistic(self) -> np.float64: ...
    @property
    def critical_values(self) -> _VectorF8: ...
    @property
    def pvalue(self) -> np.float64: ...
    def __new__(_cls, statistic: np.float64, critical_values: _VectorF8, pvalue: np.float64) -> Self: ...
    def __init__(self, /, statistic: np.float64, critical_values: _VectorF8, pvalue: np.float64) -> None: ...

class WilcoxonResult(BaseBunch[_NDT_co, _NDT_co], Generic[_NDT_co]):  # pyright: ignore[reportInvalidTypeArguments]
    @property
    def statistic(self) -> _NDT_co: ...
    @property
    def pvalue(self) -> _NDT_co: ...
    def __new__(_cls, statistic: _NDT_co, pvalue: _NDT_co) -> Self: ...
    def __init__(self, /, statistic: _NDT_co, pvalue: _NDT_co) -> None: ...

class MedianTestResult(BaseBunch[np.float64, np.float64, np.float64, onpt.Array[tuple[Literal[2], int], np.float64]]):
    @property
    def statistic(self) -> np.float64: ...
    @property
    def pvalue(self) -> np.float64: ...
    @property
    def median(self) -> np.float64: ...
    @property
    def table(self) -> onpt.Array[tuple[Literal[2], int], np.float64]: ...
    def __new__(
        _cls,
        statistic: np.float64,
        pvalue: np.float64,
        median: np.float64,
        table: onpt.Array[tuple[Literal[2], int], np.float64],
    ) -> Self: ...
    def __init__(
        self,
        /,
        statistic: np.float64,
        pvalue: np.float64,
        median: np.float64,
        table: onpt.Array[tuple[Literal[2], int], np.float64],
    ) -> None: ...

def bayes_mvs(data: _ArrayLikeFloat_co, alpha: AnyReal = 0.9) -> tuple[Mean, Variance, Std_dev]: ...
def mvsdist(data: _ArrayLikeFloat_co) -> _Tuple3[rv_continuous_frozen]: ...

#
@overload
def kstat(
    data: _ArrayLikeFloat_co,
    n: _KStatOrder = 2,
    *,
    axis: None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[0, False] = False,
) -> np.float64: ...
@overload
def kstat(
    data: _ArrayLikeFloat_co,
    n: _KStatOrder = 2,
    *,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[1, True],
) -> npt.NDArray[np.float64]: ...
@overload
def kstat(
    data: _ArrayLikeFloat_co,
    n: _KStatOrder = 2,
    *,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> np.float64 | npt.NDArray[np.float64]: ...

#
@overload
def kstatvar(
    data: _ArrayLikeFloat_co,
    n: _KStatOrder = 2,
    *,
    axis: None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[0, False] = False,
) -> np.float64: ...
@overload
def kstatvar(
    data: _ArrayLikeFloat_co,
    n: _KStatOrder = 2,
    *,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[1, True],
) -> npt.NDArray[np.float64]: ...
@overload
def kstatvar(
    data: _ArrayLikeFloat_co,
    n: _KStatOrder = 2,
    *,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> np.float64 | npt.NDArray[np.float64]: ...

#
@overload
def probplot(
    x: _ArrayLikeFloat_co,
    sparams: tuple[()] = (),
    dist: _RVC0 | _CanPPF = "norm",
    fit: Literal[True] = True,
    plot: _CanPlotText | ModuleType | None = None,
    rvalue: AnyBool = False,
) -> tuple[_Tuple2[npt.NDArray[np.float64]], _Tuple3[np.float64]]: ...
@overload
def probplot(
    x: _ArrayLikeFloat_co,
    sparams: tuple[()] = (),
    dist: _RVC0 | _CanPPF = "norm",
    *,
    fit: Literal[False],
    plot: _CanPlotText | ModuleType | None = None,
    rvalue: AnyBool = False,
) -> _Tuple2[npt.NDArray[np.float64]]: ...
@overload
def probplot(
    x: _ArrayLikeFloat_co,
    sparams: tuple[AnyReal, ...],
    dist: str | _CanPPF = "norm",
    fit: Literal[True] = True,
    plot: _CanPlotText | ModuleType | None = None,
    rvalue: AnyBool = False,
) -> tuple[_Tuple2[npt.NDArray[np.float64]], _Tuple3[np.float64]]: ...
@overload
def probplot(
    x: _ArrayLikeFloat_co,
    sparams: tuple[AnyReal],
    dist: str | _CanPPF = "norm",
    *,
    fit: Literal[False],
    plot: _CanPlotText | ModuleType | None = None,
    rvalue: AnyBool = False,
) -> _Tuple2[npt.NDArray[np.float64]]: ...

#
def ppcc_max(
    x: _ArrayLikeFloat_co,
    brack: _Tuple2[AnyReal] | _Tuple3[AnyReal] = (0.0, 1.0),
    dist: _RVC1 | _CanPPF = "tukeylambda",
) -> np.float64: ...
def ppcc_plot(
    x: _ArrayLikeFloat_co,
    a: AnyReal,
    b: AnyReal,
    dist: _RVC1 | _CanPPF = "tukeylambda",
    plot: _CanPlotText | ModuleType | None = None,
    N: int = 80,
) -> _Tuple2[npt.NDArray[np.float64]]: ...

#
def boxcox_llf(lmb: AnyReal, data: _ArrayLikeFloat_co) -> np.float64 | npt.NDArray[np.float64]: ...
@overload
def boxcox(
    x: _ArrayLikeFloat_co,
    lmbda: None = None,
    alpha: None = None,
    optimizer: _FuncMinimize1D | None = None,
) -> tuple[_VectorF8, np.float64]: ...
@overload
def boxcox(
    x: _ArrayLikeFloat_co,
    lmbda: AnyReal,
    alpha: float | np.floating[Any] | None = None,
    optimizer: _FuncMinimize1D | None = None,
) -> _VectorF8: ...
@overload
def boxcox(
    x: _ArrayLikeFloat_co,
    lmbda: None,
    alpha: float | np.floating[Any],
    optimizer: _FuncMinimize1D | None = None,
) -> tuple[_VectorF8, np.float64, _Tuple2[float]]: ...
@overload
def boxcox(
    x: _ArrayLikeFloat_co,
    lmbda: None = None,
    *,
    alpha: float | np.floating[Any],
    optimizer: _FuncMinimize1D | None = None,
) -> tuple[_VectorF8, np.float64, _Tuple2[float]]: ...
@overload
def boxcox_normmax(
    x: _ArrayLikeFloat_co,
    brack: _Tuple2[AnyReal] | None = None,
    method: Literal["pearsonr", "mle"] = "pearsonr",
    optimizer: _FuncMinimize1D | None = None,
    *,
    ymax: AnyReal | _BigFloat = ...,
) -> np.float64: ...
@overload
def boxcox_normmax(
    x: _ArrayLikeFloat_co,
    brack: _Tuple2[AnyReal] | None = None,
    *,
    method: Literal["all"],
    optimizer: _FuncMinimize1D | None = None,
    ymax: AnyReal | _BigFloat = ...,
) -> onpt.Array[tuple[Literal[2]], np.float64]: ...
@overload
def boxcox_normmax(
    x: _ArrayLikeFloat_co,
    brack: _Tuple2[AnyReal] | None,
    method: Literal["all"],
    optimizer: _FuncMinimize1D | None = None,
    *,
    ymax: AnyReal | _BigFloat = ...,
) -> onpt.Array[tuple[Literal[2]], np.float64]: ...
def boxcox_normplot(
    x: _ArrayLikeFloat_co,
    la: AnyReal,
    lb: AnyReal,
    plot: _CanPlotText | ModuleType | None = None,
    N: AnyInt = 80,
) -> _Tuple2[npt.NDArray[np.float64]]: ...

#
def yeojohnson_llf(lmb: AnyReal, data: _ArrayLikeFloat_co) -> onpt.Array[tuple[()], np.float64]: ...
@overload
def yeojohnson(x: _ArrayLikeFloat_co, lmbda: None = None) -> tuple[_VectorF8, np.float64]: ...
@overload
def yeojohnson(x: _ArrayLikeFloat_co, lmbda: AnyReal) -> _VectorF8: ...
def yeojohnson_normmax(x: _ArrayLikeFloat_co, brack: _Tuple2[AnyReal] | None = None) -> np.float64: ...
def yeojohnson_normplot(
    x: _ArrayLikeFloat_co,
    la: AnyReal,
    lb: AnyReal,
    plot: _CanPlotText | ModuleType | None = None,
    N: AnyInt = 80,
) -> _Tuple2[npt.NDArray[np.float64]]: ...

#
def anderson(x: _ArrayLikeFloat_co, dist: _RVCAnderson = "norm") -> AndersonResult: ...
def anderson_ksamp(
    samples: _ArrayLikeFloat_co,
    midrank: bool = True,
    *,
    method: PermutationMethod | None = None,
) -> Anderson_ksampResult: ...

#
@overload
def shapiro(
    x: _ArrayLikeFloat_co,
    *,
    axis: None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[0, False] = False,
) -> ShapiroResult[np.float64]: ...
@overload
def shapiro(
    x: _ArrayLikeFloat_co,
    *,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[1, True],
) -> ShapiroResult[npt.NDArray[np.float64]]: ...
@overload
def shapiro(
    x: _ArrayLikeFloat_co,
    *,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> ShapiroResult: ...

#
@overload
def ansari(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    alternative: Alternative = "two-sided",
    *,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[0, False] = False,
) -> AnsariResult[np.float64]: ...
@overload
def ansari(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    alternative: Alternative = "two-sided",
    *,
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[1, True],
) -> AnsariResult[npt.NDArray[np.float64]]: ...
@overload
def ansari(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    alternative: Alternative = "two-sided",
    *,
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> AnsariResult: ...

#
@overload
def bartlett(
    *samples: _ArrayLikeFloat_co,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[0, False] = False,
) -> BartlettResult[np.float64]: ...
@overload
def bartlett(
    *samples: _ArrayLikeFloat_co,
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[1, True],
) -> BartlettResult[npt.NDArray[np.float64]]: ...
@overload
def bartlett(
    *samples: _ArrayLikeFloat_co,
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> BartlettResult: ...

#
@overload
def levene(
    *samples: _ArrayLikeFloat_co,
    center: _CenterMethod = "median",
    proportiontocut: AnyReal = 0.05,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[0, False] = False,
) -> LeveneResult[np.float64]: ...
@overload
def levene(
    *samples: _ArrayLikeFloat_co,
    center: _CenterMethod = "median",
    proportiontocut: AnyReal = 0.05,
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[1, True],
) -> LeveneResult[npt.NDArray[np.float64]]: ...
@overload
def levene(
    *samples: _ArrayLikeFloat_co,
    center: _CenterMethod = "median",
    proportiontocut: AnyReal = 0.05,
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> LeveneResult: ...

#
@overload
def fligner(
    *samples: _ArrayLikeFloat_co,
    center: _CenterMethod = "median",
    proportiontocut: AnyReal = 0.05,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[0, False] = False,
) -> FlignerResult[np.float64]: ...
@overload
def fligner(
    *samples: _ArrayLikeFloat_co,
    center: _CenterMethod = "median",
    proportiontocut: AnyReal = 0.05,
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[1, True],
) -> FlignerResult[npt.NDArray[np.float64]]: ...
@overload
def fligner(
    *samples: _ArrayLikeFloat_co,
    center: _CenterMethod = "median",
    proportiontocut: AnyReal = 0.05,
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> FlignerResult: ...

#
@overload
def mood(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    axis: None,
    alternative: Alternative = "two-sided",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[0, False] = False,
) -> SignificanceResult[np.float64]: ...
@overload
def mood(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    axis: CanIndex | None = 0,
    alternative: Alternative = "two-sided",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[1, True],
) -> SignificanceResult[npt.NDArray[np.float64]]: ...
@overload
def mood(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    axis: CanIndex | None = 0,
    alternative: Alternative = "two-sided",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> SignificanceResult[np.float64 | npt.NDArray[np.float64]]: ...

#
@overload
def wilcoxon(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co | None = None,
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "wilcox",
    correction: AnyBool = False,
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "approx"] | PermutationMethod = "auto",
    *,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[0, False] = False,
) -> WilcoxonResult[np.float64]: ...
@overload
def wilcoxon(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co | None = None,
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "wilcox",
    correction: AnyBool = False,
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "approx"] | PermutationMethod = "auto",
    *,
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: Literal[1, True],
) -> WilcoxonResult[npt.NDArray[np.float64]]: ...
@overload
def wilcoxon(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co | None = None,
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "wilcox",
    correction: AnyBool = False,
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "approx"] | PermutationMethod = "auto",
    *,
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> WilcoxonResult: ...

#
def wilcoxon_result_object(
    statistic: np.float64,
    pvalue: np.float64,
    zstatistic: np.float64 | None = None,
) -> WilcoxonResult: ...  # undocumented
def wilcoxon_result_unpacker(res: WilcoxonResult) -> _Tuple2[np.float64] | _Tuple3[np.float64]: ...  # undocumented
def wilcoxon_outputs(kwds: dict[str, str]) -> Literal[2, 3]: ...  # undocumented

#
def median_test(
    *samples: _ArrayLikeFloat_co,
    ties: Literal["below", "above", "ignore"] = "below",
    correction: AnyBool = True,
    lambda_: AnyReal | str = 1,
    nan_policy: NanPolicy = "propagate",
) -> MedianTestResult: ...

#
@overload
def circmean(
    samples: _ArrayLikeFloat_co,
    high: AnyReal = ...,
    low: AnyReal = 0,
    axis: None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: Literal[0, False] = False,
) -> np.float64: ...
@overload
def circmean(
    samples: _ArrayLikeFloat_co,
    high: AnyReal = ...,
    low: AnyReal = 0,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: Literal[1, True],
) -> npt.NDArray[np.float64]: ...
@overload
def circmean(
    samples: _ArrayLikeFloat_co,
    high: AnyReal = ...,
    low: AnyReal = 0,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: AnyBool = False,
) -> np.float64 | npt.NDArray[np.float64]: ...

#
@overload
def circvar(
    samples: _ArrayLikeFloat_co,
    high: AnyReal = ...,
    low: AnyReal = 0,
    axis: None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: Literal[0, False] = False,
) -> np.float64: ...
@overload
def circvar(
    samples: _ArrayLikeFloat_co,
    high: AnyReal = ...,
    low: AnyReal = 0,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: Literal[1, True],
) -> npt.NDArray[np.float64]: ...
@overload
def circvar(
    samples: _ArrayLikeFloat_co,
    high: AnyReal = ...,
    low: AnyReal = 0,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: AnyBool = False,
) -> np.float64 | npt.NDArray[np.float64]: ...

#
@overload
def circstd(
    samples: _ArrayLikeFloat_co,
    high: AnyReal = ...,
    low: AnyReal = 0,
    axis: None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    normalize: AnyBool = False,
    keepdims: Literal[0, False] = False,
) -> np.float64: ...
@overload
def circstd(
    samples: _ArrayLikeFloat_co,
    high: AnyReal = ...,
    low: AnyReal = 0,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    normalize: AnyBool = False,
    keepdims: Literal[1, True],
) -> npt.NDArray[np.float64]: ...
@overload
def circstd(
    samples: _ArrayLikeFloat_co,
    high: AnyReal = ...,
    low: AnyReal = 0,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    normalize: AnyBool = False,
    keepdims: AnyBool = False,
) -> np.float64 | npt.NDArray[np.float64]: ...

#
def directional_stats(
    samples: _ArrayLikeFloat_co,
    *,
    axis: CanIndex | None = 0,
    normalize: AnyBool = True,
) -> DirectionalStats: ...

#
def false_discovery_control(
    ps: _ArrayLikeFloat_co,
    *,
    axis: CanIndex | None = 0,
    method: Literal["bh", "by"] = "bh",
) -> npt.NDArray[np.float64]: ...
