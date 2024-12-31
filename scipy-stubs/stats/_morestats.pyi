from collections.abc import Callable
from types import ModuleType
from typing import Any, Generic, Literal, NamedTuple, Protocol, TypeAlias, final, overload, type_check_only
from typing_extensions import Self, TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import Alternative, AnyBool, Falsy, NanPolicy, Truthy
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

###

_T = TypeVar("_T")
_NDT_co = TypeVar(
    "_NDT_co",
    covariant=True,
    bound=np.float64 | onp.ArrayND[np.float64],
    default=np.float64 | onp.ArrayND[np.float64],
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
    def plot(self, /, *args: float | onp.ToFloatND | str, **kwargs: object) -> Any: ...  # noqa: ANN401
    def text(self, /, x: float, y: float, s: str, fontdict: dict[str, Any] | None = None, **kwargs: object) -> Any: ...  # noqa: ANN401

@type_check_only
class _CanPPF(Protocol):
    def ppf(self, q: onp.ArrayND[np.float64], /) -> onp.ArrayND[np.float64]: ...

@type_check_only
class _HasX(Protocol):
    x: float | np.floating[Any]

_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple3: TypeAlias = tuple[_T, _T, _T]
_Float1D: TypeAlias = onp.Array1D[np.float64]

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

_ObjFun1D: TypeAlias = Callable[[float], float | np.floating[Any]]
_MinFun1D: TypeAlias = Callable[[_ObjFun1D], _HasX] | Callable[[_ObjFun1D], OptimizeResult]

_AndersonResult: TypeAlias = FitResult[Callable[[onp.ToFloat, onp.ToFloat], np.float64]]

###

@final
class _BigFloat: ...

class DirectionalStats:
    mean_direction: onp.ArrayND[np.float64]
    mean_resultant_length: onp.ArrayND[np.float64]
    #
    def __init__(self, /, mean_direction: onp.ArrayND[np.float64], mean_resultant_length: onp.ArrayND[np.float64]) -> None: ...

class ShapiroResult(_TestResult[_NDT_co], Generic[_NDT_co]): ...
class AnsariResult(_TestResult[_NDT_co], Generic[_NDT_co]): ...
class BartlettResult(_TestResult[_NDT_co], Generic[_NDT_co]): ...
class LeveneResult(_TestResult[_NDT_co], Generic[_NDT_co]): ...
class FlignerResult(_TestResult[_NDT_co], Generic[_NDT_co]): ...

#
class Mean(_ConfidenceInterval): ...
class Variance(_ConfidenceInterval): ...
class Std_dev(_ConfidenceInterval): ...

class AndersonResult(BaseBunch[np.float64, _Float1D, _Float1D]):
    @property
    def statistic(self, /) -> np.float64: ...
    @property
    def critical_values(self, /) -> _Float1D: ...
    @property
    def significance_level(self, /) -> _Float1D: ...
    @property
    def fit_result(self, /) -> _AndersonResult: ...

    #
    def __new__(
        _cls,
        statistic: np.float64,
        critical_values: _Float1D,
        significance_level: _Float1D,
        *,
        fit_result: _AndersonResult,
    ) -> Self: ...
    def __init__(
        self,
        /,
        statistic: np.float64,
        critical_values: _Float1D,
        significance_level: _Float1D,
        *,
        fit_result: _AndersonResult,
    ) -> None: ...

class Anderson_ksampResult(BaseBunch[np.float64, _Float1D, np.float64]):
    @property
    def statistic(self, /) -> np.float64: ...
    @property
    def critical_values(self, /) -> _Float1D: ...
    @property
    def pvalue(self, /) -> np.float64: ...
    def __new__(_cls, statistic: np.float64, critical_values: _Float1D, pvalue: np.float64) -> Self: ...
    def __init__(self, /, statistic: np.float64, critical_values: _Float1D, pvalue: np.float64) -> None: ...

class WilcoxonResult(BaseBunch[_NDT_co, _NDT_co], Generic[_NDT_co]):  # pyright: ignore[reportInvalidTypeArguments]
    @property
    def statistic(self, /) -> _NDT_co: ...
    @property
    def pvalue(self, /) -> _NDT_co: ...
    def __new__(_cls, statistic: _NDT_co, pvalue: _NDT_co) -> Self: ...
    def __init__(self, /, statistic: _NDT_co, pvalue: _NDT_co) -> None: ...

class MedianTestResult(BaseBunch[np.float64, np.float64, np.float64, onp.Array2D[np.float64]]):
    @property
    def statistic(self, /) -> np.float64: ...
    @property
    def pvalue(self, /) -> np.float64: ...
    @property
    def median(self, /) -> np.float64: ...
    @property
    def table(self, /) -> onp.Array2D[np.float64]: ...
    def __new__(_cls, statistic: np.float64, pvalue: np.float64, median: np.float64, table: onp.Array2D[np.float64]) -> Self: ...
    def __init__(
        self,
        /,
        statistic: np.float64,
        pvalue: np.float64,
        median: np.float64,
        table: onp.Array2D[np.float64],
    ) -> None: ...

def bayes_mvs(data: onp.ToFloatND, alpha: onp.ToFloat = 0.9) -> tuple[Mean, Variance, Std_dev]: ...

#
def mvsdist(data: onp.ToFloatND) -> _Tuple3[rv_continuous_frozen]: ...

#
@overload
def kstat(
    data: onp.ToFloatND,
    n: _KStatOrder = 2,
    *,
    axis: None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Falsy = False,
) -> np.float64: ...
@overload
def kstat(
    data: onp.ToFloatND,
    n: _KStatOrder = 2,
    *,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Truthy,
) -> onp.ArrayND[np.float64]: ...
@overload
def kstat(
    data: onp.ToFloatND,
    n: _KStatOrder = 2,
    *,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> np.float64 | onp.ArrayND[np.float64]: ...

#
@overload
def kstatvar(
    data: onp.ToFloatND,
    n: _KStatOrder = 2,
    *,
    axis: None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Falsy = False,
) -> np.float64: ...
@overload
def kstatvar(
    data: onp.ToFloatND,
    n: _KStatOrder = 2,
    *,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Truthy,
) -> onp.ArrayND[np.float64]: ...
@overload
def kstatvar(
    data: onp.ToFloatND,
    n: _KStatOrder = 2,
    *,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> np.float64 | onp.ArrayND[np.float64]: ...

#
@overload
def probplot(
    x: onp.ToFloat | onp.ToFloatND,
    sparams: tuple[()] = (),
    dist: _RVC0 | _CanPPF = "norm",
    fit: Truthy = True,
    plot: _CanPlotText | ModuleType | None = None,
    rvalue: AnyBool = False,
) -> tuple[_Tuple2[onp.ArrayND[np.float64]], _Tuple3[np.float64]]: ...
@overload
def probplot(
    x: onp.ToFloat | onp.ToFloatND,
    sparams: tuple[()] = (),
    dist: _RVC0 | _CanPPF = "norm",
    *,
    fit: Falsy,
    plot: _CanPlotText | ModuleType | None = None,
    rvalue: AnyBool = False,
) -> _Tuple2[onp.ArrayND[np.float64]]: ...
@overload
def probplot(
    x: onp.ToFloat | onp.ToFloatND,
    sparams: tuple[onp.ToFloat, ...],
    dist: str | _CanPPF = "norm",
    fit: Truthy = True,
    plot: _CanPlotText | ModuleType | None = None,
    rvalue: AnyBool = False,
) -> tuple[_Tuple2[onp.ArrayND[np.float64]], _Tuple3[np.float64]]: ...
@overload
def probplot(
    x: onp.ToFloat | onp.ToFloatND,
    sparams: tuple[onp.ToFloat],
    dist: str | _CanPPF = "norm",
    *,
    fit: Falsy,
    plot: _CanPlotText | ModuleType | None = None,
    rvalue: AnyBool = False,
) -> _Tuple2[onp.ArrayND[np.float64]]: ...

#
def ppcc_max(
    x: onp.ToFloat | onp.ToFloatND,
    brack: _Tuple2[onp.ToFloat] | _Tuple3[onp.ToFloat] = (0.0, 1.0),
    dist: _RVC1 | _CanPPF = "tukeylambda",
) -> np.float64: ...

#
def ppcc_plot(
    x: onp.ToFloat | onp.ToFloatND,
    a: onp.ToFloat,
    b: onp.ToFloat,
    dist: _RVC1 | _CanPPF = "tukeylambda",
    plot: _CanPlotText | ModuleType | None = None,
    N: int = 80,
) -> _Tuple2[onp.ArrayND[np.float64]]: ...

#
def boxcox_llf(lmb: onp.ToFloat, data: onp.ToFloatND) -> np.float64 | onp.ArrayND[np.float64]: ...

#
@overload
def boxcox(
    x: onp.ToFloat | onp.ToFloatND,
    lmbda: None = None,
    alpha: None = None,
    optimizer: _MinFun1D | None = None,
) -> tuple[_Float1D, np.float64]: ...
@overload
def boxcox(
    x: onp.ToFloat | onp.ToFloatND,
    lmbda: onp.ToFloat,
    alpha: float | np.floating[Any] | None = None,
    optimizer: _MinFun1D | None = None,
) -> _Float1D: ...
@overload
def boxcox(
    x: onp.ToFloat | onp.ToFloatND,
    lmbda: None,
    alpha: float | np.floating[Any],
    optimizer: _MinFun1D | None = None,
) -> tuple[_Float1D, np.float64, _Tuple2[float]]: ...
@overload
def boxcox(
    x: onp.ToFloat | onp.ToFloatND,
    lmbda: None = None,
    *,
    alpha: float | np.floating[Any],
    optimizer: _MinFun1D | None = None,
) -> tuple[_Float1D, np.float64, _Tuple2[float]]: ...

#
@overload
def boxcox_normmax(
    x: onp.ToFloat | onp.ToFloatND,
    brack: _Tuple2[onp.ToFloat] | None = None,
    method: Literal["pearsonr", "mle"] = "pearsonr",
    optimizer: _MinFun1D | None = None,
    *,
    ymax: onp.ToFloat | _BigFloat = ...,
) -> np.float64: ...
@overload
def boxcox_normmax(
    x: onp.ToFloat | onp.ToFloatND,
    brack: _Tuple2[onp.ToFloat] | None = None,
    *,
    method: Literal["all"],
    optimizer: _MinFun1D | None = None,
    ymax: onp.ToFloat | _BigFloat = ...,
) -> onp.Array1D[np.float64]: ...
@overload
def boxcox_normmax(
    x: onp.ToFloat | onp.ToFloatND,
    brack: _Tuple2[onp.ToFloat] | None,
    method: Literal["all"],
    optimizer: _MinFun1D | None = None,
    *,
    ymax: onp.ToFloat | _BigFloat = ...,
) -> onp.Array1D[np.float64]: ...

#
def boxcox_normplot(
    x: onp.ToFloat | onp.ToFloatND,
    la: onp.ToFloat,
    lb: onp.ToFloat,
    plot: _CanPlotText | ModuleType | None = None,
    N: onp.ToInt = 80,
) -> _Tuple2[onp.ArrayND[np.float64]]: ...

#
def yeojohnson_llf(lmb: onp.ToFloat, data: onp.ToFloatND) -> onp.Array0D[np.float64]: ...

#
@overload
def yeojohnson(x: onp.ToFloat | onp.ToFloatND, lmbda: None = None) -> tuple[_Float1D, np.float64]: ...
@overload
def yeojohnson(x: onp.ToFloat | onp.ToFloatND, lmbda: onp.ToFloat) -> _Float1D: ...

#
def yeojohnson_normmax(x: onp.ToFloat | onp.ToFloatND, brack: _Tuple2[onp.ToFloat] | None = None) -> np.float64: ...

#
def yeojohnson_normplot(
    x: onp.ToFloat | onp.ToFloatND,
    la: onp.ToFloat,
    lb: onp.ToFloat,
    plot: _CanPlotText | ModuleType | None = None,
    N: onp.ToInt = 80,
) -> _Tuple2[onp.ArrayND[np.float64]]: ...

#
def anderson(x: onp.ToFloat | onp.ToFloatND, dist: _RVCAnderson = "norm") -> AndersonResult: ...

#
def anderson_ksamp(
    samples: onp.ToFloatND,
    midrank: bool = True,
    *,
    method: PermutationMethod | None = None,
) -> Anderson_ksampResult: ...

#
@overload
def shapiro(
    x: onp.ToFloat | onp.ToFloatND,
    *,
    axis: None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Falsy = False,
) -> ShapiroResult[np.float64]: ...
@overload
def shapiro(
    x: onp.ToFloat | onp.ToFloatND,
    *,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Truthy,
) -> ShapiroResult[onp.ArrayND[np.float64]]: ...
@overload
def shapiro(
    x: onp.ToFloat | onp.ToFloatND,
    *,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> ShapiroResult: ...

#
@overload
def ansari(
    x: onp.ToFloat | onp.ToFloatND,
    y: onp.ToFloat | onp.ToFloatND,
    alternative: Alternative = "two-sided",
    *,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Falsy = False,
) -> AnsariResult[np.float64]: ...
@overload
def ansari(
    x: onp.ToFloat | onp.ToFloatND,
    y: onp.ToFloat | onp.ToFloatND,
    alternative: Alternative = "two-sided",
    *,
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: Truthy,
) -> AnsariResult[onp.ArrayND[np.float64]]: ...
@overload
def ansari(
    x: onp.ToFloat | onp.ToFloatND,
    y: onp.ToFloat | onp.ToFloatND,
    alternative: Alternative = "two-sided",
    *,
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> AnsariResult: ...

#
@overload
def bartlett(
    *samples: onp.ToFloatND,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Falsy = False,
) -> BartlettResult[np.float64]: ...
@overload
def bartlett(
    *samples: onp.ToFloatND,
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: Truthy,
) -> BartlettResult[onp.ArrayND[np.float64]]: ...
@overload
def bartlett(
    *samples: onp.ToFloatND,
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> BartlettResult: ...

#
@overload
def levene(
    *samples: onp.ToFloatND,
    center: _CenterMethod = "median",
    proportiontocut: onp.ToFloat = 0.05,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Falsy = False,
) -> LeveneResult[np.float64]: ...
@overload
def levene(
    *samples: onp.ToFloatND,
    center: _CenterMethod = "median",
    proportiontocut: onp.ToFloat = 0.05,
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: Truthy,
) -> LeveneResult[onp.ArrayND[np.float64]]: ...
@overload
def levene(
    *samples: onp.ToFloatND,
    center: _CenterMethod = "median",
    proportiontocut: onp.ToFloat = 0.05,
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> LeveneResult: ...

#
@overload
def fligner(
    *samples: onp.ToFloatND,
    center: _CenterMethod = "median",
    proportiontocut: onp.ToFloat = 0.05,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Falsy = False,
) -> FlignerResult[np.float64]: ...
@overload
def fligner(
    *samples: onp.ToFloatND,
    center: _CenterMethod = "median",
    proportiontocut: onp.ToFloat = 0.05,
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: Truthy,
) -> FlignerResult[onp.ArrayND[np.float64]]: ...
@overload
def fligner(
    *samples: onp.ToFloatND,
    center: _CenterMethod = "median",
    proportiontocut: onp.ToFloat = 0.05,
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> FlignerResult: ...

#
@overload
def mood(
    x: onp.ToFloat | onp.ToFloatND,
    y: onp.ToFloat | onp.ToFloatND,
    axis: None,
    alternative: Alternative = "two-sided",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: Falsy = False,
) -> SignificanceResult[np.float64]: ...
@overload
def mood(
    x: onp.ToFloat | onp.ToFloatND,
    y: onp.ToFloat | onp.ToFloatND,
    axis: op.CanIndex | None = 0,
    alternative: Alternative = "two-sided",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: Truthy,
) -> SignificanceResult[onp.ArrayND[np.float64]]: ...
@overload
def mood(
    x: onp.ToFloat | onp.ToFloatND,
    y: onp.ToFloat | onp.ToFloatND,
    axis: op.CanIndex | None = 0,
    alternative: Alternative = "two-sided",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: AnyBool = False,
) -> SignificanceResult[np.float64 | onp.ArrayND[np.float64]]: ...

#
@overload
def wilcoxon(
    x: onp.ToFloat | onp.ToFloatND,
    y: onp.ToFloat | onp.ToFloatND | None = None,
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "wilcox",
    correction: AnyBool = False,
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "approx"] | PermutationMethod = "auto",
    *,
    axis: None,
    nan_policy: NanPolicy = "propagate",
    keepdims: Falsy = False,
) -> WilcoxonResult[np.float64]: ...
@overload
def wilcoxon(
    x: onp.ToFloat | onp.ToFloatND,
    y: onp.ToFloat | onp.ToFloatND | None = None,
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "wilcox",
    correction: AnyBool = False,
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "approx", "asymptotic"] | PermutationMethod = "auto",
    *,
    axis: op.CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: Truthy,
) -> WilcoxonResult[onp.ArrayND[np.float64]]: ...
@overload
def wilcoxon(
    x: onp.ToFloat | onp.ToFloatND,
    y: onp.ToFloat | onp.ToFloatND | None = None,
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "wilcox",
    correction: AnyBool = False,
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "approx"] | PermutationMethod = "auto",
    *,
    axis: op.CanIndex | None = 0,
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
    *samples: onp.ToFloatND,
    ties: Literal["below", "above", "ignore"] = "below",
    correction: AnyBool = True,
    lambda_: onp.ToFloat | str = 1,
    nan_policy: NanPolicy = "propagate",
) -> MedianTestResult: ...

#
@overload
def circmean(
    samples: onp.ToFloatND,
    high: onp.ToFloat = ...,
    low: onp.ToFloat = 0,
    axis: None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: Falsy = False,
) -> np.float64: ...
@overload
def circmean(
    samples: onp.ToFloatND,
    high: onp.ToFloat = ...,
    low: onp.ToFloat = 0,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: Truthy,
) -> onp.ArrayND[np.float64]: ...
@overload
def circmean(
    samples: onp.ToFloatND,
    high: onp.ToFloat = ...,
    low: onp.ToFloat = 0,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: AnyBool = False,
) -> np.float64 | onp.ArrayND[np.float64]: ...

#
@overload
def circvar(
    samples: onp.ToFloatND,
    high: onp.ToFloat = ...,
    low: onp.ToFloat = 0,
    axis: None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: Falsy = False,
) -> np.float64: ...
@overload
def circvar(
    samples: onp.ToFloatND,
    high: onp.ToFloat = ...,
    low: onp.ToFloat = 0,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: Truthy,
) -> onp.ArrayND[np.float64]: ...
@overload
def circvar(
    samples: onp.ToFloatND,
    high: onp.ToFloat = ...,
    low: onp.ToFloat = 0,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: AnyBool = False,
) -> np.float64 | onp.ArrayND[np.float64]: ...

#
@overload
def circstd(
    samples: onp.ToFloatND,
    high: onp.ToFloat = ...,
    low: onp.ToFloat = 0,
    axis: None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    normalize: AnyBool = False,
    keepdims: Falsy = False,
) -> np.float64: ...
@overload
def circstd(
    samples: onp.ToFloatND,
    high: onp.ToFloat = ...,
    low: onp.ToFloat = 0,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    normalize: AnyBool = False,
    keepdims: Truthy,
) -> onp.ArrayND[np.float64]: ...
@overload
def circstd(
    samples: onp.ToFloatND,
    high: onp.ToFloat = ...,
    low: onp.ToFloat = 0,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    normalize: AnyBool = False,
    keepdims: AnyBool = False,
) -> np.float64 | onp.ArrayND[np.float64]: ...

#
def directional_stats(samples: onp.ToFloatND, *, axis: op.CanIndex | None = 0, normalize: AnyBool = True) -> DirectionalStats: ...

#
def false_discovery_control(
    ps: onp.ToFloat | onp.ToFloatND,
    *,
    axis: op.CanIndex | None = 0,
    method: Literal["bh", "by"] = "bh",
) -> onp.ArrayND[np.float64]: ...
