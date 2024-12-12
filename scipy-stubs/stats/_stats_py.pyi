from dataclasses import dataclass
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, Generic, Literal, Protocol, TypeAlias, final, overload, type_check_only
from typing_extensions import NamedTuple, Self, TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import optype.typing as opt
from scipy._typing import Alternative, NanPolicy, Seed
from ._resampling import BootstrapMethod, ResamplingMethod
from ._stats_mstats_common import siegelslopes, theilslopes
from ._typing import BaseBunch, PowerDivergenceStatistic

__all__ = [
    "alexandergovern",
    "brunnermunzel",
    "chisquare",
    "combine_pvalues",
    "cumfreq",
    "describe",
    "energy_distance",
    "expectile",
    "f_oneway",
    "find_repeats",
    "fisher_exact",
    "friedmanchisquare",
    "gmean",
    "gstd",
    "gzscore",
    "hmean",
    "iqr",
    "jarque_bera",
    "kendalltau",
    "kruskal",
    "ks_1samp",
    "ks_2samp",
    "kstest",
    "kurtosis",
    "kurtosistest",
    "linregress",
    "median_abs_deviation",
    "mode",
    "moment",
    "normaltest",
    "obrientransform",
    "pearsonr",
    "percentileofscore",
    "pmean",
    "pointbiserialr",
    "power_divergence",
    "quantile_test",
    "rankdata",
    "ranksums",
    "relfreq",
    "scoreatpercentile",
    "sem",
    "siegelslopes",
    "sigmaclip",
    "skew",
    "skewtest",
    "spearmanr",
    "theilslopes",
    "tiecorrect",
    "tmax",
    "tmean",
    "tmin",
    "trim1",
    "trim_mean",
    "trimboth",
    "tsem",
    "tstd",
    "ttest_1samp",
    "ttest_ind",
    "ttest_ind_from_stats",
    "ttest_rel",
    "tvar",
    "wasserstein_distance",
    "wasserstein_distance_nd",
    "weightedtau",
    "zmap",
    "zscore",
]

_SCT = TypeVar("_SCT", bound=np.generic, default=np.generic)

_Int0D: TypeAlias = np.integer[Any]
_Float0D: TypeAlias = np.floating[Any]
_Real0D: TypeAlias = _Int0D | _Float0D

_SCT_int = TypeVar("_SCT_int", bound=_Int0D, default=_Int0D)
_SCT_float = TypeVar("_SCT_float", bound=_Float0D, default=_Float0D)
_SCT_real = TypeVar("_SCT_real", bound=_Real0D, default=_Real0D)
_SCT_real_co = TypeVar("_SCT_real_co", covariant=True, bound=_Real0D, default=_Real0D)

_GenericND: TypeAlias = _SCT | onp.ArrayND[_SCT]
_IntND: TypeAlias = _GenericND[_SCT_int]
_FloatND: TypeAlias = _GenericND[_SCT_float]
_RealND: TypeAlias = _GenericND[_SCT_real]

_Interpolation: TypeAlias = Literal["linear", "lower", "higher", "nearest", "midpoint"]

_NDT_int_co = TypeVar("_NDT_int_co", covariant=True, bound=int | _IntND, default=int | _IntND)
_NDT_float = TypeVar("_NDT_float", bound=float | _FloatND, default=float | _FloatND)
_NDT_float_co = TypeVar("_NDT_float_co", covariant=True, bound=float | _FloatND, default=float | _FloatND)
_NDT_real_co = TypeVar("_NDT_real_co", covariant=True, bound=float | _RealND, default=float | _RealND)

@type_check_only
class _RVSCallable(Protocol):
    def __call__(self, /, *, size: int | tuple[int, ...]) -> onp.ArrayND[np.floating[Any]]: ...

@final
class _SimpleNormal:
    @overload
    def cdf(self, /, x: np.float64) -> np.float64: ...
    @overload  # this is a workaround for yet another mypy bug...
    def cdf(self, /, x: np.float16 | np.float32 | np.float64) -> np.float32 | np.float64: ...
    @overload
    def cdf(self, /, x: np.uint32 | np.int32 | np.uint64 | np.int64) -> np.float64: ...
    @overload
    def cdf(self, /, x: np.uint8 | np.int8 | np.uint16 | np.int16) -> np.float32: ...
    @overload  # for some reason mypy only understands this if the `np.bool_` overload is separate...
    def cdf(self, /, x: np.bool_) -> np.float32: ...
    @overload  # this is a workaround for --- yes, you guessed it -- another mypy bug!
    def cdf(self, /, x: np.complex128) -> np.complex128: ...
    @overload
    def cdf(self, /, x: np.complex64 | np.clongdouble) -> np.complex64 | np.complex128: ...
    @overload
    def cdf(self, /, x: bool) -> np.float32: ...
    @overload
    def cdf(self, /, x: opt.JustInt) -> np.float64: ...
    @overload
    def cdf(self, /, x: float) -> np.float32 | np.float64: ...
    @overload
    def cdf(self, /, x: complex) -> np.complex128 | np.float64 | np.float32: ...
    sf = cdf

@final
class _SimpleChi2:
    df: int
    def __init__(self, /, df: int) -> None: ...
    @overload
    def sf(self, /, x: float | np.float64 | np.uint64 | np.int64 | np.uint32 | np.int32 | np.bool_) -> np.float64: ...
    @overload
    def sf(self, /, x: np.uint16 | np.int16 | np.uint8 | np.int8) -> np.float32: ...
    @overload  # workaround for a mypy bug on numpy>=2.2
    def sf(self, /, x: np.float32) -> np.float32 | np.float64: ...

@type_check_only
class _TestResultTuple(NamedTuple, Generic[_NDT_float_co]):
    statistic: _NDT_float_co
    pvalue: _NDT_float_co

class SkewtestResult(_TestResultTuple[_NDT_float_co], Generic[_NDT_float_co]): ...
class KurtosistestResult(_TestResultTuple[_NDT_float_co], Generic[_NDT_float_co]): ...
class NormaltestResult(_TestResultTuple[_NDT_float_co], Generic[_NDT_float_co]): ...
class Ttest_indResult(_TestResultTuple[_NDT_float_co], Generic[_NDT_float_co]): ...
class Power_divergenceResult(_TestResultTuple[_NDT_float_co], Generic[_NDT_float_co]): ...
class RanksumsResult(_TestResultTuple[_NDT_float_co], Generic[_NDT_float_co]): ...
class KruskalResult(_TestResultTuple[_NDT_float_co], Generic[_NDT_float_co]): ...
class FriedmanchisquareResult(_TestResultTuple[_NDT_float_co], Generic[_NDT_float_co]): ...
class BrunnerMunzelResult(_TestResultTuple[_NDT_float_co], Generic[_NDT_float_co]): ...
class F_onewayResult(_TestResultTuple[_NDT_float_co], Generic[_NDT_float_co]): ...

class ConfidenceInterval(NamedTuple, Generic[_NDT_float_co]):
    low: _NDT_float_co
    high: _NDT_float_co

class DescribeResult(NamedTuple, Generic[_NDT_real_co, _NDT_float_co]):
    nobs: int
    minmax: tuple[_NDT_real_co, _NDT_real_co]
    mean: _NDT_float_co
    variance: _NDT_float_co
    skewness: _NDT_float_co
    kurtosis: _NDT_float_co

class ModeResult(NamedTuple, Generic[_NDT_real_co, _NDT_int_co]):
    mode: _NDT_real_co
    count: _NDT_int_co  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]

class HistogramResult(NamedTuple):
    count: onp.Array1D[np.float64]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]
    lowerlimit: Literal[0] | np.floating[Any]
    binsize: onp.Array1D[np.float64]
    extrapoints: int

class CumfreqResult(NamedTuple):
    cumcount: onp.Array1D[np.float64]
    lowerlimit: Literal[0] | np.floating[Any]
    binsize: onp.Array1D[np.float64]
    extrapoints: int

class RelfreqResult(NamedTuple):
    frequency: onp.Array1D[np.float64]
    lowerlimit: Literal[0] | np.floating[Any]
    binsize: onp.Array1D[np.float64]
    extrapoints: int

class SigmaclipResult(NamedTuple, Generic[_SCT_real_co, _NDT_float_co]):
    clipped: onp.Array1D[_SCT_real_co]
    lower: _NDT_float_co
    upper: _NDT_float_co

class RepeatedResults(NamedTuple):
    values: onp.Array1D[np.float64]
    counts: onp.Array1D[np.intp]

@dataclass
class AlexanderGovernResult:
    statistic: float
    pvalue: float

@dataclass
class QuantileTestResult:
    statistic: float
    statistic_type: int
    pvalue: float
    _alternative: list[str]
    _x: onp.ArrayND[_Real0D]
    _p: float
    def confidence_interval(self, /, confidence_level: float = 0.95) -> float: ...

@type_check_only
class _TestResultBunch(BaseBunch[_NDT_float_co, _NDT_float_co], Generic[_NDT_float_co]):  # pyright: ignore[reportInvalidTypeArguments]
    @property
    def statistic(self, /) -> _NDT_float_co: ...
    @property
    def pvalue(self, /) -> _NDT_float_co: ...
    def __new__(_cls, statistic: _NDT_float_co, pvalue: _NDT_float_co) -> Self: ...
    def __init__(self, /, statistic: _NDT_float_co, pvalue: _NDT_float_co) -> None: ...

class SignificanceResult(_TestResultBunch[_NDT_float_co], Generic[_NDT_float_co]): ...
class PearsonRResultBase(_TestResultBunch[_NDT_float_co], Generic[_NDT_float_co]): ...

class PearsonRResult(PearsonRResultBase[_NDT_float_co], Generic[_NDT_float_co]):
    _alternative: Alternative
    _n: int
    _x: onp.ArrayND[_Real0D]
    _y: onp.ArrayND[_Real0D]
    _axis: int
    correlation: _NDT_float_co  # alias for `statistic`
    def __init__(  # pyright: ignore[reportInconsistentConstructor]
        self,
        /,
        statistic: _NDT_float_co,
        pvalue: _NDT_float_co,
        alternative: Alternative,
        n: int,
        x: onp.ArrayND[_Real0D],
        y: onp.ArrayND[_Real0D],
        axis: int,
    ) -> None: ...
    def confidence_interval(
        self,
        /,
        confidence_level: float = 0.95,
        method: BootstrapMethod | None = None,
    ) -> ConfidenceInterval[_NDT_float_co]: ...

class TtestResultBase(_TestResultBunch[_NDT_float_co], Generic[_NDT_float_co]):
    @property
    def df(self, /) -> _NDT_float_co: ...
    def __new__(_cls, statistic: _NDT_float_co, pvalue: _NDT_float_co, *, df: _NDT_float_co) -> Self: ...
    def __init__(self, /, statistic: _NDT_float_co, pvalue: _NDT_float_co, *, df: _NDT_float_co) -> None: ...

class TtestResult(TtestResultBase[_NDT_float_co], Generic[_NDT_float_co]):
    _alternative: Alternative
    _standard_error: _NDT_float_co
    _estimate: _NDT_float_co
    _statistic_np: _NDT_float_co
    _dtype: np.dtype[np.floating[Any]]
    _xp: ModuleType

    def __init__(  # pyright: ignore[reportInconsistentConstructor]
        self,
        /,
        statistic: _NDT_float_co,
        pvalue: _NDT_float_co,
        df: _NDT_float_co,
        alternative: Alternative,
        standard_error: _NDT_float_co,
        estimate: _NDT_float_co,
        statistic_np: _NDT_float_co | None = None,
        xp: ModuleType | None = None,
    ) -> None: ...
    def confidence_interval(self, /, confidence_level: float = 0.95) -> ConfidenceInterval[_NDT_float_co]: ...

class KstestResult(_TestResultBunch[np.float64]):
    @property
    def statistic_location(self, /) -> np.float64: ...
    @property
    def statistic_sign(self, /) -> np.int8: ...
    def __new__(
        _cls,
        statistic: np.float64,
        pvalue: np.float64,
        *,
        statistic_location: np.float64,
        statistic_sign: np.int8,
    ) -> Self: ...
    def __init__(
        self,
        /,
        statistic: np.float64,
        pvalue: np.float64,
        *,
        statistic_location: np.float64,
        statistic_sign: np.int8,
    ) -> None: ...

Ks_2sampResult = KstestResult

class LinregressResult(BaseBunch[np.float64, np.float64, np.float64, float | np.float64, float | np.float64]):
    @property
    def slope(self, /) -> np.float64: ...
    @property
    def intercept(self, /) -> np.float64: ...
    @property
    def rvalue(self, /) -> np.float64: ...
    @property
    def pvalue(self, /) -> float | np.float64: ...
    @property
    def stderr(self, /) -> float | np.float64: ...
    @property
    def intercept_stderr(self, /) -> float | np.float64: ...
    def __new__(
        _cls,
        slope: np.float64,
        intercept: np.float64,
        rvalue: np.float64,
        pvalue: float | np.float64,
        stderr: float | np.float64,
        *,
        intercept_stderr: float | np.float64,
    ) -> Self: ...
    def __init__(
        self,
        /,
        slope: np.float64,
        intercept: np.float64,
        rvalue: np.float64,
        pvalue: float | np.float64,
        stderr: float | np.float64,
        *,
        intercept_stderr: float | np.float64,
    ) -> None: ...

def gmean(
    a: onp.ToFloatND,
    axis: int | None = 0,
    dtype: npt.DTypeLike | None = None,
    weights: onp.ToFloatND | None = None,
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> _RealND: ...
def hmean(
    a: onp.ToFloatND,
    axis: int | None = 0,
    dtype: npt.DTypeLike | None = None,
    *,
    weights: onp.ToFloatND | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> _RealND: ...
def pmean(
    a: onp.ToFloatND,
    p: float | _Real0D,
    *,
    axis: int | None = 0,
    dtype: npt.DTypeLike | None = None,
    weights: onp.ToFloatND | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> _RealND: ...

#
def mode(a: onp.ToFloatND, axis: int | None = 0, nan_policy: NanPolicy = "propagate", keepdims: bool = False) -> _RealND: ...

#
def tmean(
    a: onp.ToFloatND,
    limits: tuple[float | _Real0D, float | _Real0D] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: int | None = None,
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> _FloatND: ...
def tvar(
    a: onp.ToFloatND,
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: int | None = 0,
    ddof: int = 1,
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> _FloatND: ...
def tmin(
    a: onp.ToFloatND,
    lowerlimit: float | _Real0D | None = None,
    axis: int | None = 0,
    inclusive: bool = True,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: bool = False,
) -> _RealND: ...
def tmax(
    a: onp.ToFloatND,
    upperlimit: float | _Real0D | None = None,
    axis: int | None = 0,
    inclusive: bool = True,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: bool = False,
) -> _RealND: ...
def tstd(
    a: onp.ToFloatND,
    limits: tuple[float | _Real0D, float | _Real0D] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: int | None = 0,
    ddof: int = 1,
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> _FloatND: ...
def tsem(
    a: onp.ToFloatND,
    limits: tuple[float | _Real0D, float | _Real0D] | None = None,
    inclusive: tuple[bool, bool] = (True, True),
    axis: int | None = 0,
    ddof: int = 1,
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> _FloatND: ...

#
def moment(
    a: onp.ToFloatND,
    order: int = 1,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    *,
    center: float | _Float0D | None = None,
    keepdims: bool = False,
) -> _FloatND: ...
def skew(
    a: onp.ToFloatND,
    axis: int | None = 0,
    bias: bool = True,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: bool = False,
) -> _FloatND: ...
def kurtosis(
    a: onp.ToFloatND,
    axis: int | None = 0,
    fisher: bool = True,
    bias: bool = True,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: bool = False,
) -> _FloatND: ...
def describe(
    a: onp.ToFloatND,
    axis: int | None = 0,
    ddof: int = 1,
    bias: bool = True,
    nan_policy: NanPolicy = "propagate",
) -> DescribeResult: ...

#
def skewtest(
    a: onp.ToFloatND,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    alternative: Alternative = "two-sided",
    *,
    keepdims: bool = False,
) -> SkewtestResult: ...
def kurtosistest(
    a: onp.ToFloatND,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    alternative: Alternative = "two-sided",
    *,
    keepdims: bool = False,
) -> KurtosistestResult: ...
def normaltest(
    a: onp.ToFloatND,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: bool = False,
) -> NormaltestResult: ...
def jarque_bera(
    x: onp.ToFloatND,
    *,
    axis: int | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> SignificanceResult: ...

#
def scoreatpercentile(
    a: onp.ToFloat1D,
    per: onp.ToFloat | onp.ToFloatND,
    limit: tuple[float | _Real0D, float | _Real0D] | tuple[()] = (),
    interpolation_method: Literal["fraction", "lower", "higher"] = "fraction",
    axis: int | None = None,
) -> _FloatND: ...
def percentileofscore(
    a: onp.ToFloat1D,
    score: onp.ToFloat | onp.ToFloatND,
    kind: Literal["rank", "weak", "strict", "mean"] = "rank",
    nan_policy: NanPolicy = "propagate",
) -> float | np.float64: ...

#
def cumfreq(
    a: onp.ToFloatND,
    numbins: int = 10,
    defaultreallimits: tuple[float | _Real0D, float | _Real0D] | None = None,
    weights: onp.ToFloatND | None = None,
) -> CumfreqResult: ...
def relfreq(
    a: onp.ToFloatND,
    numbins: int = 10,
    defaultreallimits: tuple[float | _Real0D, float | _Real0D] | None = None,
    weights: onp.ToFloatND | None = None,
) -> RelfreqResult: ...

#
def obrientransform(*samples: onp.ToFloatND) -> onp.Array2D[_Float0D] | onp.Array1D[np.object_]: ...

#
def sem(
    a: onp.ToFloatND,
    axis: int | None = 0,
    ddof: int = 1,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: bool = False,
) -> _FloatND: ...
def zscore(
    a: onp.ToFloatND,
    axis: int | None = 0,
    ddof: int = 0,
    nan_policy: NanPolicy = "propagate",
) -> onp.ArrayND[_Float0D]: ...
def gzscore(
    a: onp.ToFloatND,
    *,
    axis: int | None = 0,
    ddof: int = 0,
    nan_policy: NanPolicy = "propagate",
) -> onp.ArrayND[_Float0D]: ...
def zmap(
    scores: onp.ToFloatND,
    compare: onp.ToFloatND,
    axis: int | None = 0,
    ddof: int = 0,
    nan_policy: NanPolicy = "propagate",
) -> onp.ArrayND[_Float0D]: ...

#
def gstd(a: onp.ToFloatND, axis: int | None = 0, ddof: int = 1) -> _FloatND: ...
def iqr(
    x: onp.ToFloatND,
    axis: int | Sequence[int] | None = None,
    rng: tuple[float, float] = (25, 75),
    scale: Literal["normal"] | onp.ToFloat | onp.ToFloatND = 1.0,
    nan_policy: NanPolicy = "propagate",
    interpolation: _Interpolation = "linear",
    keepdims: bool = False,
) -> _FloatND: ...
def median_abs_deviation(
    x: onp.ToFloatND,
    axis: int | None = 0,
    center: np.ufunc | Callable[[_NDT_float, int | None], _NDT_float] = ...,
    scale: Literal["normal"] | float = 1.0,
    nan_policy: NanPolicy = "propagate",
) -> _FloatND: ...

#
def sigmaclip(a: onp.ToFloatND, low: float = 4.0, high: float = 4.0) -> SigmaclipResult: ...
def trimboth(a: onp.ToFloatND, proportiontocut: float, axis: int | None = 0) -> onp.ArrayND[_Real0D]: ...
def trim1(a: onp.ToFloatND, proportiontocut: float, tail: str = "right", axis: int | None = 0) -> onp.ArrayND[_Real0D]: ...
def trim_mean(a: onp.ToFloatND, proportiontocut: float, axis: int | None = 0) -> _FloatND: ...

#
def f_oneway(
    *samples: onp.ToFloatND,
    nan_policy: NanPolicy = "propagate",
    axis: int | None = 0,
    keepdims: bool = False,
) -> F_onewayResult: ...
def alexandergovern(
    *samples: onp.ToFloatND,
    nan_policy: NanPolicy = "propagate",
    axis: int | None = 0,
    keepdims: bool = False,
) -> AlexanderGovernResult: ...
def pearsonr(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    *,
    alternative: Alternative = "two-sided",
    method: ResamplingMethod | None = None,
    axis: int | None = 0,
) -> PearsonRResult: ...
def fisher_exact(
    table: onp.ArrayND[_Real0D],
    alternative: Alternative = "two-sided",
) -> SignificanceResult[float]: ...

#
def spearmanr(
    a: onp.ToFloatND,
    b: onp.ToFloatND | None = None,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    alternative: Alternative = "two-sided",
) -> SignificanceResult: ...
def pointbiserialr(x: onp.ToBoolND, y: onp.ToFloatND) -> SignificanceResult[float]: ...
def kendalltau(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    *,
    nan_policy: NanPolicy = "propagate",
    method: Literal["auto", "asymptotic", "exact"] = "auto",
    variant: Literal["b", "c"] = "b",
    alternative: Alternative = "two-sided",
) -> SignificanceResult[float]: ...
def weightedtau(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    rank: onp.ToInt | onp.ToIntND = True,
    weigher: Callable[[int], float | _Real0D] | None = None,
    additive: bool = True,
) -> SignificanceResult[float]: ...

#
def pack_TtestResult(
    statistic: _NDT_float,
    pvalue: _NDT_float,
    df: _NDT_float,
    alternative: Alternative,
    standard_error: _NDT_float,
    estimate: _NDT_float,
) -> TtestResult[_NDT_float]: ...
def unpack_TtestResult(
    res: TtestResult[_NDT_float],
) -> tuple[
    _NDT_float,
    _NDT_float,
    _NDT_float,
    Alternative,
    _NDT_float,
    _NDT_float,
]: ...

#
def ttest_1samp(
    a: onp.ToFloatND,
    popmean: onp.ToFloat | onp.ToFloatND,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    alternative: Alternative = "two-sided",
    *,
    keepdims: bool = False,
) -> TtestResult: ...
def ttest_ind_from_stats(
    mean1: onp.ToFloat | onp.ToFloatND,
    std1: onp.ToFloat | onp.ToFloatND,
    nobs1: onp.ToInt | onp.ToIntND,
    mean2: onp.ToFloat | onp.ToFloatND,
    std2: onp.ToFloat | onp.ToFloatND,
    nobs2: onp.ToInt | onp.ToIntND,
    equal_var: bool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult: ...
def ttest_ind(
    a: onp.ToFloatND,
    b: onp.ToFloatND,
    axis: int | None = 0,
    equal_var: bool = True,
    nan_policy: NanPolicy = "propagate",
    permutations: float | None = None,
    random_state: Seed | None = None,
    alternative: Alternative = "two-sided",
    trim: float = 0,
    *,
    keepdims: bool = False,
) -> TtestResult: ...
def ttest_rel(
    a: onp.ToFloatND,
    b: onp.ToFloatND,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    alternative: Alternative = "two-sided",
    *,
    keepdims: bool = False,
) -> TtestResult: ...

#
def power_divergence(
    f_obs: onp.ToFloatND,
    f_exp: onp.ToFloatND | None = None,
    ddof: int = 0,
    axis: int | None = 0,
    lambda_: PowerDivergenceStatistic | float | None = None,
) -> Power_divergenceResult: ...
def chisquare(
    f_obs: onp.ToFloatND,
    f_exp: onp.ToFloatND | None = None,
    ddof: int = 0,
    axis: int | None = 0,
) -> Power_divergenceResult: ...

#
def ks_1samp(
    x: onp.ToFloatND,
    cdf: Callable[[float], float | _Real0D],
    args: tuple[object, ...] = (),
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "approx", "asymp"] = "auto",
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> KstestResult: ...
def ks_2samp(
    data1: onp.ToFloatND,
    data2: onp.ToFloatND,
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "asymp"] = "auto",
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> KstestResult: ...
def kstest(
    rvs: str | onp.ToFloatND | _RVSCallable,
    cdf: str | onp.ToFloatND | Callable[[float], float | _Float0D],
    args: tuple[object, ...] = (),
    N: int = 20,
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "approx", "asymp"] = "auto",
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> KstestResult: ...

#
def tiecorrect(rankvals: onp.ToInt | onp.ToIntND) -> float | np.float64: ...

#
def ranksums(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    alternative: Alternative = "two-sided",
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> RanksumsResult: ...

#
def kruskal(
    *samples: onp.ToFloatND,
    nan_policy: NanPolicy = "propagate",
    axis: int | None = 0,
    keepdims: bool = False,
) -> KruskalResult: ...
def friedmanchisquare(
    *samples: onp.ToFloatND,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> FriedmanchisquareResult: ...
def brunnermunzel(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    alternative: Alternative = "two-sided",
    distribution: Literal["t", "normal"] = "t",
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: bool = False,
    axis: int | None = 0,
) -> BrunnerMunzelResult: ...

#
def combine_pvalues(
    pvalues: onp.ToFloatND,
    method: Literal["fisher", "pearson", "tippett", "stouffer", "mudholkar_george"] = "fisher",
    weights: onp.ToFloatND | None = None,
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> SignificanceResult: ...

#
def quantile_test_iv(  # undocumented
    x: onp.ToFloatND,
    q: float | _Real0D,
    p: float | _Float0D,
    alternative: Alternative,
) -> tuple[onp.ArrayND[_Real0D], _Real0D, np.floating[Any], Alternative]: ...
def quantile_test(
    x: onp.ToFloatND,
    *,
    q: float | _Real0D = 0,
    p: float | _Float0D = 0.5,
    alternative: Alternative = "two-sided",
) -> QuantileTestResult: ...

#
def wasserstein_distance_nd(
    u_values: onp.ToFloatND,
    v_values: onp.ToFloatND,
    u_weights: onp.ToFloatND | None = None,
    v_weights: onp.ToFloatND | None = None,
) -> float | np.float64: ...
def wasserstein_distance(
    u_values: onp.ToFloatND,
    v_values: onp.ToFloatND,
    u_weights: onp.ToFloatND | None = None,
    v_weights: onp.ToFloatND | None = None,
) -> np.float64: ...
def energy_distance(
    u_values: onp.ToFloatND,
    v_values: onp.ToFloatND,
    u_weights: onp.ToFloatND | None = None,
    v_weights: onp.ToFloatND | None = None,
) -> np.float64: ...

#
def find_repeats(arr: onp.ToFloatND) -> RepeatedResults: ...
def rankdata(
    a: onp.ToFloatND,
    method: Literal["average", "min", "max", "dense", "ordinal"] = "average",
    *,
    axis: int | None = None,
    nan_policy: NanPolicy = "propagate",
) -> onp.ArrayND[_Real0D]: ...
def expectile(a: onp.ToFloatND, alpha: float = 0.5, *, weights: onp.ToFloatND | None = None) -> np.float64: ...
def linregress(x: onp.ToFloatND, y: onp.ToFloatND | None = None, alternative: Alternative = "two-sided") -> LinregressResult: ...
