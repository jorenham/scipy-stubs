from dataclasses import dataclass
from typing import NamedTuple

from scipy._typing import Untyped
from ._stats_mstats_common import siegelslopes, theilslopes

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

SignificanceResult: Untyped

def gmean(a, axis: int = 0, dtype: Untyped | None = None, weights: Untyped | None = None) -> Untyped: ...
def hmean(a, axis: int = 0, dtype: Untyped | None = None, *, weights: Untyped | None = None) -> Untyped: ...
def pmean(a, p, *, axis: int = 0, dtype: Untyped | None = None, weights: Untyped | None = None) -> Untyped: ...

class ModeResult(NamedTuple):
    mode: Untyped
    count: Untyped  # type: ignore[assignment]

def mode(a, axis: int = 0, nan_policy: str = "propagate", keepdims: bool = False) -> Untyped: ...
def tmean(a, limits: Untyped | None = None, inclusive=(True, True), axis: Untyped | None = None) -> Untyped: ...
def tvar(a, limits: Untyped | None = None, inclusive=(True, True), axis: int = 0, ddof: int = 1) -> Untyped: ...
def tmin(
    a,
    lowerlimit: Untyped | None = None,
    axis: int = 0,
    inclusive: bool = True,
    nan_policy: str = "propagate",
) -> Untyped: ...
def tmax(
    a,
    upperlimit: Untyped | None = None,
    axis: int = 0,
    inclusive: bool = True,
    nan_policy: str = "propagate",
) -> Untyped: ...
def tstd(a, limits: Untyped | None = None, inclusive=(True, True), axis: int = 0, ddof: int = 1) -> Untyped: ...
def tsem(a, limits: Untyped | None = None, inclusive=(True, True), axis: int = 0, ddof: int = 1) -> Untyped: ...
def moment(a, order: int = 1, axis: int = 0, nan_policy: str = "propagate", *, center: Untyped | None = None) -> Untyped: ...
def skew(a, axis: int = 0, bias: bool = True, nan_policy: str = "propagate") -> Untyped: ...
def kurtosis(a, axis: int = 0, fisher: bool = True, bias: bool = True, nan_policy: str = "propagate") -> Untyped: ...

class DescribeResult(NamedTuple):
    nobs: Untyped
    minmax: Untyped
    mean: Untyped
    variance: Untyped
    skewness: Untyped
    kurtosis: Untyped

def describe(a, axis: int = 0, ddof: int = 1, bias: bool = True, nan_policy: str = "propagate") -> Untyped: ...

class SkewtestResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def skewtest(a, axis: int = 0, nan_policy: str = "propagate", alternative: str = "two-sided") -> Untyped: ...

class KurtosistestResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def kurtosistest(a, axis: int = 0, nan_policy: str = "propagate", alternative: str = "two-sided") -> Untyped: ...

class NormaltestResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def normaltest(a, axis: int = 0, nan_policy: str = "propagate") -> Untyped: ...
def jarque_bera(x, *, axis: Untyped | None = None) -> Untyped: ...
def scoreatpercentile(a, per, limit=(), interpolation_method: str = "fraction", axis: Untyped | None = None) -> Untyped: ...
def percentileofscore(a, score, kind: str = "rank", nan_policy: str = "propagate") -> Untyped: ...

class HistogramResult(NamedTuple):
    count: Untyped  # type: ignore[assignment]
    lowerlimit: Untyped
    binsize: Untyped
    extrapoints: Untyped

class CumfreqResult(NamedTuple):
    cumcount: Untyped
    lowerlimit: Untyped
    binsize: Untyped
    extrapoints: Untyped

def cumfreq(a, numbins: int = 10, defaultreallimits: Untyped | None = None, weights: Untyped | None = None) -> Untyped: ...

class RelfreqResult(NamedTuple):
    frequency: Untyped
    lowerlimit: Untyped
    binsize: Untyped
    extrapoints: Untyped

def relfreq(a, numbins: int = 10, defaultreallimits: Untyped | None = None, weights: Untyped | None = None) -> Untyped: ...
def obrientransform(*samples) -> Untyped: ...
def sem(a, axis: int = 0, ddof: int = 1, nan_policy: str = "propagate") -> Untyped: ...
def zscore(a, axis: int = 0, ddof: int = 0, nan_policy: str = "propagate") -> Untyped: ...
def gzscore(a, *, axis: int = 0, ddof: int = 0, nan_policy: str = "propagate") -> Untyped: ...
def zmap(scores, compare, axis: int = 0, ddof: int = 0, nan_policy: str = "propagate") -> Untyped: ...
def gstd(a, axis: int = 0, ddof: int = 1) -> Untyped: ...
def iqr(
    x,
    axis: Untyped | None = None,
    rng=(25, 75),
    scale: float = 1.0,
    nan_policy: str = "propagate",
    interpolation: str = "linear",
    keepdims: bool = False,
) -> Untyped: ...
def median_abs_deviation(x, axis: int = 0, center=..., scale: float = 1.0, nan_policy: str = "propagate") -> Untyped: ...

class SigmaclipResult(NamedTuple):
    clipped: Untyped
    lower: Untyped
    upper: Untyped

def sigmaclip(a, low: float = 4.0, high: float = 4.0) -> Untyped: ...
def trimboth(a, proportiontocut, axis: int = 0) -> Untyped: ...
def trim1(a, proportiontocut, tail: str = "right", axis: int = 0) -> Untyped: ...
def trim_mean(a, proportiontocut, axis: int = 0) -> Untyped: ...

class F_onewayResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def f_oneway(*samples, axis: int = 0) -> Untyped: ...
@dataclass
class AlexanderGovernResult:
    statistic: float
    pvalue: float

def alexandergovern(*samples, nan_policy: str = "propagate", axis: int = 0) -> Untyped: ...

class ConfidenceInterval(NamedTuple):
    low: Untyped
    high: Untyped

class PearsonRResultBase(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class PearsonRResult(PearsonRResultBase):
    correlation: Untyped
    def __init__(self, statistic, pvalue, alternative, n, x, y, axis) -> None: ...
    def confidence_interval(self, confidence_level: float = 0.95, method: Untyped | None = None) -> Untyped: ...

def pearsonr(x, y, *, alternative: str = "two-sided", method: Untyped | None = None, axis: int = 0) -> Untyped: ...
def fisher_exact(table, alternative: str = "two-sided") -> Untyped: ...
def spearmanr(
    a,
    b: Untyped | None = None,
    axis: int = 0,
    nan_policy: str = "propagate",
    alternative: str = "two-sided",
) -> Untyped: ...
def pointbiserialr(x, y) -> Untyped: ...
def kendalltau(
    x,
    y,
    *,
    nan_policy: str = "propagate",
    method: str = "auto",
    variant: str = "b",
    alternative: str = "two-sided",
) -> Untyped: ...
def weightedtau(x, y, rank: bool = True, weigher: Untyped | None = None, additive: bool = True) -> Untyped: ...

class TtestResultBase(NamedTuple):
    statistic: Untyped
    pvalue: Untyped
    @property
    def df(self) -> Untyped: ...

class TtestResult(TtestResultBase):
    def __init__(
        self,
        statistic,
        pvalue,
        df,
        alternative,
        standard_error,
        estimate,
        statistic_np: Untyped | None = None,
        xp: Untyped | None = None,
    ): ...
    def confidence_interval(self, confidence_level: float = 0.95) -> Untyped: ...

def pack_TtestResult(statistic, pvalue, df, alternative, standard_error, estimate) -> Untyped: ...
def unpack_TtestResult(res) -> Untyped: ...
def ttest_1samp(a, popmean, axis: int = 0, nan_policy: str = "propagate", alternative: str = "two-sided") -> Untyped: ...

class Ttest_indResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def ttest_ind_from_stats(
    mean1,
    std1,
    nobs1,
    mean2,
    std2,
    nobs2,
    equal_var: bool = True,
    alternative: str = "two-sided",
) -> Untyped: ...
def ttest_ind(
    a,
    b,
    axis: int = 0,
    equal_var: bool = True,
    nan_policy: str = "propagate",
    permutations: Untyped | None = None,
    random_state: Untyped | None = None,
    alternative: str = "two-sided",
    trim: int = 0,
) -> Untyped: ...
def ttest_rel(a, b, axis: int = 0, nan_policy: str = "propagate", alternative: str = "two-sided") -> Untyped: ...

class Power_divergenceResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def power_divergence(
    f_obs,
    f_exp: Untyped | None = None,
    ddof: int = 0,
    axis: int = 0,
    lambda_: Untyped | None = None,
) -> Untyped: ...
def chisquare(f_obs, f_exp: Untyped | None = None, ddof: int = 0, axis: int = 0) -> Untyped: ...

KstestResult: Untyped

def ks_1samp(x, cdf, args=(), alternative: str = "two-sided", method: str = "auto") -> Untyped: ...

Ks_2sampResult = KstestResult

def ks_2samp(data1, data2, alternative: str = "two-sided", method: str = "auto") -> Untyped: ...
def kstest(rvs, cdf, args=(), N: int = 20, alternative: str = "two-sided", method: str = "auto") -> Untyped: ...
def tiecorrect(rankvals) -> Untyped: ...

class RanksumsResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def ranksums(x, y, alternative: str = "two-sided") -> Untyped: ...

class KruskalResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def kruskal(*samples, nan_policy: str = "propagate") -> Untyped: ...

class FriedmanchisquareResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def friedmanchisquare(*samples) -> Untyped: ...

class BrunnerMunzelResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def brunnermunzel(x, y, alternative: str = "two-sided", distribution: str = "t", nan_policy: str = "propagate") -> Untyped: ...
def combine_pvalues(pvalues, method: str = "fisher", weights: Untyped | None = None, *, axis: int = 0) -> Untyped: ...
@dataclass
class QuantileTestResult:
    statistic: float
    statistic_type: int
    pvalue: float
    def confidence_interval(self, confidence_level: float = 0.95) -> Untyped: ...

def quantile_test_iv(x, q, p, alternative) -> Untyped: ...
def quantile_test(x, *, q: int = 0, p: float = 0.5, alternative: str = "two-sided") -> Untyped: ...
def wasserstein_distance_nd(
    u_values,
    v_values,
    u_weights: Untyped | None = None,
    v_weights: Untyped | None = None,
) -> Untyped: ...
def wasserstein_distance(u_values, v_values, u_weights: Untyped | None = None, v_weights: Untyped | None = None) -> Untyped: ...
def energy_distance(u_values, v_values, u_weights: Untyped | None = None, v_weights: Untyped | None = None) -> Untyped: ...

class RepeatedResults(NamedTuple):
    values: Untyped
    counts: Untyped

def find_repeats(arr) -> Untyped: ...
def rankdata(a, method: str = "average", *, axis: Untyped | None = None, nan_policy: str = "propagate") -> Untyped: ...
def expectile(a, alpha: float = 0.5, *, weights: Untyped | None = None) -> Untyped: ...

LinregressResult: Untyped

def linregress(x, y: Untyped | None = None, alternative: str = "two-sided") -> Untyped: ...

class _SimpleNormal:
    def cdf(self, x) -> Untyped: ...
    def sf(self, x) -> Untyped: ...
    def isf(self, x) -> Untyped: ...

class _SimpleChi2:
    df: Untyped
    def __init__(self, df) -> None: ...
    def cdf(self, x) -> Untyped: ...
    def sf(self, x) -> Untyped: ...

class _SimpleBeta:
    a: Untyped
    b: Untyped
    loc: Untyped
    scale: Untyped
    def __init__(self, a, b, *, loc: Untyped | None = None, scale: Untyped | None = None): ...
    def cdf(self, x) -> Untyped: ...
    def sf(self, x) -> Untyped: ...

class _SimpleStudentT:
    df: Untyped
    def __init__(self, df) -> None: ...
    def cdf(self, t) -> Untyped: ...
    def sf(self, t) -> Untyped: ...
