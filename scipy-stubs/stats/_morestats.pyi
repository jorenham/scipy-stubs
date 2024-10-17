from typing import NamedTuple

from scipy._typing import Alternative, NanPolicy, Untyped

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

class DirectionalStats:
    mean_direction: Untyped
    mean_resultant_length: Untyped
    def __init__(self, mean_direction: Untyped, mean_resultant_length: Untyped) -> None: ...

class ShapiroResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class AnsariResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class BartlettResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class LeveneResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class FlignerResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class Mean(NamedTuple):
    statistic: Untyped
    minmax: Untyped

class Variance(NamedTuple):
    statistic: Untyped
    minmax: Untyped

class Std_dev(NamedTuple):
    statistic: Untyped
    minmax: Untyped

# TODO: bunches
AndersonResult: Untyped
Anderson_ksampResult: Untyped
WilcoxonResult: Untyped
MedianTestResult: Untyped

def bayes_mvs(data: Untyped, alpha: float = 0.9) -> Untyped: ...
def mvsdist(data: Untyped) -> Untyped: ...
def kstat(
    data: Untyped,
    n: int = 2,
    *,
    axis: int | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> Untyped: ...
def kstatvar(
    data: Untyped,
    n: int = 2,
    *,
    axis: int | None = None,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> Untyped: ...
def probplot(
    x: Untyped,
    sparams: Untyped = (),
    dist: str = "norm",
    fit: bool = True,
    plot: Untyped | None = None,
    rvalue: bool = False,
) -> Untyped: ...
def ppcc_max(x: Untyped, brack: Untyped = (0.0, 1.0), dist: str = "tukeylambda") -> Untyped: ...
def ppcc_plot(
    x: Untyped,
    a: Untyped,
    b: Untyped,
    dist: str = "tukeylambda",
    plot: Untyped | None = None,
    N: int = 80,
) -> Untyped: ...
def boxcox_llf(lmb: Untyped, data: Untyped) -> Untyped: ...
def boxcox(
    x: Untyped,
    lmbda: Untyped | None = None,
    alpha: Untyped | None = None,
    optimizer: Untyped | None = None,
) -> Untyped: ...
def boxcox_normmax(
    x: Untyped,
    brack: Untyped | None = None,
    method: str = "pearsonr",
    optimizer: Untyped | None = None,
    *,
    ymax: Untyped = ...,
) -> Untyped: ...
def boxcox_normplot(x: Untyped, la: Untyped, lb: Untyped, plot: Untyped | None = None, N: int = 80) -> Untyped: ...
def yeojohnson(x: Untyped, lmbda: Untyped | None = None) -> Untyped: ...
def yeojohnson_llf(lmb: Untyped, data: Untyped) -> Untyped: ...
def yeojohnson_normmax(x: Untyped, brack: Untyped | None = None) -> Untyped: ...
def yeojohnson_normplot(x: Untyped, la: Untyped, lb: Untyped, plot: Untyped | None = None, N: int = 80) -> Untyped: ...
def shapiro(x: Untyped, *, axis: int | None = None, nan_policy: NanPolicy = "propagate", keepdims: bool = False) -> Untyped: ...
def anderson(x: Untyped, dist: str = "norm") -> Untyped: ...
def anderson_ksamp(samples: Untyped, midrank: bool = True, *, method: Untyped | None = None) -> Untyped: ...
def ansari(
    x: Untyped,
    y: Untyped,
    alternative: Alternative = "two-sided",
    *,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> Untyped: ...
def bartlett(*samples: Untyped, axis: int = 0, nan_policy: NanPolicy = "propagate", keepdims: bool = False) -> Untyped: ...
def levene(
    *samples: Untyped,
    center: str = "median",
    proportiontocut: float = 0.05,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> Untyped: ...
def fligner(
    *samples: Untyped,
    center: str = "median",
    proportiontocut: float = 0.05,
    axis: int | None = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> Untyped: ...
def mood(
    x: Untyped,
    y: Untyped,
    axis: int = 0,
    alternative: Alternative = "two-sided",
    *,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> Untyped: ...
def wilcoxon_result_unpacker(res: Untyped) -> Untyped: ...
def wilcoxon_result_object(statistic: Untyped, pvalue: Untyped, zstatistic: Untyped | None = None) -> Untyped: ...
def wilcoxon_outputs(kwds: Untyped) -> Untyped: ...
def wilcoxon(
    x: Untyped,
    y: Untyped | None = None,
    zero_method: str = "wilcox",
    correction: bool = False,
    alternative: Alternative = "two-sided",
    method: str = "auto",
    *,
    axis: int = 0,
    nan_policy: NanPolicy = "propagate",
    keepdims: bool = False,
) -> Untyped: ...
def median_test(
    *samples: Untyped,
    ties: str = "below",
    correction: bool = True,
    lambda_: int = 1,
    nan_policy: NanPolicy = "propagate",
) -> Untyped: ...
def circmean(
    samples: Untyped,
    high: Untyped = ...,
    low: int = 0,
    axis: int | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: bool = False,
) -> Untyped: ...
def circvar(
    samples: Untyped,
    high: Untyped = ...,
    low: int = 0,
    axis: int | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    keepdims: bool = False,
) -> Untyped: ...
def circstd(
    samples: Untyped,
    high: Untyped = ...,
    low: int = 0,
    axis: int | None = None,
    nan_policy: NanPolicy = "propagate",
    *,
    normalize: bool = False,
    keepdims: bool = False,
) -> Untyped: ...
def directional_stats(samples: Untyped, *, axis: int = 0, normalize: bool = True) -> Untyped: ...
def false_discovery_control(ps: Untyped, *, axis: int = 0, method: str = "bh") -> Untyped: ...
