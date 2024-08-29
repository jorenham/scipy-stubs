from typing import NamedTuple

from scipy._typing import Untyped

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

class Mean(NamedTuple):
    statistic: Untyped
    minmax: Untyped

class Variance(NamedTuple):
    statistic: Untyped
    minmax: Untyped

class Std_dev(NamedTuple):
    statistic: Untyped
    minmax: Untyped

def bayes_mvs(data, alpha: float = 0.9) -> Untyped: ...
def mvsdist(data) -> Untyped: ...
def kstat(data, n: int = 2, *, axis: Untyped | None = None) -> Untyped: ...
def kstatvar(data, n: int = 2, *, axis: Untyped | None = None) -> Untyped: ...
def probplot(
    x,
    sparams=(),
    dist: str = "norm",
    fit: bool = True,
    plot: Untyped | None = None,
    rvalue: bool = False,
) -> Untyped: ...
def ppcc_max(x, brack=(0.0, 1.0), dist: str = "tukeylambda") -> Untyped: ...
def ppcc_plot(x, a, b, dist: str = "tukeylambda", plot: Untyped | None = None, N: int = 80) -> Untyped: ...
def boxcox_llf(lmb, data) -> Untyped: ...
def boxcox(x, lmbda: Untyped | None = None, alpha: Untyped | None = None, optimizer: Untyped | None = None) -> Untyped: ...

class _BigFloat: ...

def boxcox_normmax(
    x,
    brack: Untyped | None = None,
    method: str = "pearsonr",
    optimizer: Untyped | None = None,
    *,
    ymax=...,
) -> Untyped: ...
def boxcox_normplot(x, la, lb, plot: Untyped | None = None, N: int = 80) -> Untyped: ...
def yeojohnson(x, lmbda: Untyped | None = None) -> Untyped: ...
def yeojohnson_llf(lmb, data) -> Untyped: ...
def yeojohnson_normmax(x, brack: Untyped | None = None) -> Untyped: ...
def yeojohnson_normplot(x, la, lb, plot: Untyped | None = None, N: int = 80) -> Untyped: ...

class ShapiroResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def shapiro(x) -> Untyped: ...

AndersonResult: Untyped

def anderson(x, dist: str = "norm") -> Untyped: ...

Anderson_ksampResult: Untyped

def anderson_ksamp(samples, midrank: bool = True, *, method: Untyped | None = None) -> Untyped: ...

class AnsariResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class _ABW:
    m: Untyped
    n: Untyped
    astart: Untyped
    total: Untyped
    freqs: Untyped
    def __init__(self) -> None: ...
    def pmf(self, k, n, m) -> Untyped: ...
    def cdf(self, k, n, m) -> Untyped: ...
    def sf(self, k, n, m) -> Untyped: ...

def ansari(x, y, alternative: str = "two-sided") -> Untyped: ...

class BartlettResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def bartlett(*samples, axis: int = 0) -> Untyped: ...

class LeveneResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def levene(*samples, center: str = "median", proportiontocut: float = 0.05) -> Untyped: ...

class FlignerResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def fligner(*samples, center: str = "median", proportiontocut: float = 0.05) -> Untyped: ...
def mood(x, y, axis: int = 0, alternative: str = "two-sided") -> Untyped: ...

WilcoxonResult: Untyped

def wilcoxon_result_unpacker(res) -> Untyped: ...
def wilcoxon_result_object(statistic, pvalue, zstatistic: Untyped | None = None) -> Untyped: ...
def wilcoxon_outputs(kwds) -> Untyped: ...
def wilcoxon(
    x,
    y: Untyped | None = None,
    zero_method: str = "wilcox",
    correction: bool = False,
    alternative: str = "two-sided",
    method: str = "auto",
    *,
    axis: int = 0,
) -> Untyped: ...

MedianTestResult: Untyped

def median_test(
    *samples,
    ties: str = "below",
    correction: bool = True,
    lambda_: int = 1,
    nan_policy: str = "propagate",
) -> Untyped: ...
def circmean(samples, high=..., low: int = 0, axis: Untyped | None = None, nan_policy: str = "propagate") -> Untyped: ...
def circvar(samples, high=..., low: int = 0, axis: Untyped | None = None, nan_policy: str = "propagate") -> Untyped: ...
def circstd(
    samples,
    high=...,
    low: int = 0,
    axis: Untyped | None = None,
    nan_policy: str = "propagate",
    *,
    normalize: bool = False,
) -> Untyped: ...

class DirectionalStats:
    mean_direction: Untyped
    mean_resultant_length: Untyped
    def __init__(self, mean_direction, mean_resultant_length) -> None: ...

def directional_stats(samples, *, axis: int = 0, normalize: bool = True) -> Untyped: ...
def false_discovery_control(ps, *, axis: int = 0, method: str = "bh") -> Untyped: ...
