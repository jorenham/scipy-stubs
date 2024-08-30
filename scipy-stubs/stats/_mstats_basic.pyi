from typing import NamedTuple

from scipy._typing import Untyped
from . import distributions as distributions

def argstoarray(*args) -> Untyped: ...
def find_repeats(arr) -> Untyped: ...
def count_tied_groups(x, use_missing: bool = False) -> Untyped: ...
def rankdata(data, axis: Untyped | None = None, use_missing: bool = False) -> Untyped: ...

class ModeResult(NamedTuple):
    mode: Untyped
    count: Untyped  # type: ignore[assignment]

def mode(a, axis: int = 0) -> Untyped: ...
def msign(x) -> Untyped: ...
def pearsonr(x, y) -> Untyped: ...
def spearmanr(
    x,
    y: Untyped | None = None,
    use_ties: bool = True,
    axis: Untyped | None = None,
    nan_policy: str = "propagate",
    alternative: str = "two-sided",
) -> Untyped: ...
def kendalltau(
    x,
    y,
    use_ties: bool = True,
    use_missing: bool = False,
    method: str = "auto",
    alternative: str = "two-sided",
) -> Untyped: ...
def kendalltau_seasonal(x) -> Untyped: ...

class PointbiserialrResult(NamedTuple):
    correlation: Untyped
    pvalue: Untyped

def pointbiserialr(x, y) -> Untyped: ...
def linregress(x, y: Untyped | None = None) -> Untyped: ...
def theilslopes(y, x: Untyped | None = None, alpha: float = 0.95, method: str = "separate") -> Untyped: ...
def siegelslopes(y, x: Untyped | None = None, method: str = "hierarchical") -> Untyped: ...

SenSeasonalSlopesResult: Untyped

def sen_seasonal_slopes(x) -> Untyped: ...

class Ttest_1sampResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def ttest_1samp(a, popmean, axis: int = 0, alternative: str = "two-sided") -> Untyped: ...

ttest_onesamp = ttest_1samp

class Ttest_indResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def ttest_ind(a, b, axis: int = 0, equal_var: bool = True, alternative: str = "two-sided") -> Untyped: ...

class Ttest_relResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def ttest_rel(a, b, axis: int = 0, alternative: str = "two-sided") -> Untyped: ...

class MannwhitneyuResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def mannwhitneyu(x, y, use_continuity: bool = True) -> Untyped: ...

class KruskalResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def kruskal(*args) -> Untyped: ...

kruskalwallis = kruskal

def ks_1samp(x, cdf, args=(), alternative: str = "two-sided", method: str = "auto") -> Untyped: ...
def ks_2samp(data1, data2, alternative: str = "two-sided", method: str = "auto") -> Untyped: ...

ks_twosamp = ks_2samp

def kstest(data1, data2, args=(), alternative: str = "two-sided", method: str = "auto") -> Untyped: ...
def trima(a, limits: Untyped | None = None, inclusive=(True, True)) -> Untyped: ...
def trimr(a, limits: Untyped | None = None, inclusive=(True, True), axis: Untyped | None = None) -> Untyped: ...

trimdoc: str

def trim(
    a,
    limits: Untyped | None = None,
    inclusive=(True, True),
    relative: bool = False,
    axis: Untyped | None = None,
) -> Untyped: ...
def trimboth(data, proportiontocut: float = 0.2, inclusive=(True, True), axis: Untyped | None = None) -> Untyped: ...
def trimtail(
    data,
    proportiontocut: float = 0.2,
    tail: str = "left",
    inclusive=(True, True),
    axis: Untyped | None = None,
) -> Untyped: ...

trim1 = trimtail

def trimmed_mean(a, limits=(0.1, 0.1), inclusive=(1, 1), relative: bool = True, axis: Untyped | None = None) -> Untyped: ...
def trimmed_var(
    a,
    limits=(0.1, 0.1),
    inclusive=(1, 1),
    relative: bool = True,
    axis: Untyped | None = None,
    ddof: int = 0,
) -> Untyped: ...
def trimmed_std(
    a,
    limits=(0.1, 0.1),
    inclusive=(1, 1),
    relative: bool = True,
    axis: Untyped | None = None,
    ddof: int = 0,
) -> Untyped: ...
def trimmed_stde(a, limits=(0.1, 0.1), inclusive=(1, 1), axis: Untyped | None = None) -> Untyped: ...
def tmean(a, limits: Untyped | None = None, inclusive=(True, True), axis: Untyped | None = None) -> Untyped: ...
def tvar(a, limits: Untyped | None = None, inclusive=(True, True), axis: int = 0, ddof: int = 1) -> Untyped: ...
def tmin(a, lowerlimit: Untyped | None = None, axis: int = 0, inclusive: bool = True) -> Untyped: ...
def tmax(a, upperlimit: Untyped | None = None, axis: int = 0, inclusive: bool = True) -> Untyped: ...
def tsem(a, limits: Untyped | None = None, inclusive=(True, True), axis: int = 0, ddof: int = 1) -> Untyped: ...
def winsorize(
    a,
    limits: Untyped | None = None,
    inclusive=(True, True),
    inplace: bool = False,
    axis: Untyped | None = None,
    nan_policy: str = "propagate",
) -> Untyped: ...
def moment(a, moment: int = 1, axis: int = 0) -> Untyped: ...
def variation(a, axis: int = 0, ddof: int = 0) -> Untyped: ...
def skew(a, axis: int = 0, bias: bool = True) -> Untyped: ...
def kurtosis(a, axis: int = 0, fisher: bool = True, bias: bool = True) -> Untyped: ...

class DescribeResult(NamedTuple):
    nobs: Untyped
    minmax: Untyped
    mean: Untyped
    variance: Untyped
    skewness: Untyped
    kurtosis: Untyped

def describe(a, axis: int = 0, ddof: int = 0, bias: bool = True) -> Untyped: ...
def stde_median(data, axis: Untyped | None = None) -> Untyped: ...

class SkewtestResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def skewtest(a, axis: int = 0, alternative: str = "two-sided") -> Untyped: ...

class KurtosistestResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def kurtosistest(a, axis: int = 0, alternative: str = "two-sided") -> Untyped: ...

class NormaltestResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def normaltest(a, axis: int = 0) -> Untyped: ...
def mquantiles(
    a,
    prob=(0.25, 0.5, 0.75),
    alphap: float = 0.4,
    betap: float = 0.4,
    axis: Untyped | None = None,
    limit=(),
) -> Untyped: ...
def scoreatpercentile(data, per, limit=(), alphap: float = 0.4, betap: float = 0.4) -> Untyped: ...
def plotting_positions(data, alpha: float = 0.4, beta: float = 0.4) -> Untyped: ...

meppf = plotting_positions

def obrientransform(*args) -> Untyped: ...
def sem(a, axis: int = 0, ddof: int = 1) -> Untyped: ...

class F_onewayResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def f_oneway(*args) -> Untyped: ...

class FriedmanchisquareResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def friedmanchisquare(*args) -> Untyped: ...

class BrunnerMunzelResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

def brunnermunzel(x, y, alternative: str = "two-sided", distribution: str = "t") -> Untyped: ...
