from typing import NamedTuple

from scipy._typing import AnyBool, Untyped

__all__ = [
    "argstoarray",
    "brunnermunzel",
    "count_tied_groups",
    "describe",
    "f_oneway",
    "find_repeats",
    "friedmanchisquare",
    "kendalltau",
    "kendalltau_seasonal",
    "kruskal",
    "kruskalwallis",
    "ks_1samp",
    "ks_2samp",
    "ks_twosamp",
    "kstest",
    "kurtosis",
    "kurtosistest",
    "linregress",
    "mannwhitneyu",
    "meppf",
    "mode",
    "moment",
    "mquantiles",
    "msign",
    "normaltest",
    "obrientransform",
    "pearsonr",
    "plotting_positions",
    "pointbiserialr",
    "rankdata",
    "scoreatpercentile",
    "sem",
    "sen_seasonal_slopes",
    "siegelslopes",
    "skew",
    "skewtest",
    "spearmanr",
    "theilslopes",
    "tmax",
    "tmean",
    "tmin",
    "trim",
    "trima",
    "trimboth",
    "trimmed_mean",
    "trimmed_std",
    "trimmed_stde",
    "trimmed_var",
    "trimr",
    "trimtail",
    "tsem",
    "ttest_1samp",
    "ttest_ind",
    "ttest_onesamp",
    "ttest_rel",
    "tvar",
    "variation",
    "winsorize",
]

class ModeResult(NamedTuple):
    mode: Untyped
    count: Untyped

class PointbiserialrResult(NamedTuple):
    correlation: Untyped
    pvalue: Untyped

SenSeasonalSlopesResult: Untyped

class Ttest_relResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class Ttest_indResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class Ttest_1sampResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class MannwhitneyuResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class KruskalResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class DescribeResult(NamedTuple):
    nobs: Untyped
    minmax: Untyped
    mean: Untyped
    variance: Untyped
    skewness: Untyped
    kurtosis: Untyped

class KurtosistestResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class SkewtestResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class NormaltestResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class F_onewayResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class FriedmanchisquareResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

class BrunnerMunzelResult(NamedTuple):
    statistic: Untyped
    pvalue: Untyped

trimdoc: str

def argstoarray(*args: Untyped) -> Untyped: ...
def find_repeats(arr: Untyped) -> Untyped: ...
def count_tied_groups(x: Untyped, use_missing: bool = False) -> Untyped: ...
def rankdata(data: Untyped, axis: int | None = None, use_missing: bool = False) -> Untyped: ...
def mode(a: Untyped, axis: int = 0) -> Untyped: ...
def msign(x: Untyped) -> Untyped: ...
def pearsonr(x: Untyped, y: Untyped) -> Untyped: ...
def spearmanr(
    x: Untyped,
    y: Untyped | None = None,
    use_ties: bool = True,
    axis: int | None = None,
    nan_policy: str = "propagate",
    alternative: str = "two-sided",
) -> Untyped: ...
def kendalltau(
    x: Untyped,
    y: Untyped,
    use_ties: bool = True,
    use_missing: bool = False,
    method: str = "auto",
    alternative: str = "two-sided",
) -> Untyped: ...
def kendalltau_seasonal(x: Untyped) -> Untyped: ...
def pointbiserialr(x: Untyped, y: Untyped) -> Untyped: ...
def linregress(x: Untyped, y: Untyped | None = None) -> Untyped: ...
def theilslopes(y: Untyped, x: Untyped | None = None, alpha: float = 0.95, method: str = "separate") -> Untyped: ...
def siegelslopes(y: Untyped, x: Untyped | None = None, method: str = "hierarchical") -> Untyped: ...
def sen_seasonal_slopes(x: Untyped) -> Untyped: ...
def ttest_1samp(a: Untyped, popmean: Untyped, axis: int = 0, alternative: str = "two-sided") -> Untyped: ...
def ttest_ind(a: Untyped, b: Untyped, axis: int = 0, equal_var: bool = True, alternative: str = "two-sided") -> Untyped: ...
def ttest_rel(a: Untyped, b: Untyped, axis: int = 0, alternative: str = "two-sided") -> Untyped: ...
def mannwhitneyu(x: Untyped, y: Untyped, use_continuity: bool = True) -> Untyped: ...
def kruskal(*args: Untyped) -> Untyped: ...
def ks_1samp(x: Untyped, cdf: Untyped, args: Untyped = (), alternative: str = "two-sided", method: str = "auto") -> Untyped: ...
def ks_2samp(data1: Untyped, data2: Untyped, alternative: str = "two-sided", method: str = "auto") -> Untyped: ...
def kstest(
    data1: Untyped,
    data2: Untyped,
    args: Untyped = (),
    alternative: str = "two-sided",
    method: str = "auto",
) -> Untyped: ...
def trima(a: Untyped, limits: Untyped | None = None, inclusive: tuple[AnyBool, AnyBool] = (True, True)) -> Untyped: ...
def trimr(
    a: Untyped,
    limits: Untyped | None = None,
    inclusive: tuple[AnyBool, AnyBool] = (True, True),
    axis: int | None = None,
) -> Untyped: ...
def trim(
    a: Untyped,
    limits: Untyped | None = None,
    inclusive: tuple[AnyBool, AnyBool] = (True, True),
    relative: bool = False,
    axis: int | None = None,
) -> Untyped: ...
def trimboth(
    data: Untyped,
    proportiontocut: float = 0.2,
    inclusive: tuple[AnyBool, AnyBool] = (True, True),
    axis: int | None = None,
) -> Untyped: ...
def trimtail(
    data: Untyped,
    proportiontocut: float = 0.2,
    tail: str = "left",
    inclusive: tuple[AnyBool, AnyBool] = (True, True),
    axis: int | None = None,
) -> Untyped: ...
def trimmed_mean(
    a: Untyped,
    limits: Untyped = (0.1, 0.1),
    inclusive: tuple[AnyBool, AnyBool] = (1, 1),
    relative: bool = True,
    axis: int | None = None,
) -> Untyped: ...
def trimmed_var(
    a: Untyped,
    limits: Untyped = (0.1, 0.1),
    inclusive: tuple[AnyBool, AnyBool] = (1, 1),
    relative: bool = True,
    axis: int | None = None,
    ddof: int = 0,
) -> Untyped: ...
def trimmed_std(
    a: Untyped,
    limits: Untyped = (0.1, 0.1),
    inclusive: tuple[AnyBool, AnyBool] = (1, 1),
    relative: bool = True,
    axis: int | None = None,
    ddof: int = 0,
) -> Untyped: ...
def trimmed_stde(
    a: Untyped,
    limits: Untyped = (0.1, 0.1),
    inclusive: tuple[AnyBool, AnyBool] = (1, 1),
    axis: int | None = None,
) -> Untyped: ...
def tmean(
    a: Untyped,
    limits: Untyped | None = None,
    inclusive: tuple[AnyBool, AnyBool] = (True, True),
    axis: int | None = None,
) -> Untyped: ...
def tvar(
    a: Untyped,
    limits: Untyped | None = None,
    inclusive: tuple[AnyBool, AnyBool] = (True, True),
    axis: int = 0,
    ddof: int = 1,
) -> Untyped: ...
def tmin(a: Untyped, lowerlimit: Untyped | None = None, axis: int = 0, inclusive: bool = True) -> Untyped: ...
def tmax(a: Untyped, upperlimit: Untyped | None = None, axis: int = 0, inclusive: bool = True) -> Untyped: ...
def tsem(
    a: Untyped,
    limits: Untyped | None = None,
    inclusive: tuple[AnyBool, AnyBool] = (True, True),
    axis: int = 0,
    ddof: int = 1,
) -> Untyped: ...
def winsorize(
    a: Untyped,
    limits: Untyped | None = None,
    inclusive: tuple[AnyBool, AnyBool] = (True, True),
    inplace: bool = False,
    axis: int | None = None,
    nan_policy: str = "propagate",
) -> Untyped: ...
def moment(a: Untyped, moment: int = 1, axis: int = 0) -> Untyped: ...
def variation(a: Untyped, axis: int = 0, ddof: int = 0) -> Untyped: ...
def skew(a: Untyped, axis: int = 0, bias: bool = True) -> Untyped: ...
def kurtosis(a: Untyped, axis: int = 0, fisher: bool = True, bias: bool = True) -> Untyped: ...
def describe(a: Untyped, axis: int = 0, ddof: int = 0, bias: bool = True) -> Untyped: ...
def stde_median(data: Untyped, axis: int | None = None) -> Untyped: ...
def skewtest(a: Untyped, axis: int = 0, alternative: str = "two-sided") -> Untyped: ...
def kurtosistest(a: Untyped, axis: int = 0, alternative: str = "two-sided") -> Untyped: ...
def normaltest(a: Untyped, axis: int = 0) -> Untyped: ...
def mquantiles(
    a: Untyped,
    prob: Untyped = [0.25, 0.5, 0.75],
    alphap: float = 0.4,
    betap: float = 0.4,
    axis: int | None = None,
    limit: Untyped = (),
) -> Untyped: ...
def scoreatpercentile(data: Untyped, per: Untyped, limit: Untyped = (), alphap: float = 0.4, betap: float = 0.4) -> Untyped: ...
def plotting_positions(data: Untyped, alpha: float = 0.4, beta: float = 0.4) -> Untyped: ...
def obrientransform(*args: Untyped) -> Untyped: ...
def sem(a: Untyped, axis: int = 0, ddof: int = 1) -> Untyped: ...
def f_oneway(*args: Untyped) -> Untyped: ...
def friedmanchisquare(*args: Untyped) -> Untyped: ...
def brunnermunzel(x: Untyped, y: Untyped, alternative: str = "two-sided", distribution: str = "t") -> Untyped: ...

ttest_onesamp = ttest_1samp
kruskalwallis = kruskal
ks_twosamp = ks_2samp
trim1 = trimtail
meppf = plotting_positions
