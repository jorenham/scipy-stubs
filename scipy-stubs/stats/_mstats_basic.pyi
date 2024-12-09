from collections.abc import Callable
from typing import Any, Final, Generic, Literal, NamedTuple, TypeAlias, TypedDict, overload, type_check_only
from typing_extensions import Self, TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
import optype.typing as opt
from numpy._typing import _ArrayLike
from scipy._typing import Alternative, AnyBool, NanPolicy
from ._stats_mstats_common import SiegelslopesResult, TheilslopesResult
from ._stats_py import KstestResult, LinregressResult, SignificanceResult
from ._typing import BaseBunch

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

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

_SCT = TypeVar("_SCT", bound=np.generic, default=np.float64)
_SCT_f = TypeVar("_SCT_f", bound=np.floating[Any], default=np.float64)
_SCT_bifc = TypeVar("_SCT_bifc", bound=np.number[Any] | np.bool_, default=np.float64)
_SCT_bifcmO = TypeVar("_SCT_bifcmO", bound=np.number[Any] | np.timedelta64 | np.bool_ | np.object_)

_NDT_f_co = TypeVar("_NDT_f_co", covariant=True, bound=float | _MArrayND0[np.floating[Any]], default=_MArrayND[np.float64])
_NDT_fc_co = TypeVar(
    "_NDT_fc_co",
    covariant=True,
    bound=complex | _MArrayND0[np.inexact[Any]],
    default=_MArrayND0[np.float64 | np.complex128],
)

_MArray: TypeAlias = np.ma.MaskedArray[_ShapeT, np.dtype[_SCT]]
_MArrayND: TypeAlias = _MArray[tuple[int, ...], _SCT]
_MArrayND0: TypeAlias = _SCT | _MArray[tuple[int, ...], _SCT]

@type_check_only
class _TestResult(NamedTuple, Generic[_NDT_f_co, _NDT_fc_co]):
    statistic: _NDT_fc_co
    pvalue: _NDT_f_co

_KendallTauSeasonalResult = TypedDict(
    "_KendallTauSeasonalResult",
    {
        "seasonal tau": _MArrayND0[np.float64],
        "global tau": np.float64,
        "global tau (alt)": np.float64,
        "seasonal p-value": onp.ArrayND[np.float64],
        "global p-value (indep)": np.float64,
        "global p-value (dep)": np.float64,
        "chi2 total": _MArrayND[np.float64],
        "chi2 trend": _MArrayND[np.float64],
    },
)

###

trimdoc: Final[str] = ...

class ModeResult(NamedTuple):
    mode: _MArrayND
    count: _MArrayND  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]

class DescribeResult(NamedTuple):
    nobs: np.int_ | onp.ArrayND[np.int_]
    minmax: tuple[_MArrayND[np.floating[Any] | np.integer[Any]], _MArrayND[np.floating[Any] | np.integer[Any]]]
    mean: np.floating[Any]
    variance: np.floating[Any]
    skewness: np.floating[Any]
    kurtosis: np.floating[Any]

class PointbiserialrResult(NamedTuple):
    correlation: np.float64
    pvalue: np.float64

class Ttest_relResult(_TestResult[_NDT_f_co, _NDT_fc_co], Generic[_NDT_f_co, _NDT_fc_co]): ...
class Ttest_indResult(_TestResult[_NDT_f_co, _NDT_fc_co], Generic[_NDT_f_co, _NDT_fc_co]): ...
class Ttest_1sampResult(_TestResult[_NDT_f_co, _NDT_fc_co], Generic[_NDT_f_co, _NDT_fc_co]): ...
class SkewtestResult(_TestResult[_NDT_f_co, _NDT_fc_co], Generic[_NDT_f_co, _NDT_fc_co]): ...
class KurtosistestResult(_TestResult[_NDT_f_co, _NDT_fc_co], Generic[_NDT_f_co, _NDT_fc_co]): ...
class NormaltestResult(_TestResult[_NDT_f_co, _NDT_f_co], Generic[_NDT_f_co]): ...
class MannwhitneyuResult(_TestResult[np.float64, np.float64]): ...
class F_onewayResult(_TestResult[np.float64, np.float64]): ...
class KruskalResult(_TestResult[np.float64, np.float64]): ...
class FriedmanchisquareResult(_TestResult[np.float64, np.float64]): ...
class BrunnerMunzelResult(_TestResult[np.float64, np.float64]): ...

class SenSeasonalSlopesResult(BaseBunch[_MArrayND[np.float64], np.float64]):
    def __new__(_cls, intra_slope: float, inter_slope: float) -> Self: ...
    def __init__(self, /, intra_slope: float, inter_slope: float) -> None: ...
    @property
    def intra_slope(self, /) -> _MArrayND[np.float64]: ...
    @property
    def inter_slope(self, /) -> float: ...

# TODO(jorenham): Overloads for scalar vs. array
# TODO(jorenham): Overloads for specific dtypes

def argstoarray(*args: onp.ToFloatND) -> _MArrayND[np.float64]: ...
def find_repeats(arr: onp.ToFloatND) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.intp]]: ...
def count_tied_groups(x: onp.ToFloatND, use_missing: bool = False) -> dict[np.intp, np.intp]: ...
def rankdata(data: onp.ToFloatND, axis: op.CanIndex | None = None, use_missing: bool = False) -> onp.ArrayND[np.float64]: ...
def mode(a: onp.ToFloatND, axis: op.CanIndex | None = 0) -> ModeResult: ...

#
@overload
def msign(x: _ArrayLike[_SCT_bifcmO]) -> onp.ArrayND[_SCT_bifcmO]: ...
@overload
def msign(x: onp.ToComplexND) -> onp.ArrayND[np.number[Any] | np.timedelta64 | np.bool_ | np.object_]: ...

#
def pearsonr(x: onp.ToFloatND, y: onp.ToFloatND) -> tuple[np.float64, np.float64]: ...
def spearmanr(
    x: onp.ToFloatND,
    y: onp.ToFloatND | None = None,
    use_ties: bool = True,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    alternative: Alternative = "two-sided",
) -> SignificanceResult: ...
def kendalltau(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    use_ties: bool = True,
    use_missing: bool = False,
    method: Literal["auto", "asymptotic", "exact"] = "auto",
    alternative: Alternative = "two-sided",
) -> SignificanceResult: ...
def kendalltau_seasonal(x: onp.ToFloatND) -> _KendallTauSeasonalResult: ...
def pointbiserialr(x: onp.ToFloatND, y: onp.ToFloatND) -> PointbiserialrResult: ...
def linregress(x: onp.ToFloatND, y: onp.ToFloatND | None = None) -> LinregressResult: ...
def theilslopes(
    y: onp.ToFloatND,
    x: onp.ToFloatND | None = None,
    alpha: float = 0.95,
    method: Literal["joint", "separate"] = "separate",
) -> TheilslopesResult: ...
def siegelslopes(
    y: onp.ToFloatND,
    x: onp.ToFloatND | None = None,
    method: Literal["hierarchical", "separate"] = "hierarchical",
) -> SiegelslopesResult: ...
def sen_seasonal_slopes(x: onp.ToFloatND) -> SenSeasonalSlopesResult: ...

#
def ttest_1samp(
    a: onp.ToFloatND,
    popmean: onp.ToFloat | onp.ToFloatND,
    axis: op.CanIndex | None = 0,
    alternative: Alternative = "two-sided",
) -> Ttest_1sampResult: ...
def ttest_ind(
    a: onp.ToFloatND,
    b: onp.ToFloatND,
    axis: op.CanIndex | None = 0,
    equal_var: op.CanBool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult: ...
def ttest_rel(
    a: onp.ToFloatND,
    b: onp.ToFloatND,
    axis: op.CanIndex | None = 0,
    alternative: Alternative = "two-sided",
) -> Ttest_relResult: ...
def mannwhitneyu(x: onp.ToFloatND, y: onp.ToFloatND, use_continuity: op.CanBool = True) -> MannwhitneyuResult: ...
def kruskal(arg0: onp.ToFloatND, arg1: onp.ToFloatND, /, *args: onp.ToFloatND) -> KruskalResult: ...

#
def ks_1samp(
    x: onp.ToFloatND,
    cdf: str | Callable[[float], onp.ToFloat],
    args: tuple[object, ...] = (),
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "asymp"] = "auto",
) -> KstestResult: ...
def ks_2samp(
    data1: onp.ToFloatND,
    data2: onp.ToFloatND,
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "asymp"] = "auto",
) -> KstestResult: ...
def kstest(
    data1: onp.ToFloatND,
    data2: onp.ToFloatND | str | Callable[[float], onp.ToFloat],
    args: tuple[object, ...] = (),
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "approx", "asymp"] = "auto",
) -> KstestResult: ...

#
@overload
def trima(
    a: onp.SequenceND[bool],
    limits: tuple[onp.ToInt, onp.ToInt] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
) -> _MArrayND[np.bool_]: ...
@overload
def trima(
    a: onp.SequenceND[opt.JustInt],
    limits: tuple[onp.ToInt, onp.ToInt] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
) -> _MArrayND[np.int_]: ...
@overload
def trima(
    a: onp.SequenceND[float],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
) -> _MArrayND[np.float64 | np.int_ | np.bool_]: ...
@overload
def trima(
    a: onp.SequenceND[complex],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
) -> _MArrayND[np.complex128 | np.float64 | np.int_ | np.bool_]: ...
@overload
def trima(
    a: _ArrayLike[_SCT_bifc],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
) -> _MArrayND[_SCT_bifc]: ...

#
@overload
def trimr(
    a: onp.SequenceND[opt.JustInt | np.int_],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND[np.int_]: ...
@overload
def trimr(
    a: onp.SequenceND[float],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND[np.float64 | np.int_]: ...
@overload
def trimr(
    a: onp.SequenceND[complex],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND[np.complex128 | np.float64 | np.int_]: ...
@overload
def trimr(
    a: _ArrayLike[_SCT_bifc],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND[_SCT_bifc]: ...

#
@overload
def trim(
    a: onp.SequenceND[opt.JustInt | np.int_],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    relative: op.CanBool = False,
    axis: op.CanIndex | None = None,
) -> _MArrayND[np.int_]: ...
@overload
def trim(
    a: onp.SequenceND[float],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    relative: op.CanBool = False,
    axis: op.CanIndex | None = None,
) -> _MArrayND[np.float64 | np.int_]: ...
@overload
def trim(
    a: onp.SequenceND[complex],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    relative: op.CanBool = False,
    axis: op.CanIndex | None = None,
) -> _MArrayND[np.complex128 | np.float64 | np.int_]: ...
@overload
def trim(
    a: _ArrayLike[_SCT_bifc],
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    relative: op.CanBool = False,
    axis: op.CanIndex | None = None,
) -> _MArrayND[_SCT_bifc]: ...

#
@overload
def trimboth(
    data: onp.SequenceND[opt.JustInt | np.int_],
    proportiontocut: float | np.floating[Any] = 0.2,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND[np.int_]: ...
@overload
def trimboth(
    data: onp.SequenceND[float],
    proportiontocut: float | np.floating[Any] = 0.2,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND[np.float64 | np.int_]: ...
@overload
def trimboth(
    data: onp.SequenceND[complex],
    proportiontocut: float | np.floating[Any] = 0.2,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND[np.complex128 | np.float64 | np.int_]: ...
@overload
def trimboth(
    data: _ArrayLike[_SCT_bifc],
    proportiontocut: float | np.floating[Any] = 0.2,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND[_SCT_bifc]: ...

#
@overload
def trimtail(
    data: onp.SequenceND[opt.JustInt | np.int_],
    proportiontocut: float | np.floating[Any] = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND[np.int_]: ...
@overload
def trimtail(
    data: onp.SequenceND[float],
    proportiontocut: float | np.floating[Any] = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND[np.float64 | np.int_]: ...
@overload
def trimtail(
    data: onp.SequenceND[complex],
    proportiontocut: float | np.floating[Any] = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND[np.complex128 | np.float64 | np.int_]: ...
@overload
def trimtail(
    data: _ArrayLike[_SCT_bifc],
    proportiontocut: float | np.floating[Any] = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND[_SCT_bifc]: ...

#
@overload
def trimmed_mean(
    a: onp.ToFloatND,
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: op.CanBool = True,
    axis: op.CanIndex | None = None,
) -> _MArrayND0[np.floating[Any]]: ...
@overload
def trimmed_mean(
    a: onp.ToComplexND,
    limits: tuple[onp.ToComplex, onp.ToComplex] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: op.CanBool = True,
    axis: op.CanIndex | None = None,
) -> _MArrayND0[np.floating[Any] | np.complex128]: ...

#
def trimmed_var(
    a: onp.ToComplexND,
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: op.CanBool = True,
    axis: op.CanIndex | None = None,
    ddof: onp.ToInt = 0,
) -> _MArrayND0[np.float64]: ...

#
def trimmed_std(
    a: onp.ToComplexND,
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    relative: op.CanBool = True,
    axis: op.CanIndex | None = None,
    ddof: onp.ToInt = 0,
) -> _MArrayND0[np.float64]: ...

#
def trimmed_stde(
    a: onp.ToComplexND,
    limits: tuple[onp.ToFloat, onp.ToFloat] = (0.1, 0.1),
    inclusive: tuple[op.CanBool, op.CanBool] = (1, 1),
    axis: op.CanIndex | None = None,
) -> _MArrayND0[np.float64]: ...

#
@overload
def tmean(
    a: onp.ToFloatND,
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND0[np.floating[Any]]: ...
@overload
def tmean(
    a: onp.ToComplexND,
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = None,
) -> _MArrayND0[np.inexact[Any]]: ...

#
def tvar(
    a: _MArrayND,
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = 0,
    ddof: onp.ToInt = 1,
) -> _MArrayND0[np.floating[Any]]: ...

#
@overload
def tmin(
    a: onp.SequenceND[opt.JustInt | np.int_],
    lowerlimit: onp.ToFloat | None = None,
    axis: op.CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[np.int_]: ...
@overload
def tmin(
    a: onp.SequenceND[float],
    lowerlimit: onp.ToFloat | None = None,
    axis: op.CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[np.float64 | np.int_]: ...
@overload
def tmin(
    a: onp.SequenceND[complex],
    lowerlimit: onp.ToComplex | None = None,
    axis: op.CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[np.complex128 | np.float64 | np.int_]: ...
@overload
def tmin(
    a: _ArrayLike[_SCT_bifc],
    lowerlimit: onp.ToComplex | None = None,
    axis: op.CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[_SCT_bifc]: ...

#
@overload
def tmax(
    a: onp.SequenceND[opt.JustInt | np.int_],
    upperlimit: onp.ToFloat | None = None,
    axis: op.CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[np.int_]: ...
@overload
def tmax(
    a: onp.SequenceND[float],
    upperlimit: onp.ToFloat | None = None,
    axis: op.CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[np.float64 | np.int_]: ...
@overload
def tmax(
    a: onp.SequenceND[complex],
    upperlimit: onp.ToComplex | None = None,
    axis: op.CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[np.complex128 | np.float64 | np.int_]: ...
@overload
def tmax(
    a: _ArrayLike[_SCT_bifc],
    upperlimit: onp.ToComplex | None = None,
    axis: op.CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[_SCT_bifc]: ...

#
def tsem(
    a: onp.ToComplexND,
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    axis: op.CanIndex | None = 0,
    ddof: onp.ToInt = 1,
) -> _MArrayND0: ...

#
@overload
def winsorize(
    a: onp.ToIntND,
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    inplace: AnyBool = False,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> _MArrayND[np.int_]: ...
@overload
def winsorize(
    a: _ArrayLike[_SCT_f],
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    inplace: AnyBool = False,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> _MArrayND[_SCT_f]: ...
@overload
def winsorize(
    a: onp.ToFloatND,
    limits: tuple[onp.ToFloat, onp.ToFloat] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    inplace: AnyBool = False,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> _MArrayND[np.floating[Any] | np.int_]: ...
@overload
def winsorize(
    a: onp.ToComplexND,
    limits: tuple[onp.ToComplex, onp.ToComplex] | None = None,
    inclusive: tuple[op.CanBool, op.CanBool] = (True, True),
    inplace: AnyBool = False,
    axis: op.CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> _MArrayND[np.complex128 | np.floating[Any] | np.int_]: ...

# TODO(jorenham): Overloads for complex array-likes
def moment(
    a: onp.ToFloatND,
    moment: onp.ToInt | onp.ToIntND = 1,
    axis: op.CanIndex | None = 0,
) -> _MArrayND0[np.floating[Any]]: ...
def variation(a: onp.ToFloatND, axis: op.CanIndex | None = 0, ddof: onp.ToInt = 0) -> _MArrayND0[np.floating[Any]]: ...
def skew(a: onp.ToFloatND, axis: op.CanIndex | None = 0, bias: op.CanBool = True) -> _MArrayND0[np.floating[Any]]: ...
def kurtosis(
    a: onp.ToFloatND,
    axis: op.CanIndex | None = 0,
    fisher: op.CanBool = True,
    bias: op.CanBool = True,
) -> _MArrayND0[np.floating[Any]]: ...
def describe(
    a: onp.ToFloatND,
    axis: op.CanIndex | None = 0,
    ddof: onp.ToInt = 0,
    bias: op.CanBool = True,
) -> DescribeResult: ...

#
@overload
def stde_median(data: onp.ToFloatND, axis: op.CanIndex | None = None) -> _MArrayND0[np.floating[Any]]: ...
@overload
def stde_median(data: onp.ToComplexND, axis: op.CanIndex | None = None) -> _MArrayND0[np.inexact[Any]]: ...
@overload
def skewtest(
    a: onp.ToFloatND,
    axis: op.CanIndex | None = 0,
    alternative: Alternative = "two-sided",
) -> SkewtestResult[_MArrayND0[np.float64], _MArrayND0[np.float64]]: ...
@overload
def skewtest(
    a: onp.ToComplexND,
    axis: op.CanIndex | None = 0,
    alternative: Alternative = "two-sided",
) -> SkewtestResult[_MArrayND0[np.float64], _MArrayND0[np.float64 | np.complex128]]: ...
@overload
def kurtosistest(
    a: onp.ToFloatND,
    axis: op.CanIndex | None = 0,
    alternative: Alternative = "two-sided",
) -> KurtosistestResult[_MArrayND0[np.float64], _MArrayND0[np.float64]]: ...
@overload
def kurtosistest(
    a: onp.ToComplexND,
    axis: op.CanIndex | None = 0,
    alternative: Alternative = "two-sided",
) -> KurtosistestResult[_MArrayND0[np.float64], _MArrayND0[np.float64 | np.complex128]]: ...
def normaltest(a: onp.ToFloatND, axis: op.CanIndex | None = 0) -> NormaltestResult[_MArrayND0[np.float64]]: ...
def mquantiles(
    a: onp.ToFloatND,
    prob: onp.ToFloatND = [0.25, 0.5, 0.75],
    alphap: onp.ToFloat = 0.4,
    betap: onp.ToFloat = 0.4,
    axis: op.CanIndex | None = None,
    limit: tuple[onp.ToFloat, onp.ToFloat] | tuple[()] = (),
) -> _MArrayND: ...
def scoreatpercentile(
    data: onp.ToFloatND,
    per: onp.ToFloat,
    limit: tuple[onp.ToFloat, onp.ToFloat] | tuple[()] = (),
    alphap: onp.ToFloat = 0.4,
    betap: onp.ToFloat = 0.4,
) -> _MArrayND: ...
def plotting_positions(data: onp.ToFloatND, alpha: onp.ToFloat = 0.4, beta: onp.ToFloat = 0.4) -> _MArrayND: ...
def obrientransform(arg0: onp.ToFloatND, /, *args: onp.ToFloatND) -> _MArrayND: ...
def sem(a: onp.ToFloatND, axis: op.CanIndex | None = 0, ddof: onp.ToInt = 1) -> np.float64 | _MArrayND: ...
def f_oneway(arg0: onp.ToFloatND, arg1: onp.ToFloatND, /, *args: onp.ToFloatND) -> F_onewayResult: ...
def friedmanchisquare(arg0: onp.ToFloatND, *args: onp.ToFloatND) -> FriedmanchisquareResult: ...
def brunnermunzel(
    x: onp.ToFloatND,
    y: onp.ToFloatND,
    alternative: Alternative = "two-sided",
    distribution: Literal["t", "normal"] = "t",
) -> BrunnerMunzelResult: ...

ttest_onesamp = ttest_1samp
kruskalwallis = kruskal
ks_twosamp = ks_2samp
trim1 = trimtail
meppf = plotting_positions
