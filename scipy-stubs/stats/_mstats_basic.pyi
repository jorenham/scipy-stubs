from collections.abc import Callable
from typing import Any, Final, Generic, Literal, NamedTuple, TypeAlias, TypedDict, overload, type_check_only
from typing_extensions import Self, TypeVar

import numpy as np
import numpy.typing as npt
from numpy._typing import (
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeNumber_co,
    _NestedSequence,
)
from optype import CanBool, CanIndex
from scipy._typing import Alternative, AnyBool, AnyComplex, AnyInt, AnyReal, NanPolicy
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
        "seasonal p-value": np.ndarray[Any, np.dtype[np.float64]],
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
    nobs: np.int_ | npt.NDArray[np.int_]
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

def argstoarray(*args: _ArrayLikeFloat_co) -> _MArrayND[np.float64]: ...
def find_repeats(arr: _ArrayLikeFloat_co) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]: ...
def count_tied_groups(x: _ArrayLikeFloat_co, use_missing: bool = False) -> dict[np.intp, np.intp]: ...
def rankdata(data: _ArrayLikeFloat_co, axis: CanIndex | None = None, use_missing: bool = False) -> npt.NDArray[np.float64]: ...
def mode(a: _ArrayLikeFloat_co, axis: CanIndex | None = 0) -> ModeResult: ...

#
@overload
def msign(x: np.ndarray[_ShapeT, np.dtype[_SCT_bifcmO]]) -> np.ndarray[_ShapeT, np.dtype[_SCT_bifcmO]]: ...
@overload
def msign(x: _ArrayLike[_SCT_bifcmO]) -> npt.NDArray[_SCT_bifcmO]: ...
@overload
def msign(x: npt.ArrayLike) -> npt.NDArray[np.number[Any] | np.timedelta64 | np.bool_ | np.object_]: ...

#
def pearsonr(x: _ArrayLikeFloat_co, y: _ArrayLikeFloat_co) -> tuple[np.float64, np.float64]: ...
def spearmanr(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co | None = None,
    use_ties: bool = True,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
    alternative: Alternative = "two-sided",
) -> SignificanceResult: ...
def kendalltau(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    use_ties: bool = True,
    use_missing: bool = False,
    method: Literal["auto", "asymptotic", "exact"] = "auto",
    alternative: Alternative = "two-sided",
) -> SignificanceResult: ...
def kendalltau_seasonal(x: _ArrayLikeFloat_co) -> _KendallTauSeasonalResult: ...
def pointbiserialr(x: _ArrayLikeFloat_co, y: _ArrayLikeFloat_co) -> PointbiserialrResult: ...
def linregress(x: _ArrayLikeFloat_co, y: _ArrayLikeFloat_co | None = None) -> LinregressResult: ...
def theilslopes(
    y: _ArrayLikeFloat_co,
    x: _ArrayLikeFloat_co | None = None,
    alpha: float = 0.95,
    method: Literal["joint", "separate"] = "separate",
) -> TheilslopesResult: ...
def siegelslopes(
    y: _ArrayLikeFloat_co,
    x: _ArrayLikeFloat_co | None = None,
    method: Literal["hierarchical", "separate"] = "hierarchical",
) -> SiegelslopesResult: ...
def sen_seasonal_slopes(x: _ArrayLikeFloat_co) -> SenSeasonalSlopesResult: ...

#
def ttest_1samp(
    a: _ArrayLikeFloat_co,
    popmean: _ArrayLikeFloat_co,
    axis: CanIndex | None = 0,
    alternative: Alternative = "two-sided",
) -> Ttest_1sampResult: ...
def ttest_ind(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axis: CanIndex | None = 0,
    equal_var: CanBool = True,
    alternative: Alternative = "two-sided",
) -> Ttest_indResult: ...
def ttest_rel(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axis: CanIndex | None = 0,
    alternative: Alternative = "two-sided",
) -> Ttest_relResult: ...
def mannwhitneyu(x: _ArrayLikeFloat_co, y: _ArrayLikeFloat_co, use_continuity: CanBool = True) -> MannwhitneyuResult: ...
def kruskal(*args: _ArrayLikeFloat_co) -> KruskalResult: ...

#
def ks_1samp(
    x: _ArrayLikeFloat_co,
    cdf: str | Callable[[float], AnyReal],
    args: tuple[object, ...] = (),
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "asymp"] = "auto",
) -> KstestResult: ...
def ks_2samp(
    data1: _ArrayLikeFloat_co,
    data2: _ArrayLikeFloat_co,
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "asymp"] = "auto",
) -> KstestResult: ...
def kstest(
    data1: _ArrayLikeFloat_co,
    data2: _ArrayLikeFloat_co | str | Callable[[float], AnyReal],
    args: tuple[object, ...] = (),
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "approx", "asymp"] = "auto",
) -> KstestResult: ...

#
@overload
def trima(
    a: _ArrayLike[_SCT_bifc],
    limits: tuple[AnyComplex, AnyComplex] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
) -> _MArrayND[_SCT_bifc]: ...
@overload
def trima(
    a: _NestedSequence[float],
    limits: tuple[AnyReal, AnyReal] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
) -> _MArrayND[np.float64 | np.int_ | np.bool_]: ...
@overload
def trima(
    a: _NestedSequence[complex],
    limits: tuple[AnyComplex, AnyComplex] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
) -> _MArrayND[np.complex128 | np.float64 | np.int_ | np.bool_]: ...

#
@overload
def trimr(
    a: _ArrayLike[_SCT_bifc],
    limits: tuple[AnyComplex, AnyComplex] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    axis: CanIndex | None = None,
) -> _MArrayND[_SCT_bifc]: ...
@overload
def trimr(
    a: _NestedSequence[float],
    limits: tuple[AnyReal, AnyReal] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    axis: CanIndex | None = None,
) -> _MArrayND[np.float64 | np.int_ | np.bool_]: ...
@overload
def trimr(
    a: _NestedSequence[complex],
    limits: tuple[AnyComplex, AnyComplex] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    axis: CanIndex | None = None,
) -> _MArrayND[np.complex128 | np.float64 | np.int_ | np.bool_]: ...

#
@overload
def trim(
    a: _ArrayLike[_SCT_bifc],
    limits: tuple[AnyComplex, AnyComplex] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    relative: CanBool = False,
    axis: CanIndex | None = None,
) -> _MArrayND[_SCT_bifc]: ...
@overload
def trim(
    a: _NestedSequence[float],
    limits: tuple[AnyReal, AnyReal] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    relative: CanBool = False,
    axis: CanIndex | None = None,
) -> _MArrayND[np.float64 | np.int_ | np.bool_]: ...
@overload
def trim(
    a: _NestedSequence[complex],
    limits: tuple[AnyComplex, AnyComplex] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    relative: CanBool = False,
    axis: CanIndex | None = None,
) -> _MArrayND[np.complex128 | np.float64 | np.int_ | np.bool_]: ...

#
@overload
def trimboth(
    data: _ArrayLike[_SCT_bifc],
    proportiontocut: float | np.floating[Any] = 0.2,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    axis: CanIndex | None = None,
) -> _MArrayND[_SCT_bifc]: ...
@overload
def trimboth(
    data: _NestedSequence[float],
    proportiontocut: float | np.floating[Any] = 0.2,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    axis: CanIndex | None = None,
) -> _MArrayND[np.float64 | np.int_ | np.bool_]: ...
@overload
def trimboth(
    data: _NestedSequence[complex],
    proportiontocut: float | np.floating[Any] = 0.2,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    axis: CanIndex | None = None,
) -> _MArrayND[np.complex128 | np.float64 | np.int_ | np.bool_]: ...

#
@overload
def trimtail(
    data: _ArrayLike[_SCT_bifc],
    proportiontocut: float | np.floating[Any] = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[CanBool, CanBool] = (True, True),
    axis: CanIndex | None = None,
) -> _MArrayND[_SCT_bifc]: ...
@overload
def trimtail(
    data: _NestedSequence[float],
    proportiontocut: float | np.floating[Any] = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[CanBool, CanBool] = (True, True),
    axis: CanIndex | None = None,
) -> _MArrayND[np.float64 | np.int_ | np.bool_]: ...
@overload
def trimtail(
    data: _NestedSequence[complex],
    proportiontocut: float | np.floating[Any] = 0.2,
    tail: Literal["left", "right"] = "left",
    inclusive: tuple[CanBool, CanBool] = (True, True),
    axis: CanIndex | None = None,
) -> _MArrayND[np.complex128 | np.float64 | np.int_ | np.bool_]: ...

#
@overload
def trimmed_mean(
    a: _ArrayLikeFloat_co,
    limits: tuple[AnyReal, AnyReal] = (0.1, 0.1),
    inclusive: tuple[CanBool, CanBool] = (1, 1),
    relative: CanBool = True,
    axis: CanIndex | None = None,
) -> _MArrayND0[np.floating[Any]]: ...
@overload
def trimmed_mean(
    a: _ArrayLikeNumber_co,
    limits: tuple[AnyComplex, AnyComplex] = (0.1, 0.1),
    inclusive: tuple[CanBool, CanBool] = (1, 1),
    relative: CanBool = True,
    axis: CanIndex | None = None,
) -> _MArrayND0[np.floating[Any] | np.complex128]: ...
def trimmed_var(
    a: _ArrayLikeNumber_co,
    limits: tuple[AnyReal, AnyReal] = (0.1, 0.1),
    inclusive: tuple[CanBool, CanBool] = (1, 1),
    relative: CanBool = True,
    axis: CanIndex | None = None,
    ddof: AnyInt = 0,
) -> _MArrayND0[np.float64]: ...
def trimmed_std(
    a: _ArrayLikeNumber_co,
    limits: tuple[AnyReal, AnyReal] = (0.1, 0.1),
    inclusive: tuple[CanBool, CanBool] = (1, 1),
    relative: CanBool = True,
    axis: CanIndex | None = None,
    ddof: AnyInt = 0,
) -> _MArrayND0[np.float64]: ...
def trimmed_stde(
    a: _ArrayLikeNumber_co,
    limits: tuple[AnyReal, AnyReal] = (0.1, 0.1),
    inclusive: tuple[CanBool, CanBool] = (1, 1),
    axis: CanIndex | None = None,
) -> _MArrayND0[np.float64]: ...
@overload
def tmean(
    a: _ArrayLikeFloat_co,
    limits: tuple[AnyReal, AnyReal] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    axis: CanIndex | None = None,
) -> _MArrayND0[np.floating[Any]]: ...
@overload
def tmean(
    a: _ArrayLikeNumber_co,
    limits: tuple[AnyComplex, AnyComplex] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    axis: CanIndex | None = None,
) -> _MArrayND0[np.inexact[Any]]: ...
def tvar(
    a: _MArrayND,
    limits: tuple[AnyReal, AnyReal] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    axis: CanIndex | None = 0,
    ddof: AnyInt = 1,
) -> _MArrayND0[np.floating[Any]]: ...
@overload
def tmin(
    a: _ArrayLike[_SCT_bifc],
    lowerlimit: AnyComplex | None = None,
    axis: CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[_SCT_bifc]: ...
@overload
def tmin(
    a: _NestedSequence[float],
    lowerlimit: AnyReal | None = None,
    axis: CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[np.float64 | np.int_ | np.bool_]: ...
@overload
def tmin(
    a: _NestedSequence[complex],
    lowerlimit: AnyComplex | None = None,
    axis: CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[np.complex128 | np.float64 | np.int_ | np.bool_]: ...
@overload
def tmax(
    a: _ArrayLike[_SCT_bifc],
    upperlimit: AnyComplex | None = None,
    axis: CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[_SCT_bifc]: ...
@overload
def tmax(
    a: _NestedSequence[float],
    upperlimit: AnyReal | None = None,
    axis: CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[np.float64 | np.int_ | np.bool_]: ...
@overload
def tmax(
    a: _NestedSequence[complex],
    upperlimit: AnyComplex | None = None,
    axis: CanIndex | None = 0,
    inclusive: AnyBool = True,
) -> _MArrayND0[np.complex128 | np.float64 | np.int_ | np.bool_]: ...
def tsem(
    a: _ArrayLikeNumber_co,
    limits: tuple[AnyReal, AnyReal] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    axis: CanIndex | None = 0,
    ddof: AnyInt = 1,
) -> _MArrayND0: ...

#
@overload
def winsorize(
    a: _ArrayLike[_SCT_f],
    limits: tuple[AnyReal, AnyReal] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    inplace: AnyBool = False,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> _MArrayND[_SCT_f]: ...
@overload
def winsorize(
    a: _ArrayLikeBool_co,
    limits: tuple[AnyReal, AnyReal] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    inplace: AnyBool = False,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> _MArrayND[np.bool_]: ...
@overload
def winsorize(
    a: _ArrayLikeInt_co,
    limits: tuple[AnyReal, AnyReal] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    inplace: AnyBool = False,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> _MArrayND[np.bool_ | np.int_]: ...
@overload
def winsorize(
    a: _ArrayLikeFloat_co,
    limits: tuple[AnyReal, AnyReal] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    inplace: AnyBool = False,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> _MArrayND[np.bool_ | np.int_ | np.floating[Any]]: ...
@overload
def winsorize(
    a: _ArrayLikeNumber_co,
    limits: tuple[AnyComplex, AnyComplex] | None = None,
    inclusive: tuple[CanBool, CanBool] = (True, True),
    inplace: AnyBool = False,
    axis: CanIndex | None = None,
    nan_policy: NanPolicy = "propagate",
) -> _MArrayND[np.bool_ | np.int_ | np.floating[Any] | np.complex128]: ...

# TODO(jorenham): Overloads for complex array-likes
def moment(a: _ArrayLikeFloat_co, moment: _ArrayLikeInt_co = 1, axis: CanIndex | None = 0) -> _MArrayND0[np.floating[Any]]: ...
def variation(a: _ArrayLikeFloat_co, axis: CanIndex | None = 0, ddof: AnyInt = 0) -> _MArrayND0[np.floating[Any]]: ...
def skew(a: _ArrayLikeFloat_co, axis: CanIndex | None = 0, bias: CanBool = True) -> _MArrayND0[np.floating[Any]]: ...
def kurtosis(
    a: _ArrayLikeFloat_co,
    axis: CanIndex | None = 0,
    fisher: CanBool = True,
    bias: CanBool = True,
) -> _MArrayND0[np.floating[Any]]: ...
def describe(
    a: _ArrayLikeFloat_co,
    axis: CanIndex | None = 0,
    ddof: AnyInt = 0,
    bias: CanBool = True,
) -> DescribeResult: ...

#
@overload
def stde_median(data: _ArrayLikeFloat_co, axis: CanIndex | None = None) -> _MArrayND0[np.floating[Any]]: ...
@overload
def stde_median(data: _ArrayLikeNumber_co, axis: CanIndex | None = None) -> _MArrayND0[np.inexact[Any]]: ...
@overload
def skewtest(
    a: _ArrayLikeFloat_co,
    axis: CanIndex | None = 0,
    alternative: Alternative = "two-sided",
) -> SkewtestResult[_MArrayND0[np.float64], _MArrayND0[np.float64]]: ...
@overload
def skewtest(
    a: _ArrayLikeNumber_co,
    axis: CanIndex | None = 0,
    alternative: Alternative = "two-sided",
) -> SkewtestResult[_MArrayND0[np.float64], _MArrayND0[np.float64 | np.complex128]]: ...
@overload
def kurtosistest(
    a: _ArrayLikeFloat_co,
    axis: CanIndex | None = 0,
    alternative: Alternative = "two-sided",
) -> KurtosistestResult[_MArrayND0[np.float64], _MArrayND0[np.float64]]: ...
@overload
def kurtosistest(
    a: _ArrayLikeNumber_co,
    axis: CanIndex | None = 0,
    alternative: Alternative = "two-sided",
) -> KurtosistestResult[_MArrayND0[np.float64], _MArrayND0[np.float64 | np.complex128]]: ...
def normaltest(a: _ArrayLikeFloat_co, axis: CanIndex | None = 0) -> NormaltestResult[_MArrayND0[np.float64]]: ...
def mquantiles(
    a: _ArrayLikeFloat_co,
    prob: _ArrayLikeFloat_co = [0.25, 0.5, 0.75],
    alphap: AnyReal = 0.4,
    betap: AnyReal = 0.4,
    axis: CanIndex | None = None,
    limit: tuple[AnyReal, AnyReal] | tuple[()] = (),
) -> _MArrayND: ...
def scoreatpercentile(
    data: _ArrayLikeFloat_co,
    per: AnyReal,
    limit: tuple[AnyReal, AnyReal] | tuple[()] = (),
    alphap: AnyReal = 0.4,
    betap: AnyReal = 0.4,
) -> _MArrayND: ...
def plotting_positions(data: _ArrayLikeFloat_co, alpha: AnyReal = 0.4, beta: AnyReal = 0.4) -> _MArrayND: ...
def obrientransform(*args: _ArrayLikeFloat_co) -> _MArrayND: ...
def sem(a: _ArrayLikeFloat_co, axis: CanIndex | None = 0, ddof: AnyInt = 1) -> np.float64 | _MArrayND: ...
def f_oneway(*args: _ArrayLikeFloat_co) -> F_onewayResult: ...
def friedmanchisquare(*args: _ArrayLikeFloat_co) -> FriedmanchisquareResult: ...
def brunnermunzel(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    alternative: Alternative = "two-sided",
    distribution: Literal["t", "normal"] = "t",
) -> BrunnerMunzelResult: ...

ttest_onesamp = ttest_1samp
kruskalwallis = kruskal
ks_twosamp = ks_2samp
trim1 = trimtail
meppf = plotting_positions
