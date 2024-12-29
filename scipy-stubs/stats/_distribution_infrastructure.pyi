# mypy: disable-error-code="explicit-override"
# pyright: reportUnannotatedClassAttribute=false

import abc
from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
from typing import Any, ClassVar, Final, Generic, Literal as L, Protocol, TypeAlias, TypedDict, overload, type_check_only
from typing_extensions import LiteralString, Never, Self, TypeIs, TypeVar, Unpack, override

import numpy as np
import optype.numpy as onp
import optype.typing as opt
from scipy._typing import ToRNG
from ._distn_infrastructure import rv_continuous
from ._probability_distribution import _BaseDistribution

__all__ = ["Mixture", "abs", "exp", "log", "make_distribution", "order_statistic", "truncate"]

###

_FloatT = TypeVar("_FloatT", bound=_Float, default=_Float)
_FloatT_co = TypeVar("_FloatT_co", bound=_Float, default=_Float, covariant=True)

_RealT = TypeVar("_RealT", bound=_Float | _Int, default=_Float | _Int)
_RealT_co = TypeVar("_RealT_co", bound=_Float | _Int, default=_Float | _Int, covariant=True)

_ShapeT1 = TypeVar("_ShapeT1", bound=onp.AtLeast1D, default=onp.AtLeast1D)
_ShapeT = TypeVar("_ShapeT", bound=_ND, default=_ND)
_ShapeT_co = TypeVar("_ShapeT_co", bound=_ND, default=_ND, covariant=True)

_DistT0 = TypeVar("_DistT0", bound=_CDist0)
_DistT1 = TypeVar("_DistT1", bound=_CDist[_1D])
_DistT_1 = TypeVar("_DistT_1", bound=_CDist[onp.AtMost1D])
_DistT2 = TypeVar("_DistT2", bound=_CDist[_2D])
_DistT_2 = TypeVar("_DistT_2", bound=_CDist[onp.AtMost2D])
_DistT3 = TypeVar("_DistT3", bound=_CDist[_3D])
_DistT_3 = TypeVar("_DistT_3", bound=_CDist[onp.AtMost3D])
_DistT = TypeVar("_DistT", bound=ContinuousDistribution)
_DistT_co = TypeVar("_DistT_co", bound=ContinuousDistribution, default=ContinuousDistribution, covariant=True)

_AxesT = TypeVar("_AxesT", bound=_Axes, default=Any)

###

_Int: TypeAlias = np.integer[Any]
_Float: TypeAlias = np.floating[Any]
_OutFloat: TypeAlias = np.float64 | np.longdouble

_NT = TypeVar("_NT", default=int)
_0D: TypeAlias = tuple[()]  # noqa: PYI042
_1D: TypeAlias = tuple[_NT]  # noqa: PYI042
_2D: TypeAlias = tuple[_NT, _NT]  # noqa: PYI042
_3D: TypeAlias = tuple[_NT, _NT, _NT]  # noqa: PYI042
_ND: TypeAlias = tuple[_NT, ...]

_ToFloatMax1D: TypeAlias = onp.ToFloatStrict1D | onp.ToFloat
_ToFloatMax2D: TypeAlias = onp.ToFloatStrict2D | _ToFloatMax1D
_ToFloatMax3D: TypeAlias = onp.ToFloatStrict3D | _ToFloatMax2D
_ToFloatMaxND: TypeAlias = onp.ToFloatND | onp.ToFloat

_ToJustIntMax1D: TypeAlias = onp.ToJustIntStrict1D | onp.ToJustInt
_ToJustIntMax2D: TypeAlias = onp.ToJustIntStrict2D | _ToJustIntMax1D
_ToJustIntMax3D: TypeAlias = onp.ToJustIntStrict3D | _ToJustIntMax2D
_ToJustIntMaxND: TypeAlias = onp.ToJustIntND | onp.ToJustInt

_Null: TypeAlias = opt.Just[object]  # type of `_null`
_Axes: TypeAlias = object  # placeholder for `matplotlib.axes.Axes`

_DomainRegion: TypeAlias = L["domain", "typical"]
_DomainDrawType: TypeAlias = L["in", "out", "on", "nan"]
_ValidationPolicy: TypeAlias = L["skip_all"] | None
_CachePolicy: TypeAlias = L["no_cache"] | None
_PlotQuantity: TypeAlias = L["x", "pdf", "cdf", "ccdf", "icdf", "iccdf", "logpdf", "logcdf", "logccdf", "ilogcdf", "ilogccdf"]
_SMomentMethod: TypeAlias = L["formula", "general", "transform", "normalize", "cache"]

_ParamValues: TypeAlias = Mapping[str, _ToFloatMaxND]
_ToDomain: TypeAlias = tuple[onp.ToFloat | str, onp.ToFloat | str]
_ToTol: TypeAlias = opt.JustFloat | _Null
_DrawProportions: TypeAlias = tuple[onp.ToFloat, onp.ToFloat, onp.ToFloat, onp.ToFloat]
_Elementwise: TypeAlias = Callable[[onp.ArrayND[np.float64]], onp.ArrayND[_FloatT]]

_CDist: TypeAlias = ContinuousDistribution[_Float, _ShapeT]
_CDist0: TypeAlias = ContinuousDistribution[_FloatT, _0D]
_TransDist: TypeAlias = TransformedDistribution[_DistT, _FloatT, _ShapeT]
_LinDist: TypeAlias = ShiftedScaledDistribution[_DistT, _FloatT, _ShapeT]
_FoldDist: TypeAlias = FoldedDistribution[_DistT, _FloatT, _ShapeT]
_TruncDist: TypeAlias = TruncatedDistribution[_DistT, _ShapeT]

@type_check_only
class _ParamField(Protocol[_FloatT_co, _ShapeT_co]):
    # This actually works (even on mypy)!
    @overload
    def __get__(self: _ParamField[_FloatT, _0D], obj: object, tp: type | None = None, /) -> _FloatT: ...
    @overload
    def __get__(self: _ParamField[_FloatT, _ShapeT1], obj: object, tp: type | None = None, /) -> onp.Array[_ShapeT1, _FloatT]: ...

@type_check_only
class _DistOpts(TypedDict, total=False):
    tol: _ToTol
    validation_policy: _ValidationPolicy
    cache_policy: _CachePolicy

###

_null: Final[_Null] = ...

def _isnull(x: object) -> TypeIs[_Null | None]: ...

# TODO(jorenham): Generic dtype and shape
class _Domain(abc.ABC):
    # NOTE: This is a `ClassVar[dict[str, float]]` that's overridden as instance attribute in `_SimpleDomain`.
    # https://github.com/scipy/scipy/pull/22139
    symbols: Mapping[float, LiteralString] = ...

    @abc.abstractmethod
    @override
    def __str__(self, /) -> str: ...
    @abc.abstractmethod
    def contains(self, /, x: onp.ArrayND[Any]) -> onp.ArrayND[np.bool_]: ...
    @abc.abstractmethod
    def draw(self, /, n: int) -> onp.ArrayND[_FloatT]: ...
    @abc.abstractmethod
    def get_numerical_endpoints(self, /, x: _ParamValues) -> tuple[onp.ArrayND[_OutFloat], onp.ArrayND[_OutFloat]]: ...

# TODO(jorenham): Generic dtype
class _SimpleDomain(_Domain, metaclass=abc.ABCMeta):
    def __init__(self, /, endpoints: _ToDomain = ..., inclusive: tuple[bool, bool] = (False, False)) -> None: ...
    @override
    def __str__(self, /) -> str: ...  # noqa: PYI029
    @override
    def get_numerical_endpoints(self, /, parameter_values: _ParamValues) -> _2D[onp.ArrayND[_OutFloat]]: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def contains(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        item: onp.ArrayND[_Int | _Float],
        parameter_values: _ParamValues | None = None,
    ) -> onp.ArrayND[np.bool_]: ...

    #
    def define_parameters(self, /, *parameters: _Parameter) -> None: ...

class _RealDomain(_SimpleDomain):
    @override
    def draw(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        n: int,
        type_: _DomainDrawType,
        min: onp.ArrayND[_Float | _Int],
        max: onp.ArrayND[_Float | _Int],
        squeezed_base_shape: _ND,
        rng: ToRNG = None,
    ) -> onp.ArrayND[np.float64]: ...

_ValidateOut0D: TypeAlias = tuple[_RealT, np.dtype[_RealT], onp.Array0D[np.bool_]]
_ValidateOutND: TypeAlias = tuple[onp.ArrayND[_RealT, _ShapeT1], np.dtype[_RealT], onp.ArrayND[np.bool_, _ShapeT1]]

#
class _Parameter(abc.ABC, Generic[_RealT_co]):
    def __init__(
        self,
        /,
        name: str,
        *,
        domain: _Domain,
        symbol: str | None = None,
        typical: _Domain | _ToDomain | None = None,
    ) -> None: ...
    #
    @overload
    @abc.abstractmethod
    def validate(self, /, arr: onp.ToFloat) -> _ValidateOut0D[_RealT_co]: ...
    @overload
    @abc.abstractmethod
    def validate(self, /, arr: onp.ToFloatND) -> _ValidateOutND[_RealT_co]: ...
    #
    def draw(
        self,
        /,
        size: _ND | None = None,
        *,
        rng: ToRNG = None,
        region: _DomainRegion = "domain",
        proportions: _DrawProportions | None = None,
        parameter_values: _ParamValues | None = None,
    ) -> onp.ArrayND[_RealT_co]: ...

class _RealParameter(_Parameter[_FloatT_co], Generic[_FloatT_co]):
    @overload  # type: ignore[override]
    def validate(self, /, arr: onp.ToFloat, parameter_values: _ParamValues) -> _ValidateOut0D[_FloatT_co]: ...
    @overload
    def validate(self, /, arr: onp.ToFloatND, parameter_values: _ParamValues) -> _ValidateOutND[_FloatT_co]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

class _Parameterization:
    parameters: Final[Mapping[str, _Parameter]]

    def __init__(self, /, *parameters: _Parameter) -> None: ...
    def __len__(self, /) -> int: ...
    def copy(self, /) -> Self: ...
    def matches(self, /, parameters: AbstractSet[str]) -> bool: ...
    def validation(self, /, parameter_values: Mapping[str, _Parameter]) -> tuple[onp.ArrayND[np.bool_], np.dtype[_Float]]: ...
    def draw(
        self,
        /,
        sizes: _ND | Sequence[_ND] | None = None,
        rng: ToRNG = None,
        proportions: _DrawProportions | None = None,
        region: _DomainRegion = "domain",
    ) -> dict[str, onp.ArrayND[_Float]]: ...

###

class ContinuousDistribution(_BaseDistribution[_FloatT_co, _ShapeT_co], Generic[_FloatT_co, _ShapeT_co]):
    __array_priority__: ClassVar[float] = 1
    _parameterizations: ClassVar[Sequence[_Parameterization]]

    _not_implemented: Final[str]
    _original_parameters: dict[str, _FloatT_co | onp.ArrayND[_FloatT_co, _ShapeT_co]]

    _variable: _Parameter

    @property
    def tol(self, /) -> float | np.float64 | _Null | None: ...
    @tol.setter
    def tol(self, tol: float | np.float64 | _Null | None, /) -> None: ...
    #
    @property
    def validation_policy(self, /) -> _ValidationPolicy: ...
    @validation_policy.setter
    def validation_policy(self, validation_policy: _ValidationPolicy, /) -> None: ...
    #
    @property
    def cache_policy(self, /) -> _CachePolicy: ...
    @cache_policy.setter
    def cache_policy(self, cache_policy: _CachePolicy, /) -> None: ...
    #
    def __init__(
        self,
        /,
        *,
        tol: _ToTol = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...
    def __neg__(self, /) -> _LinDist[Self, _FloatT_co, _ShapeT_co]: ...
    def __abs__(self, /) -> _FoldDist[Self, _FloatT_co, _ShapeT_co]: ...

    #
    @overload
    def __add__(self, x: float | _Int | np.bool_, /) -> _LinDist[Self, np.float64 | _FloatT_co, _ShapeT_co]: ...
    @overload
    def __add__(self, x: _FloatT, /) -> _LinDist[Self, _FloatT | _FloatT_co, _ShapeT_co]: ...
    @overload
    def __add__(self, x: onp.ToFloat, /) -> _LinDist[Self, _Float, _ShapeT_co]: ...
    @overload
    def __add__(self: _DistT0, x: onp.CanArrayND[_FloatT, _ShapeT], /) -> _LinDist[_DistT0, _FloatT | _FloatT_co, _ShapeT]: ...
    @overload
    def __add__(self: _DistT_1, x: onp.ToFloatStrict1D, /) -> _LinDist[_DistT_1, _Float, _1D]: ...
    @overload
    def __add__(self: _DistT_2, x: onp.ToFloatStrict2D, /) -> _LinDist[_DistT_2, _Float, _2D]: ...
    @overload
    def __add__(self: _DistT_3, x: onp.ToFloatStrict3D, /) -> _LinDist[_DistT_3, _Float, _3D]: ...
    @overload
    def __add__(self, x: onp.ToFloatND, /) -> _LinDist[Self]: ...
    __radd__ = __add__

    #
    @overload
    def __sub__(self, lshift: float | _Int | np.bool_, /) -> _LinDist[Self, np.float64 | _FloatT_co, _ShapeT_co]: ...
    @overload
    def __sub__(self, lshift: _FloatT, /) -> _LinDist[Self, _FloatT | _FloatT_co, _ShapeT_co]: ...
    @overload
    def __sub__(self, lshift: onp.ToFloat, /) -> _LinDist[Self, _Float, _ShapeT_co]: ...
    @overload
    def __sub__(
        self: _DistT0,
        lshift: onp.CanArrayND[_FloatT, _ShapeT],
        /,
    ) -> _LinDist[_DistT0, _FloatT | _FloatT_co, _ShapeT]: ...
    @overload
    def __sub__(self: _DistT_1, lshift: onp.ToFloatStrict1D, /) -> _LinDist[_DistT_1, _Float, _1D]: ...
    @overload
    def __sub__(self: _DistT_2, lshift: onp.ToFloatStrict2D, /) -> _LinDist[_DistT_2, _Float, _2D]: ...
    @overload
    def __sub__(self: _DistT_3, lshift: onp.ToFloatStrict3D, /) -> _LinDist[_DistT_3, _Float, _3D]: ...
    @overload
    def __sub__(self, lshift: onp.ToFloatND, /) -> _LinDist[Self]: ...
    __rsub__ = __sub__

    #
    @overload
    def __mul__(self, scale: float | _Int | np.bool_, /) -> _LinDist[Self, np.float64 | _FloatT_co, _ShapeT_co]: ...
    @overload
    def __mul__(self, scale: _FloatT, /) -> _LinDist[Self, _FloatT | _FloatT_co, _ShapeT_co]: ...
    @overload
    def __mul__(self, scale: onp.ToFloat, /) -> _LinDist[Self, _Float, _ShapeT_co]: ...
    @overload
    def __mul__(
        self: _DistT0,
        scale: onp.CanArrayND[_FloatT, _ShapeT],
        /,
    ) -> _LinDist[_DistT0, _FloatT | _FloatT_co, _ShapeT]: ...
    @overload
    def __mul__(self: _DistT_1, scale: onp.ToFloatStrict1D, /) -> _LinDist[_DistT_1, _Float, _1D]: ...
    @overload
    def __mul__(self: _DistT_2, scale: onp.ToFloatStrict2D, /) -> _LinDist[_DistT_2, _Float, _2D]: ...
    @overload
    def __mul__(self: _DistT_3, scale: onp.ToFloatStrict3D, /) -> _LinDist[_DistT_3, _Float, _3D]: ...
    @overload
    def __mul__(self, scale: onp.ToFloatND, /) -> _LinDist[Self]: ...
    __rmul__ = __mul__

    #
    @overload
    def __truediv__(self, iscale: float | _Int | np.bool_, /) -> _LinDist[Self, np.float64 | _FloatT_co, _ShapeT_co]: ...
    @overload
    def __truediv__(self, iscale: _FloatT, /) -> _LinDist[Self, _FloatT | _FloatT_co, _ShapeT_co]: ...
    @overload
    def __truediv__(self, iscale: onp.ToFloat, /) -> _LinDist[Self, _Float, _ShapeT_co]: ...
    @overload
    def __truediv__(
        self: _DistT0,
        iscale: onp.CanArrayND[_FloatT, _ShapeT],
        /,
    ) -> _LinDist[_DistT0, _FloatT | _FloatT_co, _ShapeT]: ...
    @overload
    def __truediv__(self: _DistT_1, iscale: onp.ToFloatStrict1D, /) -> _LinDist[_DistT_1, _Float, _1D]: ...
    @overload
    def __truediv__(self: _DistT_2, iscale: onp.ToFloatStrict2D, /) -> _LinDist[_DistT_2, _Float, _2D]: ...
    @overload
    def __truediv__(self: _DistT_3, iscale: onp.ToFloatStrict3D, /) -> _LinDist[_DistT_3, _Float, _3D]: ...
    @overload
    def __truediv__(self, iscale: onp.ToFloatND, /) -> _LinDist[Self]: ...
    __rtruediv__ = __truediv__

    #
    def __pow__(self, exp: onp.ToInt, /) -> MonotonicTransformedDistribution[Self, _ShapeT_co]: ...
    __rpow__ = __pow__

    #
    def reset_cache(self, /) -> None: ...
    def plot(
        self,
        x: _PlotQuantity = "x",
        y: _PlotQuantity = "pdf",
        *,
        t: tuple[_PlotQuantity, onp.ToJustFloat, onp.ToJustFloat] = ("cdf", 0.0005, 0.9995),
        ax: _AxesT | None = None,
    ) -> _AxesT: ...

# 7 years of asking and >400 upvotes, but still no higher-kinded typing support: https://github.com/python/typing/issues/548
class TransformedDistribution(ContinuousDistribution[_FloatT_co, _ShapeT_co], Generic[_DistT_co, _FloatT_co, _ShapeT_co]):
    _dist: _DistT_co  # readonly

    def __init__(
        self: _TransDist[ContinuousDistribution[_FloatT, _ShapeT], _FloatT, _ShapeT],  # nice trick, eh?
        X: _DistT_co,
        /,
        *args: Never,
        **kwargs: Unpack[_DistOpts],
    ) -> None: ...

class ShiftedScaledDistribution(_TransDist[_DistT_co, _FloatT_co, _ShapeT_co], Generic[_DistT_co, _FloatT_co, _ShapeT_co]):
    _loc_domain: ClassVar[_RealDomain] = ...
    _loc_param: ClassVar[_RealParameter] = ...
    _scale_domain: ClassVar[_RealDomain] = ...
    _scale_param: ClassVar[_RealParameter] = ...

    loc: _ParamField[_FloatT_co, _ShapeT_co]
    scale: _ParamField[_FloatT_co, _ShapeT_co]

    # TODO(jorenham): override `__[r]{add,sub,mul,truediv}__` so that it returns a `Self` (but maybe with different shape)

class FoldedDistribution(_TransDist[_DistT_co, _FloatT_co, _ShapeT_co], Generic[_DistT_co, _FloatT_co, _ShapeT_co]):
    @overload
    def __init__(self: _FoldDist[_DistT0, _Float, _0D], X: _DistT0, /, *args: Never, **kwargs: Unpack[_DistOpts]) -> None: ...
    @overload
    def __init__(self: _FoldDist[_DistT1, _Float, _1D], X: _DistT1, /, *args: Never, **kwargs: Unpack[_DistOpts]) -> None: ...
    @overload
    def __init__(self: _FoldDist[_DistT2, _Float, _2D], X: _DistT2, /, *args: Never, **kwargs: Unpack[_DistOpts]) -> None: ...
    @overload
    def __init__(self: _FoldDist[_DistT3, _Float, _3D], X: _DistT3, /, *args: Never, **kwargs: Unpack[_DistOpts]) -> None: ...
    @overload
    def __init__(self: _FoldDist[_DistT, _Float, _ND], X: _DistT, /, *args: Never, **kwargs: Unpack[_DistOpts]) -> None: ...

class TruncatedDistribution(_TransDist[_DistT_co, _Float, _ShapeT_co], Generic[_DistT_co, _ShapeT_co]):
    _lb_domain: ClassVar[_RealDomain] = ...
    _lb_param: ClassVar[_RealParameter] = ...
    _ub_domain: ClassVar[_RealDomain] = ...
    _ub_param: ClassVar[_RealParameter] = ...

    lb: _ParamField[_Float, _ShapeT_co]
    ub: _ParamField[_Float, _ShapeT_co]

    @overload
    def __init__(
        self: _TruncDist[_DistT0, _0D],
        X: _DistT0,
        /,
        *args: Never,
        lb: onp.ToFloat = ...,
        ub: onp.ToFloat = ...,
        **kwargs: Unpack[_DistOpts],
    ) -> None: ...
    @overload
    def __init__(
        self: _TruncDist[_DistT1, _1D],
        X: _DistT1,
        /,
        *args: Never,
        lb: _ToFloatMax1D = ...,
        ub: _ToFloatMax1D = ...,
        **kwargs: Unpack[_DistOpts],
    ) -> None: ...
    @overload
    def __init__(
        self: _TruncDist[_DistT2, _2D],
        X: _DistT2,
        /,
        *args: Never,
        lb: _ToFloatMax2D = ...,
        ub: _ToFloatMax2D = ...,
        **kwargs: Unpack[_DistOpts],
    ) -> None: ...
    @overload
    def __init__(
        self: _TruncDist[_DistT3, _3D],
        X: _DistT3,
        /,
        *args: Never,
        lb: _ToFloatMax3D = ...,
        ub: _ToFloatMax3D = ...,
        **kwargs: Unpack[_DistOpts],
    ) -> None: ...
    @overload
    def __init__(
        self: _TruncDist[_DistT, _ND],
        X: _DistT,
        /,
        *args: Never,
        lb: _ToFloatMaxND = ...,
        ub: _ToFloatMaxND = ...,
        **kwargs: Unpack[_DistOpts],
    ) -> None: ...

# always float64 or longdouble
class OrderStatisticDistribution(_TransDist[_DistT_co, _OutFloat, _ShapeT_co], Generic[_DistT_co, _ShapeT_co]):
    # these should actually be integral; but the `_IntegerDomain` isn't finished yet
    _r_domain: ClassVar[_RealDomain] = ...
    _r_param: ClassVar[_RealParameter] = ...
    _n_domain: ClassVar[_RealDomain] = ...
    _n_param: ClassVar[_RealParameter] = ...

    @overload
    def __init__(
        self: OrderStatisticDistribution[_DistT0, _0D],
        dist: _DistT0,
        /,
        *args: Never,
        r: onp.ToJustInt,
        n: onp.ToJustInt,
        **kwargs: Unpack[_DistOpts],
    ) -> None: ...
    @overload
    def __init__(
        self: OrderStatisticDistribution[_DistT1, _1D],
        dist: _DistT1,
        /,
        *args: Never,
        r: _ToJustIntMax1D,
        n: _ToJustIntMax1D,
        **kwargs: Unpack[_DistOpts],
    ) -> None: ...
    @overload
    def __init__(
        self: OrderStatisticDistribution[_DistT2, _2D],
        dist: _DistT2,
        /,
        *args: Never,
        r: _ToJustIntMax2D,
        n: _ToJustIntMax2D,
        **kwargs: Unpack[_DistOpts],
    ) -> None: ...
    @overload
    def __init__(
        self: OrderStatisticDistribution[_DistT3, _3D],
        dist: _DistT3,
        /,
        *args: Never,
        r: _ToJustIntMax3D,
        n: _ToJustIntMax3D,
        **kwargs: Unpack[_DistOpts],
    ) -> None: ...
    @overload
    def __init__(
        self: OrderStatisticDistribution[_DistT, _ND],
        X: _DistT,
        /,
        *args: Never,
        r: _ToJustIntMaxND,
        n: _ToJustIntMaxND,
        **kwargs: Unpack[_DistOpts],
    ) -> None: ...

# without HKT there's no reasonable way tot determine the floating scalar type
class MonotonicTransformedDistribution(_TransDist[_DistT_co, _Float, _ShapeT_co], Generic[_DistT_co, _ShapeT_co]):
    _g: Final[_Elementwise]
    _h: Final[_Elementwise]
    _dh: Final[_Elementwise]
    _logdh: Final[_Elementwise]
    _increasing: Final[bool]
    _repr_pattern: Final[str]
    _str_pattern: Final[str]

    def __init__(
        self: MonotonicTransformedDistribution[_CDist[_ShapeT], _ShapeT],
        X: _DistT_co,
        /,
        *args: Never,
        g: _Elementwise,
        h: _Elementwise,
        dh: _Elementwise,
        logdh: _Elementwise | None = None,
        increasing: bool = True,
        repr_pattern: str | None = None,
        str_pattern: str | None = None,
        **kwargs: Unpack[_DistOpts],
    ) -> None: ...

class Mixture(_BaseDistribution[_FloatT_co, _0D], Generic[_FloatT_co]):
    _shape: _0D
    _dtype: np.dtype[_FloatT_co]
    _components: Sequence[_CDist0[_FloatT_co]]
    _weights: onp.Array1D[_FloatT_co]
    validation_policy: None

    @property
    def components(self, /) -> list[_CDist0[_FloatT_co]]: ...
    @property
    def weights(self, /) -> onp.Array1D[_FloatT_co]: ...
    #
    def __init__(self, /, components: Sequence[_CDist0[_FloatT_co]], *, weights: onp.ToFloat1D | None = None) -> None: ...
    #
    @override
    def kurtosis(self, /, *, method: _SMomentMethod | None = None) -> _OutFloat: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

###

# still waiting on the intersection type PEP...

@overload
def truncate(X: _DistT0, lb: onp.ToFloat = ..., ub: onp.ToFloat = ...) -> _TruncDist[_DistT0, _0D]: ...
@overload
def truncate(X: _DistT1, lb: _ToFloatMax1D = ..., ub: _ToFloatMax1D = ...) -> _TruncDist[_DistT1, _1D]: ...
@overload
def truncate(X: _DistT2, lb: _ToFloatMax2D = ..., ub: _ToFloatMax2D = ...) -> _TruncDist[_DistT2, _2D]: ...
@overload
def truncate(X: _DistT3, lb: _ToFloatMax3D = ..., ub: _ToFloatMax3D = ...) -> _TruncDist[_DistT3, _3D]: ...
@overload
def truncate(X: _DistT, lb: _ToFloatMaxND = ..., ub: _ToFloatMaxND = ...) -> _TruncDist[_DistT, _ND]: ...

#
@overload
def order_statistic(X: _DistT0, /, *, r: onp.ToJustInt, n: onp.ToJustInt) -> OrderStatisticDistribution[_DistT0, _0D]: ...
@overload
def order_statistic(X: _DistT1, /, *, r: _ToJustIntMax1D, n: _ToJustIntMax1D) -> OrderStatisticDistribution[_DistT1, _1D]: ...
@overload
def order_statistic(X: _DistT2, /, *, r: _ToJustIntMax2D, n: _ToJustIntMax2D) -> OrderStatisticDistribution[_DistT2, _2D]: ...
@overload
def order_statistic(X: _DistT3, /, *, r: _ToJustIntMax3D, n: _ToJustIntMax3D) -> OrderStatisticDistribution[_DistT3, _3D]: ...
@overload
def order_statistic(X: _DistT, /, *, r: _ToJustIntMaxND, n: _ToJustIntMaxND) -> OrderStatisticDistribution[_DistT, _ND]: ...

#
@overload
def abs(X: _DistT0, /) -> _FoldDist[_DistT0, _Float, _0D]: ...
@overload
def abs(X: _DistT1, /) -> _FoldDist[_DistT1, _Float, _1D]: ...
@overload
def abs(X: _DistT2, /) -> _FoldDist[_DistT2, _Float, _2D]: ...
@overload
def abs(X: _DistT3, /) -> _FoldDist[_DistT3, _Float, _3D]: ...
@overload
def abs(X: _DistT, /) -> _FoldDist[_DistT, _Float, _ND]: ...

#
@overload
def exp(X: _DistT0, /) -> MonotonicTransformedDistribution[_DistT0, _0D]: ...
@overload
def exp(X: _DistT1, /) -> MonotonicTransformedDistribution[_DistT1, _1D]: ...
@overload
def exp(X: _DistT2, /) -> MonotonicTransformedDistribution[_DistT2, _2D]: ...
@overload
def exp(X: _DistT3, /) -> MonotonicTransformedDistribution[_DistT3, _3D]: ...
@overload
def exp(X: _DistT, /) -> MonotonicTransformedDistribution[_DistT, _ND]: ...

#
@overload
def log(X: _DistT0, /) -> MonotonicTransformedDistribution[_DistT0, _0D]: ...
@overload
def log(X: _DistT1, /) -> MonotonicTransformedDistribution[_DistT1, _1D]: ...
@overload
def log(X: _DistT2, /) -> MonotonicTransformedDistribution[_DistT2, _2D]: ...
@overload
def log(X: _DistT3, /) -> MonotonicTransformedDistribution[_DistT3, _3D]: ...
@overload
def log(X: _DistT, /) -> MonotonicTransformedDistribution[_DistT, _ND]: ...

# NOTE: These currently don't support >0-d parameters, and it looks like they always return float64, regardless of dtype
@type_check_only
class CustomDistribution(ContinuousDistribution[np.float64, _0D]):
    _dtype: np.dtype[_Float]  # ignored

def make_distribution(dist: rv_continuous) -> type[CustomDistribution]: ...
