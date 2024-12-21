# mypy: disable-error-code="explicit-override"
# pyright: reportUnannotatedClassAttribute=false

import abc
from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
from typing import Any, ClassVar, Final, Generic, Literal as L, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import LiteralString, Never, Self, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp
import optype.typing as opt
from scipy._typing import AnyShape, ToRNG
from ._probability_distribution import _BaseDistribution

# TODO:
# `__all__ = ["Mixture", "abs", "exp", "log", "make_distribution", "order_statistic", "truncate"]

_Float: TypeAlias = np.float64 | np.longdouble
_FloatingT = TypeVar("_FloatingT", bound=np.floating[Any], default=np.floating[Any])
_FloatingT_co = TypeVar("_FloatingT_co", bound=np.floating[Any], default=np.floating[Any], covariant=True)
_RealT = TypeVar("_RealT", bound=np.floating[Any] | np.integer[Any], default=np.floating[Any] | np.integer[Any])
_RealT_co = TypeVar(
    "_RealT_co",
    bound=np.floating[Any] | np.integer[Any],
    default=np.floating[Any] | np.integer[Any],
    covariant=True,
)

_ShapeT0 = TypeVar("_ShapeT0", bound=tuple[int, ...], default=tuple[int, ...])
_ShapeT1 = TypeVar("_ShapeT1", bound=onp.AtLeast1D, default=onp.AtLeast1D)
_ShapeT0_co = TypeVar("_ShapeT0_co", bound=tuple[int, ...], default=tuple[int, ...], covariant=True)

_CDistT0 = TypeVar("_CDistT0", bound=_CDist0)
_CDistT1 = TypeVar("_CDistT1", bound=_CDist[tuple[int]])
_CDistT_1 = TypeVar("_CDistT_1", bound=_CDist[onp.AtMost1D])
_CDistT2 = TypeVar("_CDistT2", bound=_CDist[tuple[int, int]])
_CDistT_2 = TypeVar("_CDistT_2", bound=_CDist[onp.AtMost2D])
_CDistT3 = TypeVar("_CDistT3", bound=_CDist[tuple[int, int, int]])
_CDistT_3 = TypeVar("_CDistT_3", bound=_CDist[onp.AtMost3D])
_CDistT = TypeVar("_CDistT", bound=ContinuousDistribution)
_CDistT_co = TypeVar("_CDistT_co", bound=ContinuousDistribution, default=ContinuousDistribution, covariant=True)

# placeholder for `matplotlib.axes.Axes`
_Axes: TypeAlias = object
_AxesT = TypeVar("_AxesT", bound=_Axes, default=Any)

###

_JustFloat: TypeAlias = opt.Just[float] | np.floating[Any]
_Null: TypeAlias = opt.Just[object]

_DomainRegion: TypeAlias = L["domain", "typical"]
_DomainDrawType: TypeAlias = L["in", "out", "on", "nan"]
_ValidationPolicy: TypeAlias = L["skip_all"] | None
_CachePolicy: TypeAlias = L["no_cache"] | None
_PlotQuantity: TypeAlias = L["x", "pdf", "cdf", "ccdf", "icdf", "iccdf", "logpdf", "logcdf", "logccdf", "ilogcdf", "ilogccdf"]
_SMomentMethod: TypeAlias = L["formula", "general", "transform", "normalize", "cache"]

_ParamValues: TypeAlias = Mapping[str, onp.ToFloat | onp.ToFloatND]
_ToDomain: TypeAlias = _Domain | tuple[onp.ToFloat | str, onp.ToFloat | str]
_DrawProportions: TypeAlias = tuple[onp.ToFloat, onp.ToFloat, onp.ToFloat, onp.ToFloat]

_CDist: TypeAlias = ContinuousDistribution[np.floating[Any], _ShapeT0]
_CDist0: TypeAlias = ContinuousDistribution[_FloatingT, tuple[()]]

@type_check_only
class _ParameterField(Protocol[_FloatingT_co, _ShapeT0_co]):
    # This actually works (even on mypy)!
    @overload
    def __get__(
        self: _ParameterField[_FloatingT, tuple[()]],
        instance: object,
        owner: type | None = None,
        /,
    ) -> _FloatingT: ...
    @overload
    def __get__(
        self: _ParameterField[_FloatingT, _ShapeT1],
        instance: object,
        owner: type | None = None,
        /,
    ) -> onp.ArrayND[_FloatingT, _ShapeT1]: ...

###

_null: Final[_Null] = ...

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
    def draw(self, /, n: int) -> onp.ArrayND[_FloatingT]: ...
    @abc.abstractmethod
    def get_numerical_endpoints(
        self,
        /,
        x: _ParamValues,
    ) -> tuple[onp.ArrayND[_Float], onp.ArrayND[_Float]]: ...

# TODO(jorenham): Generic dtype
class _SimpleDomain(_Domain, metaclass=abc.ABCMeta):
    def __init__(
        self,
        /,
        endpoints: tuple[onp.ToFloat | str, onp.ToFloat | str] = ...,
        inclusive: tuple[bool, bool] = (False, False),
    ) -> None: ...

    #
    @override
    def __str__(self, /) -> str: ...  # noqa: PYI029
    @override
    def get_numerical_endpoints(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        parameter_values: _ParamValues,
    ) -> tuple[onp.ArrayND[_Float], onp.ArrayND[_Float]]: ...
    @override
    def contains(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        item: onp.ArrayND[np.integer[Any] | np.floating[Any]],
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
        min: onp.ArrayND[np.floating[Any] | np.integer[Any]],
        max: onp.ArrayND[np.floating[Any] | np.integer[Any]],
        squeezed_base_shape: tuple[int, ...],
        rng: ToRNG = None,
    ) -> onp.ArrayND[np.float64]: ...

_ParamValidated0D: TypeAlias = tuple[_RealT, np.dtype[_RealT], onp.Array0D[np.bool_]]
_ParamValidatedND: TypeAlias = tuple[onp.ArrayND[_RealT, _ShapeT1], np.dtype[_RealT], onp.ArrayND[np.bool_, _ShapeT1]]

#
class _Parameter(abc.ABC, Generic[_RealT_co]):
    def __init__(self, /, name: str, *, domain: _Domain, symbol: str | None = None, typical: _ToDomain | None = None) -> None: ...

    #
    @overload
    @abc.abstractmethod
    def validate(self, /, arr: onp.ToFloat) -> _ParamValidated0D[_RealT_co]: ...
    @overload
    @abc.abstractmethod
    def validate(self, /, arr: onp.ToFloatND) -> _ParamValidatedND[_RealT_co]: ...

    #
    def draw(
        self,
        /,
        size: tuple[int, ...] | None = None,
        *,
        rng: ToRNG = None,
        region: _DomainRegion = "domain",
        proportions: _DrawProportions | None = None,
        parameter_values: _ParamValues | None = None,
    ) -> onp.ArrayND[_RealT_co]: ...

class _RealParameter(_Parameter[_FloatingT_co], Generic[_FloatingT_co]):
    @overload  # type: ignore[override]
    def validate(self, /, arr: onp.ToFloat, parameter_values: _ParamValues) -> _ParamValidated0D[_FloatingT_co]: ...
    @overload
    def validate(self, /, arr: onp.ToFloatND, parameter_values: _ParamValues) -> _ParamValidatedND[_FloatingT_co]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

class _Parameterization:
    parameters: Final[Mapping[str, _Parameter]]

    def __init__(self, /, *parameters: _Parameter) -> None: ...
    def __len__(self, /) -> int: ...
    def copy(self, /) -> Self: ...
    def matches(self, /, parameters: AbstractSet[str]) -> bool: ...
    def validation(
        self,
        /,
        parameter_values: Mapping[str, _Parameter],
    ) -> tuple[onp.ArrayND[np.bool_], np.dtype[np.floating[Any]]]: ...
    def draw(
        self,
        /,
        sizes: tuple[int, ...] | Sequence[tuple[int, ...]] | None = None,
        rng: ToRNG = None,
        proportions: _DrawProportions | None = None,
        region: _DomainRegion = "domain",
    ) -> dict[str, onp.ArrayND[np.floating[Any]]]: ...

###

class ContinuousDistribution(_BaseDistribution[_FloatingT_co, _ShapeT0_co], Generic[_FloatingT_co, _ShapeT0_co]):
    __array_priority__: ClassVar[float] = 1
    _parameterizations: ClassVar[Sequence[_Parameterization]]

    _not_implemented: Final[str]
    _original_parameters: dict[str, _FloatingT_co | onp.ArrayND[_FloatingT_co, _ShapeT0_co]]

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
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...

    #
    def __neg__(self, /) -> ShiftedScaledDistribution[Self, _FloatingT_co, _ShapeT0_co]: ...
    def __abs__(self, /) -> FoldedDistribution[Self, _FloatingT_co, _ShapeT0_co]: ...

    #
    @overload
    def __add__(
        self, rshift: float | np.integer[Any] | np.bool_, /
    ) -> ShiftedScaledDistribution[Self, np.float64 | _FloatingT_co, _ShapeT0_co]: ...
    @overload
    def __add__(self, rshift: _FloatingT, /) -> ShiftedScaledDistribution[Self, _FloatingT | _FloatingT_co, _ShapeT0_co]: ...
    @overload
    def __add__(self, rshift: onp.ToFloat, /) -> ShiftedScaledDistribution[Self, np.floating[Any], _ShapeT0_co]: ...
    @overload
    def __add__(
        self: _CDistT0, rshift: onp.CanArrayND[_FloatingT, _ShapeT0], /
    ) -> ShiftedScaledDistribution[_CDistT0, _FloatingT | _FloatingT_co, _ShapeT0]: ...
    @overload
    def __add__(
        self: _CDistT_1, rshift: onp.ToFloatStrict1D, /
    ) -> ShiftedScaledDistribution[_CDistT_1, np.floating[Any], tuple[int]]: ...
    @overload
    def __add__(
        self: _CDistT_2, rshift: onp.ToFloatStrict2D, /
    ) -> ShiftedScaledDistribution[_CDistT_2, np.floating[Any], tuple[int, int]]: ...
    @overload
    def __add__(
        self: _CDistT_3, rshift: onp.ToFloatStrict3D, /
    ) -> ShiftedScaledDistribution[_CDistT_3, np.floating[Any], tuple[int, int, int]]: ...
    @overload
    def __add__(self, rshift: onp.ToFloatND, /) -> ShiftedScaledDistribution[Self]: ...
    __radd__ = __add__

    #
    @overload
    def __sub__(
        self, lshift: float | np.integer[Any] | np.bool_, /
    ) -> ShiftedScaledDistribution[Self, np.float64 | _FloatingT_co, _ShapeT0_co]: ...
    @overload
    def __sub__(self, lshift: _FloatingT, /) -> ShiftedScaledDistribution[Self, _FloatingT | _FloatingT_co, _ShapeT0_co]: ...
    @overload
    def __sub__(self, lshift: onp.ToFloat, /) -> ShiftedScaledDistribution[Self, np.floating[Any], _ShapeT0_co]: ...
    @overload
    def __sub__(
        self: _CDistT0, lshift: onp.CanArrayND[_FloatingT, _ShapeT0], /
    ) -> ShiftedScaledDistribution[_CDistT0, _FloatingT | _FloatingT_co, _ShapeT0]: ...
    @overload
    def __sub__(
        self: _CDistT_1, lshift: onp.ToFloatStrict1D, /
    ) -> ShiftedScaledDistribution[_CDistT_1, np.floating[Any], tuple[int]]: ...
    @overload
    def __sub__(
        self: _CDistT_2, lshift: onp.ToFloatStrict2D, /
    ) -> ShiftedScaledDistribution[_CDistT_2, np.floating[Any], tuple[int, int]]: ...
    @overload
    def __sub__(
        self: _CDistT_3, lshift: onp.ToFloatStrict3D, /
    ) -> ShiftedScaledDistribution[_CDistT_3, np.floating[Any], tuple[int, int, int]]: ...
    @overload
    def __sub__(self, lshift: onp.ToFloatND, /) -> ShiftedScaledDistribution[Self]: ...
    __rsub__ = __sub__

    #
    @overload
    def __mul__(
        self, scale: float | np.integer[Any] | np.bool_, /
    ) -> ShiftedScaledDistribution[Self, np.float64 | _FloatingT_co, _ShapeT0_co]: ...
    @overload
    def __mul__(self, scale: _FloatingT, /) -> ShiftedScaledDistribution[Self, _FloatingT | _FloatingT_co, _ShapeT0_co]: ...
    @overload
    def __mul__(self, scale: onp.ToFloat, /) -> ShiftedScaledDistribution[Self, np.floating[Any], _ShapeT0_co]: ...
    @overload
    def __mul__(
        self: _CDistT0, scale: onp.CanArrayND[_FloatingT, _ShapeT0], /
    ) -> ShiftedScaledDistribution[_CDistT0, _FloatingT | _FloatingT_co, _ShapeT0]: ...
    @overload
    def __mul__(
        self: _CDistT_1, scale: onp.ToFloatStrict1D, /
    ) -> ShiftedScaledDistribution[_CDistT_1, np.floating[Any], tuple[int]]: ...
    @overload
    def __mul__(
        self: _CDistT_2, scale: onp.ToFloatStrict2D, /
    ) -> ShiftedScaledDistribution[_CDistT_2, np.floating[Any], tuple[int, int]]: ...
    @overload
    def __mul__(
        self: _CDistT_3, scale: onp.ToFloatStrict3D, /
    ) -> ShiftedScaledDistribution[_CDistT_3, np.floating[Any], tuple[int, int, int]]: ...
    @overload
    def __mul__(self, scale: onp.ToFloatND, /) -> ShiftedScaledDistribution[Self]: ...
    __rmul__ = __mul__

    #
    @overload
    def __truediv__(
        self, iscale: float | np.integer[Any] | np.bool_, /
    ) -> ShiftedScaledDistribution[Self, np.float64 | _FloatingT_co, _ShapeT0_co]: ...
    @overload
    def __truediv__(self, iscale: _FloatingT, /) -> ShiftedScaledDistribution[Self, _FloatingT | _FloatingT_co, _ShapeT0_co]: ...
    @overload
    def __truediv__(self, iscale: onp.ToFloat, /) -> ShiftedScaledDistribution[Self, np.floating[Any], _ShapeT0_co]: ...
    @overload
    def __truediv__(
        self: _CDistT0, iscale: onp.CanArrayND[_FloatingT, _ShapeT0], /
    ) -> ShiftedScaledDistribution[_CDistT0, _FloatingT | _FloatingT_co, _ShapeT0]: ...
    @overload
    def __truediv__(
        self: _CDistT_1, iscale: onp.ToFloatStrict1D, /
    ) -> ShiftedScaledDistribution[_CDistT_1, np.floating[Any], tuple[int]]: ...
    @overload
    def __truediv__(
        self: _CDistT_2, iscale: onp.ToFloatStrict2D, /
    ) -> ShiftedScaledDistribution[_CDistT_2, np.floating[Any], tuple[int, int]]: ...
    @overload
    def __truediv__(
        self: _CDistT_3, iscale: onp.ToFloatStrict3D, /
    ) -> ShiftedScaledDistribution[_CDistT_3, np.floating[Any], tuple[int, int, int]]: ...
    @overload
    def __truediv__(self, iscale: onp.ToFloatND, /) -> ShiftedScaledDistribution[Self]: ...
    __rtruediv__ = __truediv__

    #
    def __pow__(self, exp: onp.ToInt, /) -> MonotonicTransformedDistribution[Self, _ShapeT0_co]: ...
    __rpow__ = __pow__

    #
    def reset_cache(self, /) -> None: ...
    def plot(
        self,
        x: _PlotQuantity = "x",
        y: _PlotQuantity = "pdf",
        *,
        t: tuple[_PlotQuantity, _JustFloat, _JustFloat] = ("cdf", 0.0005, 0.9995),
        ax: _AxesT | None = None,
    ) -> _AxesT: ...

    #
    # NOTE: This will be removed in 1.15.0rc2, see https://github.com/scipy/scipy/pull/22149
    @overload
    def llf(self, sample: onp.ToFloat | onp.ToFloatND, /, *, axis: None) -> _Float: ...
    @overload
    def llf(self: _CDist0, sample: onp.ToFloat | onp.ToFloatStrict1D, /, *, axis: AnyShape | None = -1) -> _Float: ...
    @overload
    def llf(
        self: _CDist[_ShapeT1], sample: onp.ToFloat | onp.ToFloatStrict1D, /, *, axis: AnyShape = -1
    ) -> onp.ArrayND[_Float, _ShapeT1]: ...
    @overload
    def llf(
        self: _CDist0, sample: onp.ToFloatStrict2D, /, *, axis: op.CanIndex | tuple[op.CanIndex] = -1
    ) -> onp.Array1D[_Float]: ...
    @overload
    def llf(self: _CDist0, sample: onp.ToFloatStrict2D, /, *, axis: tuple[op.CanIndex, op.CanIndex]) -> _Float: ...
    @overload
    def llf(
        self: _CDist0, sample: onp.ToFloatStrict3D, /, *, axis: op.CanIndex | tuple[op.CanIndex] = -1
    ) -> onp.Array2D[_Float]: ...
    @overload
    def llf(self: _CDist0, sample: onp.ToFloatStrict3D, /, *, axis: tuple[op.CanIndex, op.CanIndex]) -> onp.Array1D[_Float]: ...
    @overload
    def llf(self: _CDist0, sample: onp.ToFloatStrict3D, /, *, axis: tuple[op.CanIndex, op.CanIndex, op.CanIndex]) -> _Float: ...
    @overload
    def llf(
        self: _CDist[_ShapeT1], sample: onp.ToFloat | onp.ToFloatND, /, *, axis: AnyShape = -1
    ) -> onp.Array[_ShapeT1, _Float] | onp.ArrayND[_Float]: ...  # the first union type is needed on numpy <2.1
    @overload
    def llf(self, sample: onp.ToFloat | onp.ToFloatND, /, *, axis: AnyShape | None = -1) -> _Float | onp.ArrayND[_Float]: ...

_ElementwiseFunction: TypeAlias = Callable[[onp.ArrayND[np.float64]], onp.ArrayND[_FloatingT]]

# 7 years of asking and >400 upvotes, but still no higher-kinded typing support: https://github.com/python/typing/issues/548
class TransformedDistribution(
    ContinuousDistribution[_FloatingT_co, _ShapeT0_co],
    Generic[_CDistT_co, _FloatingT_co, _ShapeT0_co],
):
    def __init__(
        self: TransformedDistribution[ContinuousDistribution[_FloatingT, _ShapeT0], _FloatingT, _ShapeT0],  # nice trick, eh?
        X: _CDistT_co,
        /,
        *args: Never,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...

class ShiftedScaledDistribution(
    TransformedDistribution[_CDistT_co, _FloatingT_co, _ShapeT0_co],
    Generic[_CDistT_co, _FloatingT_co, _ShapeT0_co],
):
    _loc_domain: ClassVar[_RealDomain] = ...
    _loc_param: ClassVar[_RealParameter] = ...

    _scale_domain: ClassVar[_RealDomain] = ...
    _scale_param: ClassVar[_RealParameter] = ...

    loc: _ParameterField[_FloatingT_co, _ShapeT0_co]
    scale: _ParameterField[_FloatingT_co, _ShapeT0_co]

    # TODO(jorenham): override `__[r]{add,sub,mul,truediv}__` so that it returns a `Self` (but maybe with different shape)

class FoldedDistribution(
    TransformedDistribution[_CDistT_co, _FloatingT_co, _ShapeT0_co],
    Generic[_CDistT_co, _FloatingT_co, _ShapeT0_co],
):
    @overload
    def __init__(
        self: FoldedDistribution[_CDistT0, np.floating[Any], tuple[()]],
        X: _CDistT0,
        /,
        *args: Never,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...
    @overload
    def __init__(
        self: FoldedDistribution[_CDistT1, np.floating[Any], tuple[int]],
        X: _CDistT1,
        /,
        *args: Never,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...
    @overload
    def __init__(
        self: FoldedDistribution[_CDistT2, np.floating[Any], tuple[int, int]],
        X: _CDistT2,
        /,
        *args: Never,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...
    @overload
    def __init__(
        self: FoldedDistribution[_CDistT3, np.floating[Any], tuple[int, int, int]],
        X: _CDistT3,
        /,
        *args: Never,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...
    @overload
    def __init__(
        self: FoldedDistribution[_CDistT, np.floating[Any], tuple[int, ...]],
        X: _CDistT,
        /,
        *args: Never,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...

class TruncatedDistribution(
    TransformedDistribution[_CDistT_co, _FloatingT_co, _ShapeT0_co],
    Generic[_CDistT_co, _FloatingT_co, _ShapeT0_co],
):
    _lb_domain: ClassVar[_RealDomain] = ...
    _lb_param: ClassVar[_RealParameter] = ...

    _ub_domain: ClassVar[_RealDomain] = ...
    _ub_param: ClassVar[_RealParameter] = ...

    lb: _ParameterField[_FloatingT_co, _ShapeT0_co]
    ub: _ParameterField[_FloatingT_co, _ShapeT0_co]

    @overload
    def __init__(
        self: TruncatedDistribution[_CDistT0, np.floating[Any], tuple[()]],
        X: _CDistT0,
        /,
        *args: Never,
        lb: onp.ToFloat = ...,
        ub: onp.ToFloat = ...,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...
    @overload
    def __init__(
        self: TruncatedDistribution[_CDistT1, np.floating[Any], tuple[int]],
        X: _CDistT1,
        /,
        *args: Never,
        lb: onp.ToFloat | onp.ToFloatStrict1D = ...,
        ub: onp.ToFloat | onp.ToFloatStrict1D = ...,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...
    @overload
    def __init__(
        self: TruncatedDistribution[_CDistT2, np.floating[Any], tuple[int, int]],
        X: _CDistT2,
        /,
        *args: Never,
        lb: onp.ToFloat | onp.ToFloatStrict1D | onp.ToFloatStrict2D = ...,
        ub: onp.ToFloat | onp.ToFloatStrict1D | onp.ToFloatStrict2D = ...,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...
    @overload
    def __init__(
        self: TruncatedDistribution[_CDistT3, np.floating[Any], tuple[int, int, int]],
        X: _CDistT3,
        /,
        *args: Never,
        lb: onp.ToFloat | onp.ToFloatStrict1D | onp.ToFloatStrict2D | onp.ToFloatStrict3D = ...,
        ub: onp.ToFloat | onp.ToFloatStrict1D | onp.ToFloatStrict2D | onp.ToFloatStrict3D = ...,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...
    @overload
    def __init__(
        self: TruncatedDistribution[_CDistT, np.floating[Any], tuple[int, ...]],
        X: _CDistT,
        /,
        *args: Never,
        lb: onp.ToFloat | onp.ToFloatND = ...,
        ub: onp.ToFloat | onp.ToFloatND = ...,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...

# without HKT there's no reasonable way tot determine the floating scalar type
class MonotonicTransformedDistribution(
    TransformedDistribution[_CDistT_co, np.floating[Any], _ShapeT0_co],
    Generic[_CDistT_co, _ShapeT0_co],
):
    _g: Final[_ElementwiseFunction]
    _h: Final[_ElementwiseFunction]
    _dh: Final[_ElementwiseFunction]
    _logdh: Final[_ElementwiseFunction]
    _increasing: Final[bool]
    _repr_pattern: Final[str]

    def __init__(
        self: MonotonicTransformedDistribution[_CDist[_ShapeT0], _ShapeT0],
        X: _CDistT_co,
        /,
        *args: Never,
        g: _ElementwiseFunction,
        h: _ElementwiseFunction,
        dh: _ElementwiseFunction,
        logdh: _ElementwiseFunction | None = None,
        increasing: bool = True,
        repr_pattern: str | None = None,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...

class OrderStatisticDistribution(TransformedDistribution[_CDistT_co, np.float64, _ShapeT0_co], Generic[_CDistT_co, _ShapeT0_co]):
    # TODO(jorenham)
    ...

class Mixture(_BaseDistribution[_FloatingT_co, tuple[()]], Generic[_FloatingT_co]):
    _shape: tuple[()]
    _dtype: np.dtype[_FloatingT_co]
    _components: Sequence[_CDist0[_FloatingT_co]]
    _weights: onp.Array1D[_FloatingT_co]
    validation_policy: None

    @property
    def components(self, /) -> list[_CDist0[_FloatingT_co]]: ...
    @property
    def weights(self, /) -> onp.Array1D[_FloatingT_co]: ...

    #
    def __init__(self, /, components: Sequence[_CDist0[_FloatingT_co]], *, weights: onp.ToFloat1D | None = None) -> None: ...

    #
    @override
    def kurtosis(self, /, *, method: _SMomentMethod | None = None) -> _Float: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

###

# still waiting on the intersection type PEP...
@overload
def truncate(
    X: _CDistT0,
    lb: onp.ToFloat = ...,
    ub: onp.ToFloat = ...,
) -> TruncatedDistribution[_CDistT0, np.floating[Any], tuple[()]]: ...
@overload
def truncate(
    X: _CDistT1,
    lb: onp.ToFloat | onp.ToFloatStrict1D = ...,
    ub: onp.ToFloat | onp.ToFloatStrict1D = ...,
) -> TruncatedDistribution[_CDistT1, np.floating[Any], tuple[int]]: ...
@overload
def truncate(
    X: _CDistT2,
    lb: onp.ToFloat | onp.ToFloatStrict1D | onp.ToFloatStrict2D = ...,
    ub: onp.ToFloat | onp.ToFloatStrict1D | onp.ToFloatStrict2D = ...,
) -> TruncatedDistribution[_CDistT2, np.floating[Any], tuple[int, int]]: ...
@overload
def truncate(
    X: _CDistT3,
    lb: onp.ToFloat | onp.ToFloatStrict1D | onp.ToFloatStrict2D | onp.ToFloatStrict3D = ...,
    ub: onp.ToFloat | onp.ToFloatStrict1D | onp.ToFloatStrict2D | onp.ToFloatStrict3D = ...,
) -> TruncatedDistribution[_CDistT3, np.floating[Any], tuple[int, int, int]]: ...
@overload
def truncate(
    X: _CDistT,
    lb: onp.ToFloat | onp.ToFloatND = ...,
    ub: onp.ToFloat | onp.ToFloatND = ...,
) -> TruncatedDistribution[_CDistT, np.floating[Any], tuple[int, ...]]: ...

#
@overload
def abs(X: _CDistT0, /) -> FoldedDistribution[_CDistT0, np.floating[Any], tuple[()]]: ...
@overload
def abs(X: _CDistT1, /) -> FoldedDistribution[_CDistT1, np.floating[Any], tuple[int]]: ...
@overload
def abs(X: _CDistT2, /) -> FoldedDistribution[_CDistT2, np.floating[Any], tuple[int, int]]: ...
@overload
def abs(X: _CDistT3, /) -> FoldedDistribution[_CDistT3, np.floating[Any], tuple[int, int, int]]: ...
@overload
def abs(X: _CDistT, /) -> FoldedDistribution[_CDistT, np.floating[Any], tuple[int, ...]]: ...

#
@overload
def exp(X: _CDistT0, /) -> MonotonicTransformedDistribution[_CDistT0, tuple[()]]: ...
@overload
def exp(X: _CDistT1, /) -> MonotonicTransformedDistribution[_CDistT1, tuple[int]]: ...
@overload
def exp(X: _CDistT2, /) -> MonotonicTransformedDistribution[_CDistT2, tuple[int, int]]: ...
@overload
def exp(X: _CDistT3, /) -> MonotonicTransformedDistribution[_CDistT3, tuple[int, int, int]]: ...
@overload
def exp(X: _CDistT, /) -> MonotonicTransformedDistribution[_CDistT, tuple[int, ...]]: ...

#
@overload
def log(X: _CDistT0, /) -> MonotonicTransformedDistribution[_CDistT0, tuple[()]]: ...
@overload
def log(X: _CDistT1, /) -> MonotonicTransformedDistribution[_CDistT1, tuple[int]]: ...
@overload
def log(X: _CDistT2, /) -> MonotonicTransformedDistribution[_CDistT2, tuple[int, int]]: ...
@overload
def log(X: _CDistT3, /) -> MonotonicTransformedDistribution[_CDistT3, tuple[int, int, int]]: ...
@overload
def log(X: _CDistT, /) -> MonotonicTransformedDistribution[_CDistT, tuple[int, ...]]: ...
