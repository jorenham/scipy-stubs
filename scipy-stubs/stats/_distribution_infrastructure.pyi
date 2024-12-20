# mypy: disable-error-code="explicit-override"
# pyright: reportUnannotatedClassAttribute=false

import abc
from collections.abc import Mapping, Sequence, Set as AbstractSet
from typing import Any, ClassVar, Final, Generic, Literal as L, TypeAlias, overload
from typing_extensions import LiteralString, Never, Self, TypeVar, override

import numpy as np
import optype as op
import optype.numpy as onp
import optype.typing as opt
from scipy._typing import AnyShape, ToRNG
from ._probability_distribution import _BaseDistribution

# TODO:
# `__all__ = ["Mixture", "abs", "exp", "log", "make_distribution", "order_statistic", "truncate"]

_FloatT = TypeVar("_FloatT", bound=np.floating[Any])

_ValidationPolicy: TypeAlias = L["skip_all"] | None
_CachePolicy: TypeAlias = L["no_cache"] | None

###

# TODO(jorenham): Generic dtype
class _Domain(abc.ABC):
    # NOTE: This is a `ClassVar[dict[str, float]]` that's overridden as instance attribute in `_SimpleDomain`.
    # https://github.com/scipy/scipy/pull/22139
    symbols: Mapping[float, LiteralString] = ...

    @abc.abstractmethod
    @override
    def __str__(self, /) -> str: ...

    #
    @abc.abstractmethod
    def contains(self, /, x: onp.ArrayND[Any]) -> onp.ArrayND[np.bool_]: ...

    #
    @abc.abstractmethod
    def draw(self, /, n: int) -> onp.ArrayND[np.float64]: ...

    #
    @abc.abstractmethod
    def get_numerical_endpoints(
        self,
        /,
        x: Mapping[str, onp.ToFloat | onp.ToFloatND],
    ) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]: ...

# TODO(jorenham): Generic dtype
class _SimpleDomain(_Domain, metaclass=abc.ABCMeta):
    def __init__(self, /, endpoints: tuple[float, float] = ..., inclusive: tuple[bool, bool] = (False, False)) -> None: ...
    @override
    def __str__(self, /) -> str: ...  # noqa: PYI029

    #
    def define_parameters(self, /, *parameters: _Parameter) -> None: ...

    #
    @override
    def get_numerical_endpoints(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        parameter_values: Mapping[str, onp.ToFloat | onp.ToFloatND],
    ) -> tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]: ...

    #
    @override
    def contains(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        item: onp.ArrayND[np.integer[Any] | np.floating[Any]],
        parameter_values: Mapping[str, onp.ToFloat | onp.ToFloatND] | None = None,
    ) -> onp.ArrayND[np.bool_]: ...

class _RealDomain(_SimpleDomain):
    @override
    def draw(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        n: int,
        type_: L["in", "out", "on", "nan"],
        min: onp.ArrayND[np.floating[Any]],
        max: onp.ArrayND[np.floating[Any]],
        squeezed_base_shape: tuple[int, ...],
        rng: ToRNG = None,
    ) -> onp.ArrayND[np.float64]: ...

#
class _Parameter(abc.ABC):
    def __init__(
        self,
        /,
        name: str,
        *,
        domain: _Domain,
        symbol: str | None = None,
        typical: _Domain | tuple[int | str, int | str] | None = None,
    ) -> None: ...

    #
    def draw(
        self,
        /,
        size: tuple[int, ...] | None = None,
        *,
        rng: ToRNG = None,
        region: L["domain", "typical"] = "domain",
        proportions: tuple[onp.ToFloat, onp.ToFloat, onp.ToFloat, onp.ToFloat] | None = None,
        parameter_values: Mapping[str, onp.ToFloat | onp.ToFloatND] | None = None,
    ) -> onp.ArrayND[np.float64]: ...

    #
    @overload
    @abc.abstractmethod
    def validate(self, /, arr: onp.ToFloat) -> tuple[_FloatT, np.dtype[_FloatT], onp.Array0D[np.bool_]]: ...
    @overload
    @abc.abstractmethod
    def validate(self, /, arr: onp.ToFloatND) -> tuple[onp.ArrayND[_FloatT], np.dtype[_FloatT], onp.ArrayND[np.bool_]]: ...

class _RealParameter(_Parameter):
    @overload  # type: ignore[override]
    def validate(
        self,
        /,
        arr: onp.ToFloat,
        parameter_values: Mapping[str, onp.ToFloat | onp.ToFloatND],
    ) -> tuple[_FloatT, np.dtype[_FloatT], onp.Array0D[np.bool_]]: ...
    @overload
    def validate(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        arr: onp.ToFloatND,
        parameter_values: Mapping[str, onp.ToFloat | onp.ToFloatND],
    ) -> tuple[onp.ArrayND[_FloatT], np.dtype[_FloatT], onp.ArrayND[np.bool_]]: ...

class _Parameterization:
    parameters: Final[Mapping[str, _Parameter]]

    def __init__(self, /, *parameters: _Parameter) -> None: ...
    def __len__(self, /) -> int: ...
    def copy(self, /) -> _Parameterization: ...
    def matches(self, /, parameters: AbstractSet[str]) -> bool: ...
    def validation(
        self,
        /,
        parameter_values: Mapping[str, _Parameter],
    ) -> tuple[onp.ArrayND[np.bool_] | np.dtype[np.floating[Any]]]: ...
    def draw(
        self,
        /,
        sizes: tuple[int, ...] | Sequence[tuple[int, ...]] | None = None,
        rng: ToRNG = None,
        proportions: tuple[onp.ToFloat, onp.ToFloat, onp.ToFloat, onp.ToFloat] | None = None,
        region: L["domain", "typical"] = "domain",
    ) -> dict[str, onp.ArrayND[np.float64]]: ...

###

_XT = TypeVar("_XT", bound=np.inexact[Any])
_XT_co = TypeVar("_XT_co", bound=np.inexact[Any], default=np.float64, covariant=True)
_ShapeT0 = TypeVar("_ShapeT0", bound=tuple[int, ...])
_ShapeT0_co = TypeVar("_ShapeT0_co", bound=tuple[int, ...], default=tuple[int, ...], covariant=True)
_DistrT_co = TypeVar(
    "_DistrT_co",
    bound=ContinuousDistribution[np.inexact[Any]],
    default=ContinuousDistribution[np.inexact[Any]],
    covariant=True,
)

# placeholder for `matplotlib.axes.Axes`
_Axes: TypeAlias = object
_AxesT = TypeVar("_AxesT", bound=_Axes, default=Any)

_PlotQuantity: TypeAlias = L["x", "cdf", "ccdf", "icdf", "iccdf", "logcdf", "logccdf", "ilogcdf", "ilogccdf"]

_JustFloat: TypeAlias = opt.Just[float] | np.floating[Any]
_Null: TypeAlias = opt.Just[object]

_null: Final[_Null] = ...

class ContinuousDistribution(_BaseDistribution[_XT_co, _ShapeT0_co], Generic[_XT_co, _ShapeT0_co]):
    __array_priority__: ClassVar[float] = 1

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
    def __neg__(self, /) -> ShiftedScaledDistribution[Self, _XT_co, _ShapeT0_co]: ...
    def __abs__(self, /) -> FoldedDistribution[Self, _XT_co, _ShapeT0_co]: ...

    # TODO(jorenham): Accept `onp.ToFloatND`?
    def __add__(self, rshift: onp.ToFloat, /) -> ShiftedScaledDistribution[Self, _XT_co, _ShapeT0_co]: ...
    def __sub__(self, lshift: onp.ToFloat, /) -> ShiftedScaledDistribution[Self, _XT_co, _ShapeT0_co]: ...
    def __mul__(self, scale: onp.ToFloat, /) -> ShiftedScaledDistribution[Self, _XT_co, _ShapeT0_co]: ...
    def __truediv__(self, iscale: onp.ToFloat, /) -> ShiftedScaledDistribution[Self, _XT_co, _ShapeT0_co]: ...
    def __pow__(self, exp: onp.ToInt, /) -> MonotonicTransformedDistribution[Self, _XT_co, _ShapeT0_co]: ...
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__
    __rpow__ = __pow__

    #
    def reset_cache(self, /) -> None: ...

    #
    def plot(
        self,
        x: str = "x",
        y: str = "pdf",
        *,
        t: tuple[_PlotQuantity, _JustFloat, _JustFloat] = ("cdf", 0.0005, 0.9995),
        ax: _AxesT | None = None,
    ) -> _AxesT: ...

    #
    @overload
    def llf(self, sample: onp.ToFloat | onp.ToFloatND, /, *, axis: None) -> np.float64: ...
    @overload
    def llf(self, sample: onp.ToFloat | onp.ToFloatStrict1D, /, *, axis: AnyShape | None = -1) -> np.float64: ...
    @overload
    def llf(self, sample: onp.ToFloatStrict2D, /, *, axis: op.CanIndex | tuple[op.CanIndex] = -1) -> onp.Array1D[np.float64]: ...
    @overload
    def llf(self, sample: onp.ToFloatStrict2D, /, *, axis: tuple[op.CanIndex, op.CanIndex]) -> np.float64: ...
    @overload
    def llf(self, sample: onp.ToFloatStrict3D, /, *, axis: op.CanIndex | tuple[op.CanIndex] = -1) -> onp.Array2D[np.float64]: ...
    @overload
    def llf(self, sample: onp.ToFloatStrict3D, /, *, axis: tuple[op.CanIndex, op.CanIndex]) -> onp.Array1D[np.float64]: ...
    @overload
    def llf(self, sample: onp.ToFloatStrict3D, /, *, axis: tuple[op.CanIndex, op.CanIndex, op.CanIndex]) -> np.float64: ...
    @overload
    def llf(
        self,
        sample: onp.ToFloat | onp.ToFloatND,
        /,
        *,
        axis: AnyShape | None = -1,
    ) -> np.float64 | onp.ArrayND[np.float64]: ...

    #

class TransformedDistribution(ContinuousDistribution[_XT_co, _ShapeT0_co], Generic[_DistrT_co, _XT_co, _ShapeT0_co]):
    def __init__(
        self: TransformedDistribution[ContinuousDistribution[_XT, _ShapeT0], _XT, _ShapeT0],  # nice trick, eh?
        X: _DistrT_co,
        /,
        *args: Never,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...

class MonotonicTransformedDistribution(
    TransformedDistribution[_DistrT_co, _XT_co, _ShapeT0_co],
    Generic[_DistrT_co, _XT_co, _ShapeT0_co],
):
    # TODO(jorenham)
    ...

_DistrT0f = TypeVar("_DistrT0f", bound=ContinuousDistribution[np.floating[Any], tuple[()]])
_DistrT1f = TypeVar("_DistrT1f", bound=ContinuousDistribution[np.floating[Any], tuple[int]])
_DistrT2f = TypeVar("_DistrT2f", bound=ContinuousDistribution[np.floating[Any], tuple[int, int]])
_DistrT3f = TypeVar("_DistrT3f", bound=ContinuousDistribution[np.floating[Any], tuple[int, int, int]])
_DistrTNf = TypeVar("_DistrTNf", bound=ContinuousDistribution[np.floating[Any], tuple[int, ...]])

# still waiting on the intersection type PEP...
@overload
def truncate(
    X: _DistrT0f,
    lb: onp.ToFloat = ...,
    ub: onp.ToFloat = ...,
) -> TruncatedDistribution[_DistrT0f, np.floating[Any], tuple[()]]: ...
@overload
def truncate(
    X: _DistrT1f,
    lb: onp.ToFloat | onp.ToFloatStrict1D = ...,
    ub: onp.ToFloat | onp.ToFloatStrict1D = ...,
) -> TruncatedDistribution[_DistrT1f, np.floating[Any], tuple[int]]: ...
@overload
def truncate(
    X: _DistrT2f,
    lb: onp.ToFloat | onp.ToFloatStrict1D | onp.ToFloatStrict2D = ...,
    ub: onp.ToFloat | onp.ToFloatStrict1D | onp.ToFloatStrict2D = ...,
) -> TruncatedDistribution[_DistrT2f, np.floating[Any], tuple[int, int]]: ...
@overload
def truncate(
    X: _DistrT3f,
    lb: onp.ToFloat | onp.ToFloatStrict1D | onp.ToFloatStrict2D | onp.ToFloatStrict3D = ...,
    ub: onp.ToFloat | onp.ToFloatStrict1D | onp.ToFloatStrict2D | onp.ToFloatStrict3D = ...,
) -> TruncatedDistribution[_DistrT3f, np.floating[Any], tuple[int, int, int]]: ...
@overload
def truncate(
    X: _DistrTNf,
    lb: onp.ToFloat | onp.ToFloatND = ...,
    ub: onp.ToFloat | onp.ToFloatND = ...,
) -> TruncatedDistribution[_DistrTNf, np.floating[Any], tuple[int, ...]]: ...
@overload
def truncate(
    X: ContinuousDistribution[_XT, _ShapeT0],
    lb: onp.ToFloat = ...,
    ub: onp.ToFloat = ...,
) -> TruncatedDistribution[ContinuousDistribution[_XT, _ShapeT0], _XT, _ShapeT0]: ...
@overload
def truncate(
    X: ContinuousDistribution[_XT],
    lb: onp.ToFloat | onp.ToFloatND = ...,
    ub: onp.ToFloat | onp.ToFloatND = ...,
) -> TruncatedDistribution[ContinuousDistribution[_XT], _XT]: ...

class TruncatedDistribution(TransformedDistribution[_DistrT_co, _XT_co, _ShapeT0_co], Generic[_DistrT_co, _XT_co, _ShapeT0_co]):
    lb: _XT_co | onp.ArrayND[_XT_co, _ShapeT0_co]
    ub: _XT_co | onp.ArrayND[_XT_co, _ShapeT0_co]

    @overload
    def __init__(
        self: TruncatedDistribution[_DistrT0f, np.floating[Any], tuple[()]],
        X: _DistrT0f,
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
        self: TruncatedDistribution[_DistrT1f, np.floating[Any], tuple[int]],
        X: _DistrT1f,
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
        self: TruncatedDistribution[_DistrT2f, np.floating[Any], tuple[int, int]],
        X: _DistrT2f,
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
        self: TruncatedDistribution[_DistrT3f, np.floating[Any], tuple[int, int, int]],
        X: _DistrT3f,
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
        self: TruncatedDistribution[_DistrTNf, np.floating[Any], tuple[int, ...]],
        X: _DistrTNf,
        /,
        *args: Never,
        lb: onp.ToFloat | onp.ToFloatND = ...,
        ub: onp.ToFloat | onp.ToFloatND = ...,
        tol: opt.Just[float] | _Null = ...,
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...

class FoldedDistribution(
    TransformedDistribution[_DistrT_co, _XT_co, _ShapeT0_co],
    Generic[_DistrT_co, _XT_co, _ShapeT0_co],
):
    # TODO(jorenham)
    ...

class ShiftedScaledDistribution(
    TransformedDistribution[_DistrT_co, _XT_co, _ShapeT0_co],
    Generic[_DistrT_co, _XT_co, _ShapeT0_co],
):
    # TODO(jorenham)
    ...

class OrderStatisticDistribution(TransformedDistribution[_DistrT_co, np.float64, _ShapeT0_co], Generic[_DistrT_co, _ShapeT0_co]):
    # TODO(jorenham)
    ...

class Mixture(_BaseDistribution[_XT_co, tuple[()]], Generic[_XT_co]):
    _shape: tuple[()]
    _dtype: np.dtype[_XT_co]
    _components: Sequence[ContinuousDistribution[_XT_co, tuple[()]]]
    _weights: onp.Array1D[_XT_co]
    validation_policy: None

    @property
    def components(self, /) -> list[ContinuousDistribution[_XT_co, tuple[()]]]: ...
    @property
    def weights(self, /) -> onp.Array1D[_XT_co]: ...

    #
    def __init__(
        self,
        /,
        components: Sequence[ContinuousDistribution[_XT_co, tuple[()]]],
        *,
        weights: onp.ToFloat1D | None = None,
    ) -> None: ...

    #
    @override
    def kurtosis(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        *,
        method: L["formula", "general", "transform", "normalize", "cache"] | None = None,
    ) -> np.float64 | np.longdouble: ...
