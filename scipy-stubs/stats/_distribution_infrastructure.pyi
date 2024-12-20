# mypy: disable-error-code="explicit-override"

import abc
from collections.abc import Mapping, Sequence, Set as AbstractSet
from typing import Any, Final, Literal as L, TypeAlias, overload
from typing_extensions import LiteralString, TypeVar, override

import numpy as np
import optype.numpy as onp
import optype.typing as opt
from scipy._typing import ToRNG
from ._probability_distribution import _BaseDistribution

# TODO:
# `__all__ = ["Mixture", "abs", "exp", "log", "make_distribution", "order_statistic", "truncate"]

_ValidationPolicy: TypeAlias = L["skip_all"] | None
_CachePolicy: TypeAlias = L["no_cache"] | None

_FloatT = TypeVar("_FloatT", bound=np.floating[Any])

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

class ContinuousDistribution(_BaseDistribution):
    def __init__(
        self,
        /,
        *,
        tol: opt.Just[float],
        validation_policy: _ValidationPolicy = None,
        cache_policy: _CachePolicy = None,
    ) -> None: ...
    def reset_cache(self, /) -> None: ...

class TransformedDistribution(ContinuousDistribution):
    # TODO(jorenham)
    ...

class TruncatedDistribution(TransformedDistribution):
    # TODO(jorenham)
    ...

class ShiftedScaledDistribution(TransformedDistribution):
    # TODO(jorenham)
    ...

class OrderStatisticDistribution(TransformedDistribution):
    # TODO(jorenham)
    ...

class MonotonicTransformedDistribution(TransformedDistribution):
    # TODO(jorenham)
    ...

class FoldedDistribution(TransformedDistribution):
    # TODO(jorenham)
    ...

class Mixture(_BaseDistribution):
    # TODO(jorenham)
    ...
