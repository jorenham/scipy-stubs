# NOTE(scipy-stubs): This ia a module only exists `if typing.TYPE_CHECKING: ...`

import abc
from typing import Generic, Literal, TypeAlias, final, type_check_only
from typing_extensions import Self, TypeVarTuple, Unpack

__all__ = ("BaseBunch", "PowerDivergenceStatistic")

_Ts = TypeVarTuple("_Ts")

@type_check_only
class BaseBunch(tuple[Unpack[_Ts]], Generic[Unpack[_Ts]]):
    # A helper baseclass for annotating the return type of a *specific*
    # `scipy._lib.bunch._make_tuple_bunch` call.
    #
    # NOTE: Subtypes must implement:
    #
    # - `def __new__(_cls, {fields}, *, {extra_fields}) -> Self: ...`
    # - `def __init__(self, /, {fields}, *, {extra_fields}) -> None: ...`
    #
    # NOTE: The `_cls` parameter in `__new__` must be kept as-is, and shouldn't
    # be made positional only.
    #
    # NOTE: Each field in `{fields}` and `{extra_fields}` must be implemented as
    # a (read-only) `@property`
    # NOTE: The (variadic) generic type parameters coorespond to the types of
    # `{fields}`, **not** `{extra_fields}`

    @abc.abstractmethod
    def __new__(_cls) -> Self: ...
    @abc.abstractmethod
    def __init__(self, /) -> None: ...
    @final
    def __getnewargs_ex__(self, /) -> tuple[tuple[Unpack[_Ts]], dict[str, object]]: ...

    # NOTE: `._fields` and `._extra_fields` are mutually exclusive (disjoint)
    @property
    def _fields(self, /) -> tuple[str, ...]: ...
    @property
    def _extra_fields(self, /) -> tuple[str, ...]: ...

    # NOTE: `._asdict()` includes both `{fields}` and `{extra_fields}`
    def _asdict(self, /) -> dict[str, object]: ...

PowerDivergenceStatistic: TypeAlias = Literal[
    "pearson",
    "log-likelihood",
    "freeman-tukey",
    "mod-log-likelihood",
    "neyman",
    "cressie-read",
]
