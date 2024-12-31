# NOTE: This private(!) module only exists in `if typing.TYPE_CHECKING: ...` and in `.pyi` stubs

from os import PathLike
from collections.abc import Sequence
from types import TracebackType
from typing import IO, Any, Literal, Protocol, TypeAlias, type_check_only
from typing_extensions import LiteralString, Self, TypeVar

import numpy as np
import optype as op
import optype.numpy as onp

__all__ = [
    "RNG",
    "Alternative",
    "AnyBool",
    "AnyShape",
    "ByteOrder",
    "Casting",
    "ConvMode",
    "EnterNoneMixin",
    "EnterSelfMixin",
    "Falsy",
    "FileLike",
    "FileModeRW",
    "FileModeRWA",
    "FileName",
    "NanPolicy",
    "OrderCF",
    "OrderKACF",
    "ToRNG",
    "Truthy",
    "_FortranFunction",
]

# helper mixins
@type_check_only
class EnterSelfMixin:
    def __enter__(self, /) -> Self: ...
    def __exit__(self, /, type: type[BaseException] | None, value: BaseException | None, tb: TracebackType | None) -> None: ...

@type_check_only
class EnterNoneMixin:
    def __enter__(self, /) -> None: ...
    def __exit__(self, /, type: type[BaseException] | None, value: BaseException | None, tb: TracebackType | None) -> None: ...

# used in `scipy.linalg.blas` and `scipy.linalg.lapack`
@type_check_only
class _FortranFunction(Protocol):
    @property
    def dtype(self, /) -> np.dtype[np.number[Any]]: ...
    @property
    def int_dtype(self, /) -> np.dtype[np.integer[Any]]: ...
    @property
    def module_name(self, /) -> LiteralString: ...
    @property
    def prefix(self, /) -> LiteralString: ...
    @property
    def typecode(self, /) -> LiteralString: ...
    def __call__(self, /, *args: object, **kwargs: object) -> object: ...

# I/O
_ByteSOrStr = TypeVar("_ByteSOrStr", bytes, str)
FileName: TypeAlias = str | PathLike[str]
FileLike: TypeAlias = FileName | IO[_ByteSOrStr]
FileModeRW: TypeAlias = Literal["r", "w"]
FileModeRWA: TypeAlias = Literal[FileModeRW, "a"]

# TODO(jorenham): Include `np.bool[L[False]]` once we have `numpy>=2.2`
Falsy: TypeAlias = Literal[False, 0]
Truthy: TypeAlias = Literal[True, 1]

# keep in sync with `numpy._typing._scalars`
AnyBool: TypeAlias = bool | np.bool_ | Literal[0, 1]

# equivalent to `numpy._typing._shape._ShapeLike`
AnyShape: TypeAlias = op.CanIndex | Sequence[op.CanIndex]

RNG: TypeAlias = np.random.Generator | np.random.RandomState
# NOTE: This is less incorrect and more accurate than the current `np.random.default_rng` `seed` param annotation.
ToRNG: TypeAlias = (
    int
    | np.integer[Any]
    | np.timedelta64
    | onp.ArrayND[np.integer[Any] | np.timedelta64 | np.flexible | np.object_]
    | np.random.SeedSequence
    | np.random.BitGenerator
    | RNG
    | None
)

# numpy literals
ByteOrder: TypeAlias = Literal["S", "<", "little", ">", "big", "=", "native", "|", "I"]
OrderCF: TypeAlias = Literal["C", "F"]
OrderKACF: TypeAlias = Literal["K", "A", OrderCF]
Casting: TypeAlias = Literal["no", "equiv", "safe", "same_kind", "unsafe"]
ConvMode: TypeAlias = Literal["valid", "same", "full"]

# scipy literals
NanPolicy: TypeAlias = Literal["raise", "propagate", "omit"]
Alternative: TypeAlias = Literal["two-sided", "less", "greater"]
NormalizationMode: TypeAlias = Literal["backward", "ortho", "forward"]
DCTType: TypeAlias = Literal[1, 2, 3, 4]
