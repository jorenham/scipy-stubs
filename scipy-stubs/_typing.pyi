# NOTE(scipy-stubs): This ia a module only exists `if typing.TYPE_CHECKING: ...`

from os import PathLike
from collections.abc import Callable, Sequence
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
    "CorrelateMode",
    "EnterNoneMixin",
    "EnterSelfMixin",
    "FileLike",
    "FileModeRW",
    "FileModeRWA",
    "FileName",
    "NanPolicy",
    "OrderCF",
    "OrderKACF",
    "Seed",
    "Untyped",
    "UntypedArray",
    "UntypedCallable",
    "UntypedDict",
    "UntypedList",
    "UntypedTuple",
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

# placeholders for missing annotations
Untyped: TypeAlias = Any
UntypedTuple: TypeAlias = tuple[Untyped, ...]
UntypedList: TypeAlias = list[Untyped]
UntypedDict: TypeAlias = dict[Untyped, Untyped]
UntypedCallable: TypeAlias = Callable[..., Untyped]
UntypedArray: TypeAlias = onp.Array[Any, np.generic]

# I/O
_ByteSOrStr = TypeVar("_ByteSOrStr", bytes, str)
FileName: TypeAlias = str | PathLike[str]
FileLike: TypeAlias = IO[_ByteSOrStr] | FileName
FileModeRW: TypeAlias = Literal["r", "w"]
FileModeRWA: TypeAlias = Literal[FileModeRW, "a"]

# keep in sync with `numpy._typing._scalars`
AnyBool: TypeAlias = bool | np.bool_ | Literal[0, 1]

# equivalent to `numpy._typing._shape._ShapeLike`
AnyShape: TypeAlias = op.CanIndex | Sequence[op.CanIndex]

# numpy types
RNG: TypeAlias = np.random.BitGenerator | np.random.Generator | np.random.RandomState
Seed: TypeAlias = RNG | np.random.SeedSequence | onp.ToJustInt | onp.ToJustIntND
ByteOrder: TypeAlias = Literal["S", "<", "little", ">", "big", "=", "native", "|", "I"]
OrderCF: TypeAlias = Literal["C", "F"]
OrderKACF: TypeAlias = Literal["K", "A", "C", "F"]
Casting: TypeAlias = Literal["no", "equiv", "safe", "same_kind", "unsafe"]
CorrelateMode: TypeAlias = Literal["valid", "same", "full"]

# scipy literals
NanPolicy: TypeAlias = Literal["raise", "propagate", "omit"]
Alternative: TypeAlias = Literal["two-sided", "less", "greater"]
DCTType: TypeAlias = Literal[1, 2, 3, 4]
NormalizationMode: TypeAlias = Literal["backward", "ortho", "forward"]

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
