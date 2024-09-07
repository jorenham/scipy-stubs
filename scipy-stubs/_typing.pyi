# Helper types for internal use (type-check only).
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, TypeAlias, type_check_only
from typing_extensions import LiteralString, TypeVar

import numpy as np
import optype as op
import optype.numpy as onpt

__all__ = [
    "RNG",
    "AnyBool",
    "AnyChar",
    "AnyComplex",
    "AnyInt",
    "AnyReal",
    "AnyScalar",
    "AnyShape",
    "Array0D",
    "CorrelateMode",
    "NanPolicy",
    "Seed",
    "Untyped",
    "UntypedArray",
    "UntypedCallable",
    "UntypedDict",
    "UntypedList",
    "UntypedTuple",
    "_FortranFunction",
]

# placeholders for missing annotations
Untyped: TypeAlias = object
UntypedTuple: TypeAlias = tuple[Untyped, ...]
UntypedList: TypeAlias = list[Untyped]
UntypedDict: TypeAlias = dict[Untyped, Untyped]
UntypedCallable: TypeAlias = Callable[..., Untyped]
UntypedArray: TypeAlias = onpt.Array[tuple[int, ...], np.generic]

_SCT = TypeVar("_SCT", bound=np.generic, default=np.generic)
Array0D: TypeAlias = np.ndarray[tuple[()], np.dtype[_SCT]]

# keep in sync with `numpy._typing._scalars`
AnyBool: TypeAlias = bool | np.bool_ | Literal[0, 1]
AnyInt: TypeAlias = int | np.integer[Any] | np.bool_
AnyReal: TypeAlias = int | float | np.floating[Any] | np.integer[Any] | np.bool_
AnyComplex: TypeAlias = int | float | complex | np.number[Any] | np.bool_
AnyChar: TypeAlias = str | bytes  # `np.str_ <: builtins.str` and `np.bytes_ <: builtins.bytes`
AnyScalar: TypeAlias = int | float | complex | AnyChar | np.generic

# equivalent to `numpy._typing._shape._ShapeLike`
AnyShape: TypeAlias = op.CanIndex | Sequence[op.CanIndex]

# numpy literals
RNG: TypeAlias = np.random.Generator | np.random.RandomState
Seed: TypeAlias = int | RNG
CorrelateMode: TypeAlias = Literal["valid", "same", "full"]

# scipy literals
NanPolicy: TypeAlias = Literal["raise", "propagate", "omit"]

# used in `scipy.linalg.blas` and `scipy.linalg.lapack`
@type_check_only
class _FortranFunction(Protocol):
    @property
    def dtype(self) -> np.dtype[np.number[Any]]: ...
    @property
    def int_dtype(self) -> np.dtype[np.integer[Any]]: ...
    @property
    def module_name(self) -> LiteralString: ...
    @property
    def prefix(self) -> LiteralString: ...
    @property
    def typecode(self) -> LiteralString: ...
    def __call__(self, /, *args: object, **kwargs: object) -> object: ...
