# Helper types for internal use (type-check only).
from collections.abc import Callable
from typing import Literal, Protocol, TypeAlias, type_check_only
from typing_extensions import LiteralString, TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt

__all__ = [
    "AnyBool",
    "AnyChar",
    "AnyComplex",
    "AnyInt",
    "AnyReal",
    "AnyScalar",
    "Array0D",
    "CorrelateMode",
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
AnyInt: TypeAlias = int | np.integer[npt.NBitBase] | np.bool_
AnyReal: TypeAlias = int | float | np.floating[npt.NBitBase] | np.integer[npt.NBitBase] | np.bool_
AnyComplex: TypeAlias = int | float | complex | np.number[npt.NBitBase] | np.bool_
AnyChar: TypeAlias = str | bytes  # `np.str_ <: builtins.str` and `np.bytes_ <: builtins.bytes`
AnyScalar: TypeAlias = int | float | complex | AnyChar | np.generic

# numpy literals
Seed: TypeAlias = int | np.random.Generator | np.random.RandomState
CorrelateMode: TypeAlias = Literal["valid", "same", "full"]

# used in `scipy.linalg.blas` and `scipy.linalg.lapack`
@type_check_only
class _FortranFunction(Protocol):
    @property
    def dtype(self) -> np.dtype[np.number[npt.NBitBase]]: ...
    @property
    def int_dtype(self) -> np.dtype[np.integer[npt.NBitBase]]: ...
    @property
    def module_name(self) -> LiteralString: ...
    @property
    def prefix(self) -> LiteralString: ...
    @property
    def typecode(self) -> LiteralString: ...
    def __call__(self, /, *args: object, **kwargs: object) -> object: ...
