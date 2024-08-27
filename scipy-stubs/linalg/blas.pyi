from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol, type_check_only

import numpy as np
import numpy.typing as npt
from typing_extensions import LiteralString

__all__ = ["find_best_blas_type", "get_blas_funcs"]

@type_check_only
class _FortranFunction(Protocol):
    @property
    def dtype(self) -> np.dtype[np.number[Any]]: ...  # type: ignore[no-any-explicit]
    @property
    def int_dtype(self) -> np.dtype[np.integer[Any]]: ...  # type: ignore[no-any-explicit]
    @property
    def module_name(self) -> LiteralString: ...
    @property
    def prefix(self) -> LiteralString: ...
    @property
    def typecode(self) -> LiteralString: ...

    def __call__(self, /, *args: object, **kwargs: object) -> object: ...

def find_best_blas_type(
    arrays: Sequence[npt.NDArray[np.generic]] = (),
    dtype: npt.DTypeLike | None = None,
) -> (
    # see `scipy.linalg.blas._type_conv`
    tuple[Literal["s"], np.dtypes.Float32DType, bool]
    | tuple[Literal["f"], np.dtypes.Float64DType, bool]
    | tuple[Literal["c"], np.dtypes.Complex64DType, bool]
    | tuple[Literal["z"], np.dtypes.Complex128DType, bool]
): ...
def get_blas_funcs(
    names: Iterable[str] | str,
    arrays: Sequence[npt.NDArray[np.generic]] = (),
    dtype: npt.DTypeLike | None = None,
    ilp64: Literal[True, False, "preferred"] = False,
) -> list[_FortranFunction] | _FortranFunction: ...
