from collections.abc import Sequence
from typing import Generic, overload
from typing_extensions import Self, TypeVar, override

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLike
from scipy.sparse import coo_matrix
from ._typing import BaseBunch

_SCT = TypeVar("_SCT", bound=np.generic, default=np.generic)

class CrosstabResult(BaseBunch[_SCT], Generic[_SCT]):
    @property
    def elements(self, /) -> tuple[npt.NDArray[_SCT], ...]: ...
    @property
    @override
    def count(self, /) -> onpt.Array[tuple[int, int], np.intp] | coo_matrix: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def __new__(
        _cls,
        elements: tuple[npt.NDArray[_SCT], ...],
        count: onpt.Array[tuple[int, int], np.intp] | coo_matrix,
    ) -> Self: ...
    def __init__(
        self,
        /,
        elements: tuple[npt.NDArray[_SCT], ...],
        count: onpt.Array[tuple[int, int], np.intp] | coo_matrix,
    ) -> None: ...

@overload
def crosstab(
    *args: _ArrayLike[_SCT],
    levels: _ArrayLike[_SCT] | None = None,
    sparse: bool = False,
) -> CrosstabResult[_SCT]: ...
@overload
def crosstab(
    *args: Sequence[bool],
    levels: Sequence[bool] | None = None,
    sparse: bool = False,
) -> CrosstabResult[np.bool_]: ...
@overload
def crosstab(
    *args: Sequence[int],
    levels: Sequence[int] | None = None,
    sparse: bool = False,
) -> CrosstabResult[np.int_] | CrosstabResult[np.bool_]: ...
@overload
def crosstab(
    *args: Sequence[float],
    levels: Sequence[float] | None = None,
    sparse: bool = False,
) -> CrosstabResult[np.float64] | CrosstabResult[np.int_] | CrosstabResult[np.bool_]: ...
@overload
def crosstab(
    *args: Sequence[complex],
    levels: Sequence[complex] | None = None,
    sparse: bool = False,
) -> CrosstabResult[np.complex128] | CrosstabResult[np.float64] | CrosstabResult[np.int_] | CrosstabResult[np.bool_]: ...
@overload
def crosstab(
    *args: Sequence[bytes],
    levels: Sequence[bytes] | None = None,
    sparse: bool = False,
) -> CrosstabResult[np.bytes_]: ...
@overload
def crosstab(
    *args: Sequence[str],
    levels: Sequence[str] | None = None,
    sparse: bool = False,
) -> CrosstabResult[np.str_]: ...
