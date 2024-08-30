from collections.abc import Sequence
from typing import Any, Final, Generic, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt

__all__ = ["Covariance"]

# `float16` and `longdouble` aren't supported in `scipy.linalg`, and neither is `bool_`
_Scalar_uif: TypeAlias = np.float32 | np.float64 | np.integer[Any]
_ArrayLike_uif_1d: TypeAlias = (
    Sequence[float | _Scalar_uif]
    | onpt.CanArray[tuple[int], np.dtype[_Scalar_uif]]
    | npt.NDArray[_Scalar_uif]
)  # fmt: skip
_ArrayLike_uif_2d: TypeAlias = (
    Sequence[_ArrayLike_uif_1d]
    | onpt.CanArray[tuple[int, int], np.dtype[_Scalar_uif]]
    | npt.NDArray[_Scalar_uif]
)  # fmt: skip

_SCT = TypeVar("_SCT", bound=_Scalar_uif)
_SCT_co = TypeVar("_SCT_co", bound=_Scalar_uif, covariant=True, default=np.float64)

class Covariance(Generic[_SCT_co]):
    @staticmethod
    @overload
    def from_diagonal(diagonal: Sequence[int]) -> CovViaDiagonal[np.int_]: ...
    @staticmethod
    @overload
    def from_diagonal(diagonal: Sequence[float]) -> CovViaDiagonal[np.int_ | np.float64]: ...
    @staticmethod
    @overload
    def from_diagonal(diagonal: Sequence[_SCT] | onpt.CanArray[Any, np.dtype[_SCT]]) -> CovViaDiagonal[_SCT]: ...
    @staticmethod
    def from_precision(precision: _ArrayLike_uif_2d, covariance: _ArrayLike_uif_2d | None = None) -> CovViaPrecision: ...
    @staticmethod
    def from_cholesky(cholesky: _ArrayLike_uif_2d) -> CovViaCholesky: ...
    @staticmethod
    def from_eigendecomposition(eigendecomposition: tuple[_ArrayLike_uif_1d, _ArrayLike_uif_2d]) -> CovViaEigendecomposition: ...
    def whiten(self, /, x: onpt.AnyIntegerArray | onpt.AnyFloatingArray) -> npt.NDArray[np.floating[Any]]: ...
    def colorize(self, /, x: onpt.AnyIntegerArray | onpt.AnyFloatingArray) -> npt.NDArray[np.floating[Any]]: ...
    @property
    def log_pdet(self, /) -> np.float64: ...
    @property
    def rank(self, /) -> np.int_: ...
    @property
    def covariance(self, /) -> onpt.Array[tuple[int, int], _SCT_co]: ...
    @property
    def shape(self, /) -> tuple[int, int]: ...

class CovViaDiagonal(Covariance[_SCT_co], Generic[_SCT_co]):
    @overload
    def __init__(self: CovViaDiagonal[np.int_], /, diagonal: Sequence[int]) -> None: ...
    @overload
    def __init__(self: CovViaDiagonal[np.int_ | np.float64], /, diagonal: Sequence[float]) -> None: ...
    @overload
    def __init__(self, /, diagonal: Sequence[_SCT_co] | onpt.CanArray[Any, np.dtype[_Scalar_uif]]) -> None: ...

class CovViaPrecision(Covariance[np.float64]):
    def __init__(self, /, precision: _ArrayLike_uif_2d, covariance: _ArrayLike_uif_2d | None = None) -> None: ...

class CovViaCholesky(Covariance[np.float64]):
    def __init__(self, /, cholesky: _ArrayLike_uif_2d) -> None: ...

class CovViaEigendecomposition(Covariance[np.float64]):
    def __init__(self, /, eigendecomposition: tuple[_ArrayLike_uif_1d, _ArrayLike_uif_2d]) -> None: ...

@type_check_only
class _PSD(Protocol):
    _M: npt.NDArray[np.float64]
    V: npt.NDArray[np.float64]
    U: npt.NDArray[np.float64]
    eps: np.float64 | float
    log_pdet: np.float64 | float
    cond: np.float64 | float
    rank: int

    @property
    def pinv(self, /) -> npt.NDArray[np.floating[Any]]: ...

class CovViaPSD(Covariance[np.float64]):
    _LP: Final[npt.NDArray[np.float64]]
    _log_pdet: Final[np.float64 | float]
    _rank: Final[int]
    _covariance: Final[npt.NDArray[np.float64]]
    _shape: tuple[int, int]
    _psd: Final[_PSD]
    _allow_singular: Final = False

    def __init__(self, /, psd: _PSD) -> None: ...
