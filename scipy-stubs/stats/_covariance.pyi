from collections.abc import Sequence
from typing import Final, Generic, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

__all__ = ["Covariance"]

# `float16` and `longdouble` aren't supported in `scipy.linalg`, and neither is `bool_`
_Scalar_uif: TypeAlias = np.float32 | np.float64 | npc.integer

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
    def from_diagonal(diagonal: Sequence[_SCT] | onp.CanArrayND[_SCT]) -> CovViaDiagonal[_SCT]: ...
    @staticmethod
    def from_precision(precision: onp.ToFloat2D, covariance: onp.ToFloat2D | None = None) -> CovViaPrecision: ...
    @staticmethod
    def from_cholesky(cholesky: onp.ToFloat2D) -> CovViaCholesky: ...
    @staticmethod
    def from_eigendecomposition(eigendecomposition: tuple[onp.ToFloat1D, onp.ToFloat2D]) -> CovViaEigendecomposition: ...
    def whiten(self, /, x: onp.AnyIntegerArray | onp.AnyFloatingArray) -> onp.ArrayND[npc.floating]: ...
    def colorize(self, /, x: onp.AnyIntegerArray | onp.AnyFloatingArray) -> onp.ArrayND[npc.floating]: ...
    @property
    def log_pdet(self, /) -> np.float64: ...
    @property
    def rank(self, /) -> np.int_: ...
    @property
    def covariance(self, /) -> onp.Array2D[_SCT_co]: ...
    @property
    def shape(self, /) -> tuple[int, int]: ...

class CovViaDiagonal(Covariance[_SCT_co], Generic[_SCT_co]):
    @overload
    def __init__(self: CovViaDiagonal[np.int_], /, diagonal: Sequence[int]) -> None: ...
    @overload
    def __init__(self: CovViaDiagonal[np.int_ | np.float64], /, diagonal: Sequence[float]) -> None: ...
    @overload
    def __init__(self, /, diagonal: Sequence[float | _SCT_co] | onp.CanArrayND[_SCT_co]) -> None: ...

class CovViaPrecision(Covariance[np.float64]):
    def __init__(self, /, precision: onp.ToFloat2D, covariance: onp.ToFloat2D | None = None) -> None: ...

class CovViaCholesky(Covariance[np.float64]):
    def __init__(self, /, cholesky: onp.ToFloat2D) -> None: ...

class CovViaEigendecomposition(Covariance[np.float64]):
    def __init__(self, /, eigendecomposition: tuple[onp.ToFloat1D, onp.ToFloat2D]) -> None: ...

@type_check_only
class _PSD(Protocol):
    _M: onp.ArrayND[np.float64]
    V: onp.ArrayND[np.float64]
    U: onp.ArrayND[np.float64]
    eps: np.float64 | float
    log_pdet: np.float64 | float
    cond: np.float64 | float
    rank: int

    @property
    def pinv(self, /) -> onp.ArrayND[npc.floating]: ...

class CovViaPSD(Covariance[np.float64]):
    _LP: Final[onp.ArrayND[np.float64]]
    _log_pdet: Final[np.float64 | float]
    _rank: Final[int]
    _covariance: Final[onp.ArrayND[np.float64]]
    _shape: tuple[int, int]
    _psd: Final[_PSD]
    _allow_singular: Final = False

    def __init__(self, /, psd: _PSD) -> None: ...
