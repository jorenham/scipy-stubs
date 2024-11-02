from typing import Any, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co
from scipy._typing import AnyReal

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])
_DTypeLikeFloat_co: TypeAlias = np.dtype[np.floating[Any] | np.integer[Any] | np.bool_]

@overload
def tukeylambda_variance(lam: AnyReal) -> onpt.Array[tuple[()], np.float64]: ...
@overload
def tukeylambda_variance(lam: onpt.CanArray[_ShapeT, _DTypeLikeFloat_co]) -> onpt.Array[_ShapeT, np.float64]: ...
@overload
def tukeylambda_variance(lam: _ArrayLikeFloat_co) -> npt.NDArray[np.float64]: ...

#
@overload
def tukeylambda_kurtosis(lam: AnyReal) -> onpt.Array[tuple[()], np.float64]: ...
@overload
def tukeylambda_kurtosis(lam: onpt.CanArray[_ShapeT, _DTypeLikeFloat_co]) -> onpt.Array[_ShapeT, np.float64]: ...
@overload
def tukeylambda_kurtosis(lam: _ArrayLikeFloat_co) -> npt.NDArray[np.float64]: ...
