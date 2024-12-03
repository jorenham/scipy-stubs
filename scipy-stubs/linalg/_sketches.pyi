from collections.abc import Sequence
from typing import Any, TypeAlias, TypeVar, overload

import numpy as np
import optype.numpy as onp
import optype.typing as opt
import scipy._typing as spt
from scipy.sparse import csc_matrix

__all__ = ["clarkson_woodruff_transform"]

_ST = TypeVar("_ST", bound=np.generic)
_VT = TypeVar("_VT")
_ToJust2D: TypeAlias = onp.CanArrayND[_ST] | Sequence[onp.CanArrayND[_ST]] | Sequence[Sequence[opt.Just[_VT] | _ST]]

###

def cwt_matrix(n_rows: onp.ToInt, n_columns: onp.ToInt, seed: spt.Seed | None = None) -> csc_matrix: ...

#
@overload
def clarkson_woodruff_transform(
    input_matrix: onp.ToInt2D,
    sketch_size: onp.ToInt,
    seed: spt.Seed | None = None,
) -> onp.Array2D[np.int_]: ...
@overload
def clarkson_woodruff_transform(
    input_matrix: _ToJust2D[np.floating[Any], float],
    sketch_size: onp.ToInt,
    seed: spt.Seed | None = None,
) -> onp.Array2D[np.float64 | np.longdouble]: ...
@overload
def clarkson_woodruff_transform(
    input_matrix: _ToJust2D[np.complexfloating[Any, Any], complex],
    sketch_size: onp.ToInt,
    seed: spt.Seed | None = None,
) -> onp.Array2D[np.complex64 | np.clongdouble]: ...
