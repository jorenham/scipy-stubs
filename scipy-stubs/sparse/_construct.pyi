from typing import Any, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt
import optype.typing as opt
from scipy._typing import Untyped
from scipy.sparse import (
    bsr_array,
    bsr_matrix,
    coo_array,
    coo_matrix,
    csc_array,
    csc_matrix,
    csr_array,
    csr_matrix,
    dia_array,
    dia_matrix,
    dok_array,
    dok_matrix,
    lil_array,
    lil_matrix,
)
from ._typing import Scalar, SPFormat, ToDType, ToDTypeBool, ToDTypeComplex, ToDTypeFloat, ToDTypeInt

__all__ = [
    "block_array",
    "block_diag",
    "bmat",
    "diags",
    "diags_array",
    "eye",
    "eye_array",
    "hstack",
    "identity",
    "kron",
    "kronsum",
    "rand",
    "random",
    "random_array",
    "spdiags",
    "vstack",
]

_SCT = TypeVar("_SCT", bound=Scalar, default=Any)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int] | tuple[int, int], default=tuple[int] | tuple[int, int])

_SpMatrix: TypeAlias = (
    bsr_matrix[_SCT]
    | coo_matrix[_SCT]
    | csc_matrix[_SCT]
    | csr_matrix[_SCT]
    | dia_matrix[_SCT]
    | dok_matrix[_SCT]
    | lil_matrix[_SCT]
)

_SpArray: TypeAlias = (
    bsr_array[_SCT]
    | coo_array[_SCT, _ShapeT]
    | csc_array[_SCT]
    | csr_array[_SCT, _ShapeT]
    | dia_array[_SCT]
    | dok_array[_SCT, _ShapeT]
    | lil_array[_SCT]
)
_SpArray2D: TypeAlias = _SpArray[_SCT, tuple[int, int]]

###

#
def spdiags(
    data: Untyped,
    diags: Untyped,
    m: Untyped | None = None,
    n: Untyped | None = None,
    format: Untyped | None = None,
) -> Untyped: ...

#
def diags_array(
    diagonals: Untyped,
    /,
    *,
    offsets: int = 0,
    shape: Untyped | None = None,
    format: Untyped | None = None,
    dtype: Untyped | None = None,
) -> Untyped: ...

#
def diags(
    diagonals: Untyped,
    offsets: int = 0,
    shape: Untyped | None = None,
    format: Untyped | None = None,
    dtype: Untyped | None = None,
) -> Untyped: ...

#
def identity(n: Untyped, dtype: str = "d", format: Untyped | None = None) -> Untyped: ...

#
@overload  # dtype like bool, format: None = ...
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeBool,
    format: None = None,
) -> dia_array[np.bool_]: ...
@overload  # dtype like int, format: None = ...
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeInt,
    format: None = None,
) -> dia_array[np.int_]: ...
@overload  # dtype like float (default), format: None = ...
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeFloat = ...,
    format: None = None,
) -> dia_array[np.float64]: ...
@overload  # dtype like complex, format: None = ...
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeComplex,
    format: None = None,
) -> dia_array[np.complex128]: ...
@overload  # dtype like <known>, format: None = ...
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDType[_SCT],
    format: None = None,
) -> dia_array[_SCT]: ...
@overload  # dtype like <unknown>, format: None = ...
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: npt.DTypeLike,
    format: None = None,
) -> dia_array: ...
@overload  # dtype like float (default), format: <given>
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeFloat = ...,
    format: SPFormat,
) -> _SpArray2D[np.float64]: ...
@overload  # dtype like bool, format: <given>
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeBool,
    format: SPFormat,
) -> _SpArray2D[np.bool_]: ...
@overload  # dtype like int, format: <given>
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeInt,
    format: SPFormat,
) -> _SpArray2D[np.int_]: ...
@overload  # dtype like complex, format: <given>
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeComplex,
    format: SPFormat,
) -> _SpArray2D[np.complex128]: ...
@overload  # dtype like <known>, format: <given>
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDType[_SCT],
    format: SPFormat,
) -> _SpArray2D[_SCT]: ...
@overload  # dtype like <unknown>, fformat: <given>
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: npt.DTypeLike,
    format: SPFormat,
) -> _SpArray2D: ...

#
@overload  # dtype like float (default)
def eye(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    k: int = 0,
    dtype: ToDTypeFloat = ...,
    format: SPFormat | None = None,
) -> _SpMatrix[np.float64]: ...
@overload  # dtype like bool (positional)
def eye(
    m: opt.AnyInt,
    n: opt.AnyInt | None,
    k: int,
    dtype: ToDTypeBool,
    format: SPFormat | None = None,
) -> _SpMatrix[np.bool_]: ...
@overload  # dtype like bool (keyword)
def eye(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    k: int = 0,
    *,
    dtype: ToDTypeBool,
    format: SPFormat | None = None,
) -> _SpMatrix[np.bool_]: ...
@overload  # dtype like int (positional)
def eye(
    m: opt.AnyInt,
    n: opt.AnyInt | None,
    k: int,
    dtype: ToDTypeInt,
    format: SPFormat | None = None,
) -> _SpMatrix[np.int_]: ...
@overload  # dtype like int (keyword)
def eye(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    k: int = 0,
    *,
    dtype: ToDTypeInt,
    format: SPFormat | None = None,
) -> _SpMatrix[np.int_]: ...
@overload  # dtype like complex (positional)
def eye(
    m: opt.AnyInt,
    n: opt.AnyInt | None,
    k: int,
    dtype: ToDTypeComplex,
    format: SPFormat | None = None,
) -> _SpMatrix[np.complex128]: ...
@overload  # dtype like complex (keyword)
def eye(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    k: int = 0,
    *,
    dtype: ToDTypeComplex,
    format: SPFormat | None = None,
) -> _SpMatrix[np.complex128]: ...
@overload  # dtype like <known> (positional)
def eye(
    m: opt.AnyInt,
    n: opt.AnyInt | None,
    k: int,
    dtype: ToDType[_SCT],
    format: SPFormat | None = None,
) -> _SpMatrix[_SCT]: ...
@overload  # dtype like <known> (keyword)
def eye(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    k: int = 0,
    *,
    dtype: ToDType[_SCT],
    format: SPFormat | None = None,
) -> _SpMatrix[_SCT]: ...
@overload  # dtype like <unknown> (positional)
def eye(
    m: opt.AnyInt,
    n: opt.AnyInt | None,
    k: int,
    dtype: npt.DTypeLike,
    format: SPFormat | None = None,
) -> _SpMatrix: ...
@overload  # dtype like <unknown> (keyword)
def eye(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    k: int = 0,
    *,
    dtype: npt.DTypeLike,
    format: SPFormat | None = None,
) -> _SpMatrix: ...

#
def kron(A: Untyped, B: Untyped, format: Untyped | None = None) -> Untyped: ...

#
def kronsum(A: Untyped, B: Untyped, format: Untyped | None = None) -> Untyped: ...

#
def hstack(blocks: Untyped, format: Untyped | None = None, dtype: Untyped | None = None) -> Untyped: ...

#
def vstack(blocks: Untyped, format: Untyped | None = None, dtype: Untyped | None = None) -> Untyped: ...

#
def bmat(blocks: Untyped, format: Untyped | None = None, dtype: Untyped | None = None) -> Untyped: ...

#
def block_array(blocks: Untyped, *, format: Untyped | None = None, dtype: Untyped | None = None) -> Untyped: ...

#
def block_diag(mats: Untyped, format: Untyped | None = None, dtype: Untyped | None = None) -> Untyped: ...

#
def random_array(
    shape: Untyped,
    *,
    density: float = 0.01,
    format: str = "coo",
    dtype: Untyped | None = None,
    random_state: Untyped | None = None,
    data_sampler: Untyped | None = None,
) -> Untyped: ...

#
def random(
    m: Untyped,
    n: Untyped,
    density: float = 0.01,
    format: str = "coo",
    dtype: Untyped | None = None,
    random_state: Untyped | None = None,
    data_rvs: Untyped | None = None,
) -> Untyped: ...

#
def rand(
    m: Untyped,
    n: Untyped,
    density: float = 0.01,
    format: str = "coo",
    dtype: Untyped | None = None,
    random_state: Untyped | None = None,
) -> Untyped: ...
