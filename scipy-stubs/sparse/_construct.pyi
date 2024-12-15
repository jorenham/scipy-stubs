from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt
import optype.typing as opt
from scipy._typing import Seed, Untyped, UntypedCallable
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
from ._typing import Float, Scalar, SPFormat, ToDType, ToDTypeBool, ToDTypeComplex, ToDTypeFloat, ToDTypeInt, ToShape, ToShape2D

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
_SpArray1D: TypeAlias = coo_array[_SCT, tuple[int]] | csr_array[_SCT, tuple[int]] | dok_array[_SCT, tuple[int]]
_SpArray2D: TypeAlias = _SpArray[_SCT, tuple[int, int]]

_FmtDIA: TypeAlias = Literal["dia"]
_FmtNonDIA: TypeAlias = Literal["bsr", "coo", "csc", "csr", "dok", "lil"]

###

#
def spdiags(
    data: Untyped,
    diags: Untyped,
    m: Untyped | None = None,
    n: Untyped | None = None,
    format: SPFormat | None = None,
) -> Untyped: ...

#
def diags_array(
    diagonals: Untyped,
    /,
    *,
    offsets: int = 0,
    shape: ToShape | None = None,
    format: SPFormat | None = None,
    dtype: npt.DTypeLike | None = None,
) -> _SpArray1D | _SpArray2D: ...

#
def diags(
    diagonals: Untyped,
    offsets: int = 0,
    shape: ToShape2D | None = None,
    format: SPFormat | None = None,
    dtype: npt.DTypeLike | None = None,
) -> _SpMatrix: ...

#
@overload  # dtype like bool, format: None = ...
def identity(n: opt.AnyInt, dtype: ToDTypeBool, format: _FmtDIA | None = None) -> dia_matrix[np.bool_]: ...
@overload  # dtype like int, format: None = ...
def identity(n: opt.AnyInt, dtype: ToDTypeInt, format: _FmtDIA | None = None) -> dia_matrix[np.int_]: ...
@overload  # dtype like float (default), format: None = ...
def identity(n: opt.AnyInt, dtype: ToDTypeFloat = "d", format: _FmtDIA | None = None) -> dia_matrix[np.float64]: ...
@overload  # dtype like complex, format: None = ...
def identity(n: opt.AnyInt, dtype: ToDTypeComplex, format: _FmtDIA | None = None) -> dia_matrix[np.complex128]: ...
@overload  # dtype like <known>, format: None = ...
def identity(n: opt.AnyInt, dtype: ToDType[_SCT], format: _FmtDIA | None = None) -> dia_matrix[_SCT]: ...
@overload  # dtype like <unknown>, format: None = ...
def identity(n: opt.AnyInt, dtype: npt.DTypeLike, format: _FmtDIA | None = None) -> dia_matrix: ...
@overload  # dtype like float, format: <given> (positional)
def identity(n: opt.AnyInt, dtype: ToDTypeFloat, format: _FmtNonDIA) -> _SpMatrix[np.float64]: ...
@overload  # dtype like float (default), format: <given> (keyword)
def identity(n: opt.AnyInt, dtype: ToDTypeFloat = "d", *, format: _FmtNonDIA) -> _SpMatrix[np.float64]: ...
@overload  # dtype like bool, format: <given>
def identity(n: opt.AnyInt, dtype: ToDTypeBool, format: _FmtNonDIA) -> _SpMatrix[np.bool_]: ...
@overload  # dtype like int, format: <given>
def identity(n: opt.AnyInt, dtype: ToDTypeInt, format: _FmtNonDIA) -> _SpMatrix[np.int_]: ...
@overload  # dtype like complex, format: <given>
def identity(n: opt.AnyInt, dtype: ToDTypeComplex, format: _FmtNonDIA) -> _SpMatrix[np.complex128]: ...
@overload  # dtype like <known>, format: <given>
def identity(n: opt.AnyInt, dtype: ToDType[_SCT], format: _FmtNonDIA) -> _SpMatrix[_SCT]: ...
@overload  # dtype like <unknown>, fformat: <given>
def identity(n: opt.AnyInt, dtype: npt.DTypeLike, format: _FmtNonDIA) -> _SpMatrix: ...

#
@overload  # dtype like bool, format: None = ...
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeBool,
    format: _FmtDIA | None = None,
) -> dia_array[np.bool_]: ...
@overload  # dtype like int, format: None = ...
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeInt,
    format: _FmtDIA | None = None,
) -> dia_array[np.int_]: ...
@overload  # dtype like float (default), format: None = ...
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeFloat = ...,
    format: _FmtDIA | None = None,
) -> dia_array[np.float64]: ...
@overload  # dtype like complex, format: None = ...
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeComplex,
    format: _FmtDIA | None = None,
) -> dia_array[np.complex128]: ...
@overload  # dtype like <known>, format: None = ...
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDType[_SCT],
    format: _FmtDIA | None = None,
) -> dia_array[_SCT]: ...
@overload  # dtype like <unknown>, format: None = ...
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: npt.DTypeLike,
    format: _FmtDIA | None = None,
) -> dia_array: ...
@overload  # dtype like float (default), format: <given>
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeFloat = ...,
    format: _FmtNonDIA,
) -> _SpArray2D[np.float64]: ...
@overload  # dtype like bool, format: <given>
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeBool,
    format: _FmtNonDIA,
) -> _SpArray2D[np.bool_]: ...
@overload  # dtype like int, format: <given>
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeInt,
    format: _FmtNonDIA,
) -> _SpArray2D[np.int_]: ...
@overload  # dtype like complex, format: <given>
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDTypeComplex,
    format: _FmtNonDIA,
) -> _SpArray2D[np.complex128]: ...
@overload  # dtype like <known>, format: <given>
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: ToDType[_SCT],
    format: _FmtNonDIA,
) -> _SpArray2D[_SCT]: ...
@overload  # dtype like <unknown>, fformat: <given>
def eye_array(
    m: opt.AnyInt,
    n: opt.AnyInt | None = None,
    *,
    k: int = 0,
    dtype: npt.DTypeLike,
    format: _FmtNonDIA,
) -> _SpArray2D: ...

# NOTE: `eye_array` should be prefered over `eye`
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
def kron(A: Untyped, B: Untyped, format: SPFormat | None = None) -> _SpArray2D | _SpMatrix: ...
def kronsum(A: Untyped, B: Untyped, format: SPFormat | None = None) -> _SpArray2D | _SpMatrix: ...

#
def hstack(blocks: Untyped, format: SPFormat | None = None, dtype: npt.DTypeLike | None = None) -> _SpArray | _SpMatrix: ...
def vstack(blocks: Untyped, format: SPFormat | None = None, dtype: npt.DTypeLike | None = None) -> _SpArray | _SpMatrix: ...

#
def bmat(blocks: Untyped, format: SPFormat | None = None, dtype: npt.DTypeLike | None = None) -> _SpArray | _SpMatrix: ...
def block_array(blocks: Untyped, *, format: SPFormat | None = None, dtype: npt.DTypeLike | None = None) -> _SpArray: ...
def block_diag(mats: Untyped, format: SPFormat | None = None, dtype: npt.DTypeLike | None = None) -> _SpArray | _SpMatrix: ...

#
def random_array(
    shape: ToShape,
    *,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: npt.DTypeLike | None = None,
    random_state: Seed | None = None,
    data_sampler: UntypedCallable | None = None,
) -> _SpArray1D | _SpArray2D: ...

# NOTE: `random_array` should be prefered over `random`
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: npt.DTypeLike | None = None,
    random_state: Seed | None = None,
    data_rvs: UntypedCallable | None = None,
) -> _SpMatrix: ...

# NOTE: `random_array` should be prefered over `rand`
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: npt.DTypeLike | None = None,
    random_state: Seed | None = None,
) -> _SpMatrix: ...
