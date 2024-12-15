from collections.abc import Callable, Iterable, Sequence as Seq
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, overload, type_check_only

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import optype.typing as opt
from scipy._typing import Seed
from ._base import _spbase, sparray
from ._bsr import bsr_array, bsr_matrix
from ._coo import coo_array, coo_matrix
from ._csc import csc_array, csc_matrix
from ._csr import csr_array, csr_matrix
from ._dia import dia_array, dia_matrix
from ._dok import dok_array, dok_matrix
from ._lil import lil_array, lil_matrix
from ._matrix import spmatrix
from ._typing import Float, Scalar, SPFormat, ToDType, ToDTypeBool, ToDTypeComplex, ToDTypeFloat, ToDTypeInt, ToShape2D

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
_SCT1 = TypeVar("_SCT1", bound=Scalar)
_SCT2 = TypeVar("_SCT2", bound=Scalar)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int] | tuple[int, int], default=tuple[int] | tuple[int, int])

_ToArray1D: TypeAlias = Seq[_SCT] | onp.CanArrayND[_SCT]
_ToArray2D: TypeAlias = Seq[Seq[_SCT] | onp.CanArrayND[_SCT]] | onp.CanArrayND[_SCT]
_ToSpMatrix: TypeAlias = spmatrix[_SCT] | _ToArray2D[_SCT]

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

_BSRArray: TypeAlias = bsr_array[_SCT]
_CSRArray: TypeAlias = csr_array[_SCT, tuple[int, int]]
_NonBSRArray: TypeAlias = (
    coo_array[_SCT, tuple[int, int]]
    | csc_array[_SCT]
    | csr_array[_SCT, tuple[int, int]]
    | dia_array[_SCT]
    | dok_array[_SCT, tuple[int, int]]
    | lil_array[_SCT]
)
_NonCOOArray: TypeAlias = (
    bsr_array[_SCT]
    | csc_array[_SCT]
    | csr_array[_SCT, tuple[int, int]]
    | dia_array[_SCT]
    | dok_array[_SCT, tuple[int, int]]
    | lil_array[_SCT]
)
_NonCSRArray: TypeAlias = (
    bsr_array[_SCT]
    | coo_array[_SCT, tuple[int, int]]
    | csc_array[_SCT]
    | dia_array[_SCT]
    | dok_array[_SCT, tuple[int, int]]
    | lil_array[_SCT]
)
_NonDIAArray: TypeAlias = (
    bsr_array[_SCT]
    | coo_array[_SCT, tuple[int, int]]
    | csc_array[_SCT]
    | csr_array[_SCT, tuple[int, int]]
    | dok_array[_SCT, tuple[int, int]]
    | lil_array[_SCT]
)
_NonBSRMatrix: TypeAlias = (
    coo_matrix[_SCT] | csr_matrix[_SCT] | csc_matrix[_SCT] | dia_matrix[_SCT] | dok_matrix[_SCT] | lil_matrix[_SCT]
)
_NonCOOMatrix: TypeAlias = (
    bsr_matrix[_SCT] | csc_matrix[_SCT] | csr_matrix[_SCT] | dia_matrix[_SCT] | dok_matrix[_SCT] | lil_matrix[_SCT]
)
_NonCSRMatrix: TypeAlias = (
    bsr_matrix[_SCT] | coo_matrix[_SCT] | csc_matrix[_SCT] | dia_matrix[_SCT] | dok_matrix[_SCT] | lil_matrix[_SCT]
)
_NonDIAMatrix: TypeAlias = (
    bsr_matrix[_SCT] | coo_matrix[_SCT] | csc_matrix[_SCT] | csr_matrix[_SCT] | dok_matrix[_SCT] | lil_matrix[_SCT]
)

_SpMatrixOut: TypeAlias = coo_matrix[_SCT] | csc_matrix[_SCT] | csr_matrix[_SCT]
_SpMatrixNonOut: TypeAlias = bsr_matrix[_SCT] | dia_matrix[_SCT] | dok_matrix[_SCT] | lil_matrix[_SCT]
_SpArrayOut: TypeAlias = coo_array[_SCT, _ShapeT] | csc_array[_SCT] | csr_array[_SCT, _ShapeT]
_SpArrayNonOut: TypeAlias = bsr_array[_SCT] | dia_array[_SCT] | dok_array[_SCT, tuple[int, int]] | lil_array[_SCT]

_FmtBSR: TypeAlias = Literal["bsr"]
_FmtCOO: TypeAlias = Literal["coo"]
_FmtCSR: TypeAlias = Literal["csr"]
_FmtDIA: TypeAlias = Literal["dia"]
_FmtOut: TypeAlias = Literal["coo", "csc", "csr"]
_FmtNonBSR: TypeAlias = Literal["coo", "csc", "csr", "dia", "dok", "lil"]
_FmtNonCOO: TypeAlias = Literal["bsr", "csc", "csr", "dia", "dok", "lil"]
_FmtNonCSR: TypeAlias = Literal["bsr", "coo", "csc", "dia", "dok", "lil"]
_FmtNonDIA: TypeAlias = Literal["bsr", "coo", "csc", "csr", "dok", "lil"]
_FmtNonOut: TypeAlias = Literal["bsr", "dia", "dok", "lil"]

_DataRVS: TypeAlias = Callable[[int], onp.ArrayND[Scalar]]

_ToBlocks: TypeAlias = Seq[Seq[_spbase]] | onp.ArrayND[np.object_]

@type_check_only
class _DataSampler(Protocol):
    def __call__(self, /, *, size: int) -> onp.ArrayND[Scalar]: ...

###

#
@overload  # diagonals: <known>, dtype: None = ..., format: {"dia", None} = ...
def diags_array(
    diagonals: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: None = None,
) -> dia_array[_SCT]: ...
@overload  # diagonals: <known>, dtype: None = ..., format: <otherwise>
def diags_array(
    diagonals: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtNonDIA,
    dtype: None = None,
) -> _NonDIAArray[_SCT]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: int
def diags_array(
    diagonals: onp.ToFloat1D | onp.ToFloat2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: ToDTypeInt,
) -> dia_array[np.int_]: ...
@overload  # diagonals: <unknown>, format: <otherwise>, dtype: int
def diags_array(
    diagonals: onp.ToFloat1D | onp.ToFloat2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtNonDIA,
    dtype: ToDTypeInt,
) -> _NonDIAArray[np.int_]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: float
def diags_array(
    diagonals: onp.ToFloat1D | onp.ToFloat2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: ToDTypeFloat,
) -> dia_array[np.float64]: ...
@overload  # diagonals: <unknown>, format: <otherwise>, dtype: float
def diags_array(
    diagonals: onp.ToFloat1D | onp.ToFloat2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtNonDIA,
    dtype: ToDTypeFloat,
) -> _NonDIAArray[np.float64]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: complex
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: ToDTypeComplex,
) -> dia_array[np.complex128]: ...
@overload  # diagonals: <unknown>, format: <otherwise>, dtype: complex
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtNonDIA,
    dtype: ToDTypeComplex,
) -> _NonDIAArray[np.complex128]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: <known>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: ToDType[_SCT],
) -> dia_array[_SCT]: ...
@overload  # diagonals: <unknown>, format: <otherwise>, dtype: <known>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtNonDIA,
    dtype: ToDType[_SCT],
) -> _NonDIAArray[_SCT]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: <unknown>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: npt.DTypeLike | None = None,
) -> dia_array: ...
@overload  # diagonals: <unknown>, format: <otherwise>, dtype: <unknown>
def diags_array(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    /,
    *,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtNonDIA,
    dtype: npt.DTypeLike | None = None,
) -> _NonDIAArray: ...

# NOTE: `diags_array` should be prefered over `diags`
@overload  # diagonals: <known>, format: {"dia", None} = ...
def diags(
    diagonals: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: ToDType[_SCT] | None = None,
) -> dia_array[_SCT]: ...
@overload  # diagonals: <known>, format: <otherwise> (positional)
def diags(
    diagonals: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    offsets: onp.ToInt | onp.ToInt1D,
    shape: ToShape2D | None,
    format: _FmtNonDIA,
    dtype: ToDType[_SCT] | None = None,
) -> _NonDIAArray[_SCT]: ...
@overload  # diagonals: <known>, format: <otherwise> (keyword)
def diags(
    diagonals: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    *,
    format: _FmtNonDIA,
    dtype: ToDType[_SCT] | None = None,
) -> _NonDIAArray[_SCT]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: <known> (positional)
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D,
    shape: ToShape2D | None,
    format: _FmtDIA | None,
    dtype: ToDType[_SCT],
) -> dia_array[_SCT]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: <known> (keyword)
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    *,
    dtype: ToDType[_SCT],
) -> dia_array[_SCT]: ...
@overload  # diagonals: <unknown>, format: <otherwise> (positional), dtype: <known>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D,
    shape: ToShape2D | None,
    format: _FmtNonDIA,
    dtype: ToDType[_SCT],
) -> _NonDIAArray[_SCT]: ...
@overload  # diagonals: <unknown>, format: <otherwise> (keyword), dtype: <known>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    *,
    format: _FmtNonDIA,
    dtype: ToDType[_SCT],
) -> _NonDIAArray[_SCT]: ...
@overload  # diagonals: <unknown>, format: {"dia", None} = ..., dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    format: _FmtDIA | None = None,
    dtype: npt.DTypeLike | None = None,
) -> dia_array: ...
@overload  # diagonals: <unknown>, format: <otherwise> (positional), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D,
    shape: ToShape2D | None,
    format: _FmtNonDIA,
    dtype: npt.DTypeLike | None = None,
) -> _NonDIAArray: ...
@overload  # diagonals: <unknown>, format: <otherwise> (keyword), dtype: <unknown>
def diags(
    diagonals: onp.ToComplex1D | onp.ToComplex2D,
    offsets: onp.ToInt | onp.ToInt1D = 0,
    shape: ToShape2D | None = None,
    *,
    format: _FmtNonDIA,
    dtype: npt.DTypeLike | None = None,
) -> _NonDIAArray: ...

# NOTE: `diags_array` should be prefered over `spdiags`
@overload
def spdiags(
    data: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: onp.ToJustInt,
    n: onp.ToJustInt,
    format: _FmtDIA | None = None,
) -> dia_matrix[_SCT]: ...
@overload
def spdiags(
    data: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    n: None = None,
    format: _FmtDIA | None = None,
) -> dia_matrix[_SCT]: ...
@overload
def spdiags(
    data: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: onp.ToJustInt,
    n: onp.ToJustInt,
    format: _FmtNonDIA,
) -> _NonDIAMatrix[_SCT]: ...
@overload
def spdiags(
    data: _ToArray1D[_SCT] | _ToArray2D[_SCT],
    diags: onp.ToInt | onp.ToInt1D,
    m: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    n: None = None,
    *,
    format: _FmtNonDIA,
) -> _NonDIAMatrix[_SCT]: ...
@overload
def spdiags(
    data: Seq[complex] | Seq[Seq[complex]],
    diags: onp.ToInt | onp.ToInt1D,
    m: onp.ToJustInt,
    n: onp.ToJustInt,
    format: SPFormat | None = None,
) -> _SpMatrix: ...
@overload
def spdiags(
    data: Seq[complex] | Seq[Seq[complex]],
    diags: onp.ToInt | onp.ToInt1D,
    m: tuple[onp.ToJustInt, onp.ToJustInt] | None = None,
    n: None = None,
    *,
    format: SPFormat | None = None,
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
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: {"bsr", None} = ...
def kron(A: _ToSpMatrix[_SCT1], B: _ToSpMatrix[_SCT2], format: _FmtBSR | None = None) -> bsr_matrix[_SCT1 | _SCT2]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: <otherwise>
def kron(A: _ToSpMatrix[_SCT1], B: _ToSpMatrix[_SCT2], format: _FmtNonBSR) -> _NonBSRMatrix[_SCT1 | _SCT2]: ...
@overload  # A: sparray, B: sparse, format: {"bsr", None} = ...
def kron(A: _SpArray[_SCT1], B: _spbase[_SCT2], format: _FmtBSR | None = None) -> _BSRArray[_SCT1 | _SCT2]: ...
@overload  # A: sparray, B: sparse, format: <otherwise>
def kron(A: _SpArray[_SCT1], B: _spbase[_SCT2], format: _FmtNonBSR) -> _NonBSRArray[_SCT1 | _SCT2]: ...
@overload  # A: sparse, B: sparray, format: {"bsr", None} = ...
def kron(A: _spbase[_SCT1], B: _SpArray[_SCT2], format: _FmtBSR | None = None) -> _BSRArray[_SCT1 | _SCT2]: ...
@overload  # A: sparse, B: sparray, format: <otherwise>
def kron(A: _spbase[_SCT1], B: _SpArray[_SCT2], format: _FmtNonBSR) -> _NonBSRArray[_SCT1 | _SCT2]: ...
@overload  # A: unknown array-like, B: unknown array-like  (catch-all)
def kron(
    A: onp.ToComplex2D | _spbase[_SCT],
    B: onp.ToComplex2D | _spbase[_SCT],
    format: SPFormat | None = None,
) -> _SpArray2D[_SCT] | _SpMatrix[_SCT]: ...

#
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: {"csr", None} = ...
def kronsum(A: _ToSpMatrix[_SCT1], B: _ToSpMatrix[_SCT2], format: _FmtCSR | None = None) -> csr_matrix[_SCT1 | _SCT2]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: <otherwise>
def kronsum(A: _ToSpMatrix[_SCT1], B: _ToSpMatrix[_SCT2], format: _FmtNonCSR) -> _NonCSRMatrix[_SCT1 | _SCT2]: ...
@overload  # A: sparray, B: sparse, format: {"csr", None} = ...
def kronsum(A: _SpArray[_SCT1], B: _spbase[_SCT2], format: _FmtCSR | None = None) -> _CSRArray[_SCT1 | _SCT2]: ...
@overload  # A: sparray, B: sparse, format: <otherwise>
def kronsum(A: _SpArray[_SCT1], B: _spbase[_SCT2], format: _FmtNonCSR) -> _NonCSRArray[_SCT1 | _SCT2]: ...
@overload  # A: sparse, B: sparray, format: {"csr", None} = ...
def kronsum(A: _spbase[_SCT1], B: _SpArray[_SCT2], format: _FmtCSR | None = None) -> _CSRArray[_SCT1 | _SCT2]: ...
@overload  # A: sparse, B: sparray, format: <otherwise>
def kronsum(A: _spbase[_SCT1], B: _SpArray[_SCT2], format: _FmtNonCSR) -> _NonCSRArray[_SCT1 | _SCT2]: ...
@overload  # A: unknown array-like, B: unknown array-like  (catch-all)
def kronsum(
    A: onp.ToComplex2D | _spbase[_SCT],
    B: onp.ToComplex2D | _spbase[_SCT],
    format: SPFormat | None = None,
) -> _SpArray2D[_SCT] | _SpMatrix[_SCT]: ...

# NOTE: hstack and vstack have identical signatures
@overload  # sparray, format: <default>, dtype: None
def hstack(blocks: Seq[_SpArray[_SCT]], format: _FmtOut | None = None, dtype: None = None) -> _SpArrayOut[_SCT]: ...
@overload  # sparray, format: <non-default>, dtype: None
def hstack(blocks: Seq[_SpArray[_SCT]], format: _FmtNonOut, dtype: None = None) -> _SpArrayNonOut[_SCT]: ...
@overload  # sparray, format: <default>, dtype: <int>
def hstack(blocks: Seq[_SpArray], format: _FmtOut | None = None, *, dtype: ToDTypeBool) -> _SpArrayOut[np.bool_]: ...
@overload  # sparray, format: <non-default>, dtype: <int>
def hstack(blocks: Seq[_SpArray], format: _FmtNonOut, dtype: ToDTypeBool) -> _SpArrayNonOut[np.bool_]: ...
@overload  # sparray, format: <default>, dtype: <int>
def hstack(blocks: Seq[_SpArray], format: _FmtOut | None = None, *, dtype: ToDTypeInt) -> _SpArrayOut[np.int_]: ...
@overload  # sparray, format: <non-default>, dtype: <int>
def hstack(blocks: Seq[_SpArray], format: _FmtNonOut, dtype: ToDTypeInt) -> _SpArrayNonOut[np.int_]: ...
@overload  # sparray, format: <default>, dtype: <float>
def hstack(blocks: Seq[_SpArray], format: _FmtOut | None = None, *, dtype: ToDTypeFloat) -> _SpArrayOut[np.float64]: ...
@overload  # sparray, format: <non-default>, dtype: <float>
def hstack(blocks: Seq[_SpArray], format: _FmtNonOut, dtype: ToDTypeFloat) -> _SpArrayNonOut[np.float64]: ...
@overload  # sparray, format: <default>, dtype: <complex>
def hstack(blocks: Seq[_SpArray], format: _FmtOut | None = None, *, dtype: ToDTypeComplex) -> _SpArrayOut[np.complex128]: ...
@overload  # sparray, format: <non-default>, dtype: <complex>
def hstack(blocks: Seq[_SpArray], format: _FmtNonOut, dtype: ToDTypeComplex) -> _SpArrayNonOut[np.complex128]: ...
@overload  # sparray, format: <default>, dtype: <known>
def hstack(blocks: Seq[_SpArray], format: _FmtOut | None = None, *, dtype: ToDType[_SCT]) -> _SpArrayOut[_SCT]: ...
@overload  # sparray, format: <non-default>, dtype: <known>
def hstack(blocks: Seq[_SpArray], format: _FmtNonOut, dtype: ToDType[_SCT]) -> _SpArrayNonOut[_SCT]: ...
@overload  # sparray, format: <default>, dtype: <unknown>
def hstack(blocks: Seq[_SpArray], format: _FmtOut | None = None, *, dtype: npt.DTypeLike) -> _SpArrayOut: ...
@overload  # sparray, format: <non-default>, dtype: <unknown>
def hstack(blocks: Seq[_SpArray], format: _FmtNonOut, dtype: npt.DTypeLike) -> _SpArrayNonOut: ...
@overload  # spmatrix, format: <default>, dtype: None
def hstack(blocks: Seq[spmatrix[_SCT]], format: _FmtOut | None = None, dtype: None = None) -> _SpMatrixOut[_SCT]: ...
@overload  # spmatrix, format: <non-default>, dtype: None
def hstack(blocks: Seq[spmatrix[_SCT]], format: _FmtNonOut, dtype: None = None) -> _SpMatrixNonOut[_SCT]: ...
@overload  # spmatrix, format: <default>, dtype: <int>
def hstack(blocks: Seq[spmatrix], format: _FmtOut | None = None, *, dtype: ToDTypeBool) -> _SpMatrixOut[np.bool_]: ...
@overload  # spmatrix, format: <non-default>, dtype: <int>
def hstack(blocks: Seq[spmatrix], format: _FmtNonOut, dtype: ToDTypeBool) -> _SpMatrixNonOut[np.bool_]: ...
@overload  # spmatrix, format: <default>, dtype: <int>
def hstack(blocks: Seq[spmatrix], format: _FmtOut | None = None, *, dtype: ToDTypeInt) -> _SpMatrixOut[np.int_]: ...
@overload  # spmatrix, format: <non-default>, dtype: <int>
def hstack(blocks: Seq[spmatrix], format: _FmtNonOut, dtype: ToDTypeInt) -> _SpMatrixNonOut[np.int_]: ...
@overload  # spmatrix, format: <default>, dtype: <float>
def hstack(blocks: Seq[spmatrix], format: _FmtOut | None = None, *, dtype: ToDTypeFloat) -> _SpMatrixOut[np.float64]: ...
@overload  # spmatrix, format: <non-default>, dtype: <float>
def hstack(blocks: Seq[spmatrix], format: _FmtNonOut, dtype: ToDTypeFloat) -> _SpMatrixNonOut[np.float64]: ...
@overload  # spmatrix, format: <default>, dtype: <complex>
def hstack(blocks: Seq[spmatrix], format: _FmtOut | None = None, *, dtype: ToDTypeComplex) -> _SpMatrixOut[np.complex128]: ...
@overload  # spmatrix, format: <non-default>, dtype: <complex>
def hstack(blocks: Seq[spmatrix], format: _FmtNonOut, dtype: ToDTypeComplex) -> _SpMatrixNonOut[np.complex128]: ...
@overload  # spmatrix, format: <default>, dtype: <known>
def hstack(blocks: Seq[spmatrix], format: _FmtOut | None = None, *, dtype: ToDType[_SCT]) -> _SpMatrixOut[_SCT]: ...
@overload  # spmatrix, format: <non-default>, dtype: <known>
def hstack(blocks: Seq[spmatrix], format: _FmtNonOut, dtype: ToDType[_SCT]) -> _SpMatrixNonOut[_SCT]: ...
@overload  # spmatrix, format: <default>, dtype: <unknown>
def hstack(blocks: Seq[spmatrix], format: _FmtOut | None = None, *, dtype: npt.DTypeLike) -> _SpMatrixOut: ...
@overload  # spmatrix, format: <non-default>, dtype: <unknown>
def hstack(blocks: Seq[spmatrix], format: _FmtNonOut, dtype: npt.DTypeLike) -> _SpMatrixNonOut: ...

#
@overload  # sparray, format: <default>, dtype: None
def vstack(blocks: Seq[_SpArray[_SCT]], format: _FmtOut | None = None, dtype: None = None) -> _SpArrayOut[_SCT]: ...
@overload  # sparray, format: <non-default>, dtype: None
def vstack(blocks: Seq[_SpArray[_SCT]], format: _FmtNonOut, dtype: None = None) -> _SpArrayNonOut[_SCT]: ...
@overload  # sparray, format: <default>, dtype: <int>
def vstack(blocks: Seq[_SpArray], format: _FmtOut | None = None, *, dtype: ToDTypeBool) -> _SpArrayOut[np.bool_]: ...
@overload  # sparray, format: <non-default>, dtype: <int>
def vstack(blocks: Seq[_SpArray], format: _FmtNonOut, dtype: ToDTypeBool) -> _SpArrayNonOut[np.bool_]: ...
@overload  # sparray, format: <default>, dtype: <int>
def vstack(blocks: Seq[_SpArray], format: _FmtOut | None = None, *, dtype: ToDTypeInt) -> _SpArrayOut[np.int_]: ...
@overload  # sparray, format: <non-default>, dtype: <int>
def vstack(blocks: Seq[_SpArray], format: _FmtNonOut, dtype: ToDTypeInt) -> _SpArrayNonOut[np.int_]: ...
@overload  # sparray, format: <default>, dtype: <float>
def vstack(blocks: Seq[_SpArray], format: _FmtOut | None = None, *, dtype: ToDTypeFloat) -> _SpArrayOut[np.float64]: ...
@overload  # sparray, format: <non-default>, dtype: <float>
def vstack(blocks: Seq[_SpArray], format: _FmtNonOut, dtype: ToDTypeFloat) -> _SpArrayNonOut[np.float64]: ...
@overload  # sparray, format: <default>, dtype: <complex>
def vstack(blocks: Seq[_SpArray], format: _FmtOut | None = None, *, dtype: ToDTypeComplex) -> _SpArrayOut[np.complex128]: ...
@overload  # sparray, format: <non-default>, dtype: <complex>
def vstack(blocks: Seq[_SpArray], format: _FmtNonOut, dtype: ToDTypeComplex) -> _SpArrayNonOut[np.complex128]: ...
@overload  # sparray, format: <default>, dtype: <known>
def vstack(blocks: Seq[_SpArray], format: _FmtOut | None = None, *, dtype: ToDType[_SCT]) -> _SpArrayOut[_SCT]: ...
@overload  # sparray, format: <non-default>, dtype: <known>
def vstack(blocks: Seq[_SpArray], format: _FmtNonOut, dtype: ToDType[_SCT]) -> _SpArrayNonOut[_SCT]: ...
@overload  # sparray, format: <default>, dtype: <unknown>
def vstack(blocks: Seq[_SpArray], format: _FmtOut | None = None, *, dtype: npt.DTypeLike) -> _SpArrayOut: ...
@overload  # sparray, format: <non-default>, dtype: <unknown>
def vstack(blocks: Seq[_SpArray], format: _FmtNonOut, dtype: npt.DTypeLike) -> _SpArrayNonOut: ...
@overload  # spmatrix, format: <default>, dtype: None
def vstack(blocks: Seq[spmatrix[_SCT]], format: _FmtOut | None = None, dtype: None = None) -> _SpMatrixOut[_SCT]: ...
@overload  # spmatrix, format: <non-default>, dtype: None
def vstack(blocks: Seq[spmatrix[_SCT]], format: _FmtNonOut, dtype: None = None) -> _SpMatrixNonOut[_SCT]: ...
@overload  # spmatrix, format: <default>, dtype: <int>
def vstack(blocks: Seq[spmatrix], format: _FmtOut | None = None, *, dtype: ToDTypeBool) -> _SpMatrixOut[np.bool_]: ...
@overload  # spmatrix, format: <non-default>, dtype: <int>
def vstack(blocks: Seq[spmatrix], format: _FmtNonOut, dtype: ToDTypeBool) -> _SpMatrixNonOut[np.bool_]: ...
@overload  # spmatrix, format: <default>, dtype: <int>
def vstack(blocks: Seq[spmatrix], format: _FmtOut | None = None, *, dtype: ToDTypeInt) -> _SpMatrixOut[np.int_]: ...
@overload  # spmatrix, format: <non-default>, dtype: <int>
def vstack(blocks: Seq[spmatrix], format: _FmtNonOut, dtype: ToDTypeInt) -> _SpMatrixNonOut[np.int_]: ...
@overload  # spmatrix, format: <default>, dtype: <float>
def vstack(blocks: Seq[spmatrix], format: _FmtOut | None = None, *, dtype: ToDTypeFloat) -> _SpMatrixOut[np.float64]: ...
@overload  # spmatrix, format: <non-default>, dtype: <float>
def vstack(blocks: Seq[spmatrix], format: _FmtNonOut, dtype: ToDTypeFloat) -> _SpMatrixNonOut[np.float64]: ...
@overload  # spmatrix, format: <default>, dtype: <complex>
def vstack(blocks: Seq[spmatrix], format: _FmtOut | None = None, *, dtype: ToDTypeComplex) -> _SpMatrixOut[np.complex128]: ...
@overload  # spmatrix, format: <non-default>, dtype: <complex>
def vstack(blocks: Seq[spmatrix], format: _FmtNonOut, dtype: ToDTypeComplex) -> _SpMatrixNonOut[np.complex128]: ...
@overload  # spmatrix, format: <default>, dtype: <known>
def vstack(blocks: Seq[spmatrix], format: _FmtOut | None = None, *, dtype: ToDType[_SCT]) -> _SpMatrixOut[_SCT]: ...
@overload  # spmatrix, format: <non-default>, dtype: <known>
def vstack(blocks: Seq[spmatrix], format: _FmtNonOut, dtype: ToDType[_SCT]) -> _SpMatrixNonOut[_SCT]: ...
@overload  # spmatrix, format: <default>, dtype: <unknown>
def vstack(blocks: Seq[spmatrix], format: _FmtOut | None = None, *, dtype: npt.DTypeLike) -> _SpMatrixOut: ...
@overload  # spmatrix, format: <non-default>, dtype: <unknown>
def vstack(blocks: Seq[spmatrix], format: _FmtNonOut, dtype: npt.DTypeLike) -> _SpMatrixNonOut: ...

_COOArray2D: TypeAlias = coo_array[_SCT, tuple[int, int]]

#
@overload  # blocks: <known dtype>, format: <default>, dtype: <default>
def block_array(blocks: Seq[Seq[_spbase[_SCT]]], *, format: _FmtCOO | None = None, dtype: None = None) -> _COOArray2D[_SCT]: ...
@overload  # blocks: <unknown dtype>, format: <default>, dtype: <known>
def block_array(blocks: _ToBlocks, *, format: _FmtCOO | None = None, dtype: ToDType[_SCT]) -> _COOArray2D[_SCT]: ...
@overload  # blocks: <unknown dtype>, format: <default>, dtype: <unknown>
def block_array(blocks: _ToBlocks, *, format: _FmtCOO | None = None, dtype: npt.DTypeLike) -> _COOArray2D: ...
@overload  # blocks: <known dtype>, format: <otherwise>, dtype: <default>
def block_array(blocks: Seq[Seq[_spbase[_SCT]]], *, format: _FmtNonCOO, dtype: None = None) -> _NonCOOArray[_SCT]: ...
@overload  # blocks: <unknown dtype>, format: <otherwise>, dtype: <known>
def block_array(blocks: _ToBlocks, *, format: _FmtNonCOO, dtype: ToDType[_SCT]) -> _NonCOOArray[_SCT]: ...
@overload  # blocks: <unknown dtype>, format: <otherwise>, dtype: <unknown>
def block_array(blocks: _ToBlocks, *, format: _FmtNonCOO, dtype: npt.DTypeLike) -> _NonCOOArray: ...

#
@overload  # blocks: <array, known dtype>, format: <default>, dtype: <default>
def bmat(blocks: Seq[Seq[_SpArray[_SCT]]], format: _FmtCOO | None = None, dtype: None = None) -> _COOArray2D[_SCT]: ...
@overload  # blocks: <matrix, known dtype>, format: <default>, dtype: <default>
def bmat(blocks: Seq[Seq[spmatrix[_SCT]]], format: _FmtCOO | None = None, dtype: None = None) -> coo_matrix[_SCT]: ...
@overload  # sparray, blocks: <unknown, unknown dtype>, format: <default>, dtype: <known> (positional)
def bmat(blocks: _ToBlocks, format: _FmtCOO | None, dtype: ToDType[_SCT]) -> _COOArray2D[_SCT] | coo_matrix[_SCT]: ...
@overload  # sparray, blocks: <unknown, unknown dtype>, format: <default>, dtype: <known> (keyword)
def bmat(blocks: _ToBlocks, format: _FmtCOO | None = None, *, dtype: ToDType[_SCT]) -> _COOArray2D[_SCT] | coo_matrix[_SCT]: ...
@overload  # sparray, blocks: <unknown, unknown dtype>, format: <default>, dtype: <unknown>
def bmat(blocks: _ToBlocks, format: _FmtCOO | None = None, dtype: npt.DTypeLike | None = None) -> _COOArray2D | coo_matrix: ...
@overload  # sparray, blocks: <array, known dtype>, format: <otherwise>, dtype: <default>
def bmat(blocks: Seq[Seq[_SpArray[_SCT]]], format: _FmtNonCOO, dtype: None = None) -> _NonCOOArray[_SCT]: ...
@overload  # sparray, blocks: <matrix, known dtype>, format: <otherwise>, dtype: <default>
def bmat(blocks: Seq[Seq[spmatrix[_SCT]]], format: _FmtNonCOO, dtype: None = None) -> _NonCOOMatrix[_SCT]: ...
@overload  # sparray, blocks: <unknown, unknown dtype>, format: <otherwise>, dtype: <known>
def bmat(blocks: _ToBlocks, format: _FmtNonCOO, dtype: ToDType[_SCT]) -> _NonCOOArray[_SCT] | _NonCOOMatrix[_SCT]: ...
@overload  # sparray, blocks: <unknown, unknown dtype>, format: <otherwise>, dtype: <unknown>
def bmat(blocks: _ToBlocks, format: _FmtNonCOO, dtype: npt.DTypeLike) -> _NonCOOArray | _NonCOOMatrix: ...

#
@overload  # mats: <array, known dtype>
def block_diag(
    mats: Iterable[_SpArray[_SCT]],
    format: SPFormat | None = None,
    dtype: None = None,
) -> _SpArray[_SCT, tuple[int, int]]: ...
@overload  # mats: <matrix, known dtype>
def block_diag(
    mats: Iterable[spmatrix[_SCT]],
    format: SPFormat | None = None,
    dtype: None = None,
) -> _SpMatrix[_SCT]: ...
@overload  # mats: <unknown, known dtype>
def block_diag(
    mats: Iterable[_spbase[_SCT] | onp.ArrayND[_SCT]],
    format: SPFormat | None = None,
    dtype: None = None,
) -> _SpArray[_SCT, tuple[int, int]] | _SpMatrix[_SCT]: ...
@overload  # mats: <array, unknown dtype>, dtype: <known>  (positional)
def block_diag(
    mats: Iterable[sparray],
    format: SPFormat | None,
    dtype: ToDType[_SCT],
) -> _SpArray[_SCT, tuple[int, int]]: ...
@overload  # mats: <array, unknown dtype>, dtype: <known>  (keyword)
def block_diag(
    mats: Iterable[sparray],
    format: SPFormat | None = None,
    *,
    dtype: ToDType[_SCT],
) -> _SpArray[_SCT, tuple[int, int]]: ...
@overload  # mats: <matrix, unknown dtype>, dtype: <known>  (positional)
def block_diag(
    mats: Iterable[spmatrix | onp.ArrayND[Scalar] | complex | list[onp.ToComplex] | list[onp.ToComplex1D]],
    format: SPFormat | None,
    dtype: ToDType[_SCT],
) -> _SpMatrix[_SCT]: ...
@overload  # mats: <matrix, unknown dtype>, dtype: <known>  (keyword)
def block_diag(
    mats: Iterable[spmatrix | onp.ArrayND[Scalar] | complex | list[onp.ToComplex] | list[onp.ToComplex1D]],
    format: SPFormat | None = None,
    *,
    dtype: ToDType[_SCT],
) -> _SpMatrix[_SCT]: ...
@overload  # mats: <unknown, unknown dtype>, dtype: <known>  (positional)
def block_diag(
    mats: Iterable[_spbase | onp.ArrayND[Scalar] | complex | list[onp.ToComplex] | list[onp.ToComplex1D]],
    format: SPFormat | None,
    dtype: ToDType[_SCT],
) -> _SpArray[_SCT, tuple[int, int]] | _SpMatrix[_SCT]: ...
@overload  # mats: <unknown, unknown dtype>, dtype: <known>  (keyword)
def block_diag(
    mats: Iterable[_spbase | onp.ArrayND[Scalar] | complex | list[onp.ToComplex] | list[onp.ToComplex1D]],
    format: SPFormat | None = None,
    *,
    dtype: ToDType[_SCT],
) -> _SpArray[_SCT, tuple[int, int]] | _SpMatrix[_SCT]: ...
@overload  # catch-all
def block_diag(
    mats: Iterable[_spbase | onp.ArrayND[Scalar] | complex | list[onp.ToComplex] | list[onp.ToComplex1D]],
    format: SPFormat | None = None,
    dtype: npt.DTypeLike | None = None,
) -> _SpArray[Any, tuple[int, int]] | _SpMatrix[Any]: ...

#
@overload  # shape: 1d, dtype: <default>
def random_array(
    shape: tuple[int],
    *,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: ToDTypeFloat | None = None,
    random_state: Seed | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray1D[np.float64]: ...
@overload  # shape: 1d, dtype: <known>
def random_array(
    shape: tuple[int],
    *,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: ToDType[_SCT],
    random_state: Seed | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray1D[_SCT]: ...
@overload  # shape: 1d, dtype: complex
def random_array(
    shape: tuple[int],
    *,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: ToDTypeComplex,
    random_state: Seed | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray1D[np.complex128]: ...
@overload  # shape: 1d, dtype: <unknown>
def random_array(
    shape: tuple[int],
    *,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: npt.DTypeLike,
    random_state: Seed | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray1D: ...
@overload  # shape: 2d, dtype: <default>
def random_array(
    shape: tuple[int, int],
    *,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: ToDTypeFloat | None = None,
    random_state: Seed | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray2D[np.float64]: ...
@overload  # shape: 2d, dtype: <known>
def random_array(
    shape: tuple[int, int],
    *,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: ToDType[_SCT],
    random_state: Seed | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray2D[_SCT]: ...
@overload  # shape: 2d, dtype: complex
def random_array(
    shape: tuple[int, int],
    *,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: ToDTypeComplex,
    random_state: Seed | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray2D[np.complex128]: ...
@overload  # shape: 2d, dtype: <unknown>
def random_array(
    shape: tuple[int, int],
    *,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: npt.DTypeLike,
    random_state: Seed | None = None,
    data_sampler: _DataSampler | None = None,
) -> _SpArray2D: ...

# NOTE: `random_array` should be prefered over `random`
@overload  # dtype: <default>
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: ToDTypeFloat | None = None,
    random_state: Seed | None = None,
    data_rvs: _DataRVS | None = None,
) -> _SpMatrix[np.float64]: ...
@overload  # dtype: <known> (positional)
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float,
    format: SPFormat,
    dtype: ToDType[_SCT],
    random_state: Seed | None = None,
    data_rvs: _DataRVS | None = None,
) -> _SpMatrix[_SCT]: ...
@overload  # dtype: <known> (keyword)
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    *,
    dtype: ToDType[_SCT],
    random_state: Seed | None = None,
    data_rvs: _DataRVS | None = None,
) -> _SpMatrix[_SCT]: ...
@overload  # dtype: complex (positional)
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float,
    format: SPFormat,
    dtype: ToDTypeComplex,
    random_state: Seed | None = None,
    data_rvs: _DataRVS | None = None,
) -> _SpMatrix[np.complex128]: ...
@overload  # dtype: complex (keyword)
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    *,
    dtype: ToDTypeComplex,
    random_state: Seed | None = None,
    data_rvs: _DataRVS | None = None,
) -> _SpMatrix[np.complex128]: ...
@overload  # dtype: <unknown>
def random(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: npt.DTypeLike | None = None,
    random_state: Seed | None = None,
    data_rvs: _DataRVS | None = None,
) -> _SpMatrix: ...

# NOTE: `random_array` should be prefered over `rand`
@overload  # dtype: <default>
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: ToDTypeFloat | None = None,
    random_state: Seed | None = None,
) -> _SpMatrix[np.float64]: ...
@overload  # dtype: <known> (positional)
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float,
    format: SPFormat,
    dtype: ToDType[_SCT],
    random_state: Seed | None = None,
) -> _SpMatrix[_SCT]: ...
@overload  # dtype: <known> (keyword)
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    *,
    dtype: ToDType[_SCT],
    random_state: Seed | None = None,
) -> _SpMatrix[_SCT]: ...
@overload  # dtype: complex (positional)
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float,
    format: SPFormat,
    dtype: ToDTypeComplex,
    random_state: Seed | None = None,
) -> _SpMatrix[np.complex128]: ...
@overload  # dtype: complex (keyword)
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    *,
    dtype: ToDTypeComplex,
    random_state: Seed | None = None,
) -> _SpMatrix[np.complex128]: ...
@overload  # dtype: <unknown>
def rand(
    m: opt.AnyInt,
    n: opt.AnyInt,
    density: float | Float = 0.01,
    format: SPFormat = "coo",
    dtype: npt.DTypeLike | None = None,
    random_state: Seed | None = None,
) -> _SpMatrix: ...
