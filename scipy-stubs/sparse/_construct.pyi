from collections.abc import Sequence as Seq
from typing import Any, Literal, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import optype.typing as opt
from scipy._typing import Seed, Untyped, UntypedCallable
from ._base import _spbase
from ._bsr import bsr_array, bsr_matrix
from ._coo import coo_array, coo_matrix
from ._csc import csc_array, csc_matrix
from ._csr import csr_array, csr_matrix
from ._dia import dia_array, dia_matrix
from ._dok import dok_array, dok_matrix
from ._lil import lil_array, lil_matrix
from ._matrix import spmatrix
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
_SCT1 = TypeVar("_SCT1", bound=Scalar)
_SCT2 = TypeVar("_SCT2", bound=Scalar)
_ShapeT = TypeVar("_ShapeT", bound=tuple[int] | tuple[int, int], default=tuple[int] | tuple[int, int])

_ToMatrix: TypeAlias = spmatrix[_SCT] | Seq[Seq[_SCT] | onp.CanArrayND[_SCT]] | onp.CanArrayND[_SCT]

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
_NonCSRArray: TypeAlias = (
    bsr_array[_SCT]
    | coo_array[_SCT, tuple[int, int]]
    | csc_array[_SCT]
    | dia_array[_SCT]
    | dok_array[_SCT, tuple[int, int]]
    | lil_array[_SCT]
)
_NonBSRMatrix: TypeAlias = (
    coo_matrix[_SCT] | csr_matrix[_SCT] | csc_matrix[_SCT] | dia_matrix[_SCT] | dok_matrix[_SCT] | lil_matrix[_SCT]
)
_NonCSRMatrix: TypeAlias = (
    bsr_matrix[_SCT] | coo_matrix[_SCT] | csc_matrix[_SCT] | dia_matrix[_SCT] | dok_matrix[_SCT] | lil_matrix[_SCT]
)

_SpMatrixOut: TypeAlias = coo_matrix[_SCT] | csc_matrix[_SCT] | csr_matrix[_SCT]
_SpMatrixNonOut: TypeAlias = bsr_matrix[_SCT] | dia_matrix[_SCT] | dok_matrix[_SCT] | lil_matrix[_SCT]
_SpArrayOut: TypeAlias = coo_array[_SCT, _ShapeT] | csc_array[_SCT] | csr_array[_SCT, _ShapeT]
_SpArrayNonOut: TypeAlias = bsr_array[_SCT] | dia_array[_SCT] | dok_array[_SCT, tuple[int, int]] | lil_array[_SCT]

_FmtBSR: TypeAlias = Literal["bsr"]
_FmtCSR: TypeAlias = Literal["csr"]
_FmtDIA: TypeAlias = Literal["dia"]
_FmtOut: TypeAlias = Literal["coo", "csc", "csr"]
_FmtNonBSR: TypeAlias = Literal["coo", "csc", "csr", "dia", "dok", "lil"]
_FmtNonCSR: TypeAlias = Literal["bsr", "coo", "csc", "dia", "dok", "lil"]
_FmtNonDIA: TypeAlias = Literal["bsr", "coo", "csc", "csr", "dok", "lil"]
_FmtNonOut: TypeAlias = Literal["bsr", "dia", "dok", "lil"]

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
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: {"bsr", None} = ...
def kron(A: _ToMatrix[_SCT1], B: _ToMatrix[_SCT2], format: _FmtBSR | None = None) -> bsr_matrix[_SCT1 | _SCT2]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: <otherwise>
def kron(A: _ToMatrix[_SCT1], B: _ToMatrix[_SCT2], format: _FmtNonBSR) -> _NonBSRMatrix[_SCT1 | _SCT2]: ...
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
def kronsum(A: _ToMatrix[_SCT1], B: _ToMatrix[_SCT2], format: _FmtCSR | None = None) -> csr_matrix[_SCT1 | _SCT2]: ...
@overload  # A: spmatrix or 2d array-like, B: spmatrix or 2d array-like, format: <otherwise>
def kronsum(A: _ToMatrix[_SCT1], B: _ToMatrix[_SCT2], format: _FmtNonCSR) -> _NonCSRMatrix[_SCT1 | _SCT2]: ...
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
