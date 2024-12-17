from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeVar, deprecated

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import optype.typing as opt
from scipy._typing import CorrelateMode

__all__ = [
    "block_diag",
    "circulant",
    "companion",
    "convolution_matrix",
    "dft",
    "fiedler",
    "fiedler_companion",
    "hadamard",
    "hankel",
    "helmert",
    "hilbert",
    "invhilbert",
    "invpascal",
    "kron",
    "leslie",
    "pascal",
    "toeplitz",
]

_SCT = TypeVar("_SCT", bound=np.generic, default=np.number[Any] | np.bool_ | np.object_)

_Kind: TypeAlias = Literal["symmetric", "upper", "lower"]

_Array2d: TypeAlias = onp.Array2D[_SCT]
_Int2d: TypeAlias = _Array2d[np.int_]
_Float2d: TypeAlias = _Array2d[np.float64]
_Complex2d: TypeAlias = _Array2d[np.complex128]

_ToArray1d: TypeAlias = onp.CanArrayND[_SCT] | Sequence[_SCT]
_ToArrayStrict1d: TypeAlias = onp.CanArray1D[_SCT] | Sequence[_SCT]
_ToArrayStrict1dPlus: TypeAlias = onp.CanArrayND[_SCT, onp.AtLeast1D] | onp.SequenceND[Sequence[_SCT] | onp.CanArrayND[_SCT]]

_JustInt1d: TypeAlias = Sequence[opt.JustInt | np.int_]
_JustFloat1d: TypeAlias = Sequence[opt.Just[float] | np.float64]
_JustComplex1d: TypeAlias = Sequence[opt.Just[complex] | np.complex128]
_JustInt1dPlus: TypeAlias = onp.SequenceND[_JustInt1d]
_JustFloat1dPlus: TypeAlias = onp.SequenceND[_JustFloat1d]
_JustComplex1dPlus: TypeAlias = onp.SequenceND[_JustComplex1d]

###

#
@overload
@deprecated("`kron` has been deprecated in favour of `numpy.kron` in SciPy 1.15.0 and will be removed in SciPy 1.17.0.")
def kron(a: _Array2d[_SCT], b: _Array2d[_SCT]) -> _Array2d[_SCT]: ...
@overload
@deprecated("`kron` has been deprecated in favour of `numpy.kron` in SciPy 1.15.0 and will be removed in SciPy 1.17.0.")
def kron(a: onp.ArrayND[_SCT], b: onp.ArrayND[_SCT]) -> onp.Array[onp.AtLeast2D, _SCT]: ...

#
@overload
def toeplitz(c: _JustInt1d, r: _JustInt1d | None = None) -> _Int2d: ...
@overload
def toeplitz(c: _JustFloat1d, r: _JustFloat1d | None = None) -> _Float2d: ...
@overload
def toeplitz(c: _JustComplex1d, r: _JustComplex1d | None = None) -> _Complex2d: ...
@overload
def toeplitz(c: _ToArrayStrict1d[_SCT], r: _ToArrayStrict1d[_SCT] | None = None) -> _Array2d[_SCT]: ...
@overload
@deprecated(
    "Beginning in SciPy 1.17, multidimensional input will be treated as a batch, not `ravel`ed. "
    "To preserve the existing behavior and silence this warning, `ravel` aruments before passing them to `toeplitz`.",
    category=FutureWarning,
)
def toeplitz(c: _JustInt1dPlus, r: _JustInt1dPlus | None = None) -> _Int2d: ...
@overload
@deprecated(
    "Beginning in SciPy 1.17, multidimensional input will be treated as a batch, not `ravel`ed. "
    "To preserve the existing behavior and silence this warning, `ravel` aruments before passing them to `toeplitz`.",
    category=FutureWarning,
)
def toeplitz(c: _JustFloat1dPlus, r: _JustFloat1dPlus | None = None) -> _Float2d: ...
@overload
@deprecated(
    "Beginning in SciPy 1.17, multidimensional input will be treated as a batch, not `ravel`ed. "
    "To preserve the existing behavior and silence this warning, `ravel` aruments before passing them to `toeplitz`.",
    category=FutureWarning,
)
def toeplitz(c: _JustComplex1dPlus, r: _JustComplex1dPlus | None = None) -> _Complex2d: ...
@overload
@deprecated(
    "Beginning in SciPy 1.17, multidimensional input will be treated as a batch, not `ravel`ed. "
    "To preserve the existing behavior and silence this warning, `ravel` aruments before passing them to `toeplitz`.",
    category=FutureWarning,
)
def toeplitz(c: _ToArrayStrict1dPlus[_SCT], r: _ToArrayStrict1dPlus[_SCT] | None = None) -> _Array2d[_SCT]: ...
@overload
def toeplitz(c: _ToArray1d[_SCT], r: _ToArray1d[_SCT] | None = None) -> _Array2d[_SCT]: ...

#
@overload
def circulant(c: _JustInt1d) -> _Int2d: ...
@overload
def circulant(c: _JustFloat1d) -> _Float2d: ...
@overload
def circulant(c: _JustComplex1d) -> _Complex2d: ...
@overload
def circulant(c: _ToArray1d[_SCT]) -> _Array2d[_SCT]: ...

#
@overload
def hankel(c: _JustInt1d, r: _JustInt1d | None = None) -> _Int2d: ...
@overload
def hankel(c: _JustFloat1d, r: _JustFloat1d | None = None) -> _Float2d: ...
@overload
def hankel(c: _JustComplex1d, r: _JustComplex1d | None = None) -> _Complex2d: ...
@overload
def hankel(c: _ToArray1d[_SCT], r: onp.CanArrayND[_SCT] | None = None) -> _Array2d[_SCT]: ...

#
@overload
def hadamard(n: onp.ToInt, dtype: type[opt.JustInt]) -> _Int2d: ...
@overload
def hadamard(n: onp.ToInt, dtype: type[opt.Just[float]]) -> _Float2d: ...
@overload
def hadamard(n: onp.ToInt, dtype: type[opt.Just[complex]]) -> _Complex2d: ...
@overload
def hadamard(n: onp.ToInt, dtype: onp.HasDType[np.dtype[_SCT]] | np.dtype[_SCT]) -> _Array2d[_SCT]: ...
@overload
def hadamard(n: onp.ToInt, dtype: npt.DTypeLike = ...) -> _Array2d[np.generic]: ...

#
@overload
def leslie(f: _JustInt1d, s: _JustInt1d) -> _Int2d: ...
@overload
def leslie(f: _JustFloat1d, s: _JustFloat1d) -> _Float2d: ...
@overload
def leslie(f: _JustComplex1d, s: _JustComplex1d) -> _Complex2d: ...
@overload
def leslie(f: _ToArray1d[_SCT], s: _ToArray1d[_SCT]) -> _Array2d[_SCT]: ...

#
@overload
def block_diag() -> _Float2d: ...
@overload
def block_diag(arr0: _JustInt1d, /, *arrs: _JustInt1d) -> _Int2d: ...
@overload
def block_diag(arr0: _JustFloat1d, /, *arrs: _JustFloat1d) -> _Float2d: ...
@overload
def block_diag(arr0: _JustComplex1d, /, *arrs: _JustComplex1d) -> _Complex2d: ...
@overload
def block_diag(arr0: _ToArray1d[_SCT], /, *arrs: _ToArray1d[_SCT]) -> _Array2d[_SCT]: ...

#
@overload
def companion(a: _JustInt1d) -> _Int2d: ...
@overload
def companion(a: _JustFloat1d) -> _Float2d: ...
@overload
def companion(a: _JustComplex1d) -> _Complex2d: ...
@overload
def companion(a: _ToArray1d[_SCT]) -> _Array2d[_SCT]: ...

#
def helmert(n: onp.ToInt, full: bool = False) -> _Float2d: ...

#
def hilbert(n: onp.ToInt) -> _Float2d: ...

#
@overload
def fiedler(a: _JustInt1d) -> _Int2d: ...
@overload
def fiedler(a: _JustFloat1d) -> _Float2d: ...
@overload
def fiedler(a: _JustComplex1d) -> _Complex2d: ...
@overload
def fiedler(a: _ToArray1d[_SCT]) -> _Array2d[_SCT]: ...

#
@overload
def fiedler_companion(a: _JustInt1d) -> _Int2d: ...
@overload
def fiedler_companion(a: _JustFloat1d) -> _Float2d: ...
@overload
def fiedler_companion(a: _JustComplex1d) -> _Complex2d: ...
@overload
def fiedler_companion(a: _ToArray1d[_SCT]) -> _Array2d[_SCT]: ...

# TODO
@overload
def convolution_matrix(a: _JustInt1d, n: onp.ToInt, mode: CorrelateMode = "full") -> _Int2d: ...
@overload
def convolution_matrix(a: _JustFloat1d, n: onp.ToInt, mode: CorrelateMode = "full") -> _Float2d: ...
@overload
def convolution_matrix(a: _JustComplex1d, n: onp.ToInt, mode: CorrelateMode = "full") -> _Complex2d: ...
@overload
def convolution_matrix(a: _ToArray1d[_SCT], n: onp.ToInt, mode: CorrelateMode = "full") -> _Array2d[_SCT]: ...

#
@overload
def invhilbert(n: onp.ToInt, exact: Literal[False] = False) -> _Float2d: ...
@overload
def invhilbert(n: onp.ToInt, exact: Literal[True]) -> _Array2d[np.int64] | _Array2d[np.object_]: ...

#
@overload
def pascal(n: onp.ToInt, kind: _Kind = "symmetric", exact: Literal[True] = True) -> _Array2d[np.uint64 | np.object_]: ...
@overload
def pascal(n: onp.ToInt, kind: _Kind = "symmetric", *, exact: Literal[False]) -> _Float2d: ...
@overload
def pascal(n: onp.ToInt, kind: _Kind, exact: Literal[False]) -> _Float2d: ...

#
@overload
def invpascal(n: onp.ToInt, kind: _Kind = "symmetric", exact: Literal[True] = True) -> _Array2d[np.int64 | np.object_]: ...
@overload
def invpascal(n: onp.ToInt, kind: _Kind = "symmetric", *, exact: Literal[False]) -> _Float2d: ...
@overload
def invpascal(n: onp.ToInt, kind: _Kind, exact: Literal[False]) -> _Float2d: ...

#
def dft(n: onp.ToInt, scale: Literal["sqrtn", "n"] | None = None) -> _Complex2d: ...
