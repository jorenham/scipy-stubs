from collections.abc import Sequence
from typing import Literal as L, TypeAlias, overload
from typing_extensions import TypeVar, deprecated

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
import optype.numpy.compat as npc
import optype.typing as opt
from scipy._typing import ConvMode

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

_SCT = TypeVar("_SCT", bound=np.generic, default=npc.number | np.bool_ | np.object_)

_Kind: TypeAlias = L["symmetric", "upper", "lower"]

_Array2ND: TypeAlias = onp.Array[onp.AtLeast2D, _SCT]
_Array3ND: TypeAlias = onp.Array[onp.AtLeast3D, _SCT]

_Int2D: TypeAlias = onp.Array2D[np.int_]
_Int3ND: TypeAlias = _Array3ND[np.int_]
_Float2D: TypeAlias = onp.Array2D[np.float64]
_Float3ND: TypeAlias = _Array3ND[np.float64]
_Complex2D: TypeAlias = onp.Array2D[np.complex128]
_Complex3ND: TypeAlias = _Array3ND[np.complex128]

_To0D: TypeAlias = _SCT | onp.CanArray0D[_SCT]
_To1D: TypeAlias = Sequence[_To0D[_SCT]] | onp.CanArrayND[_SCT]
_ToStrict1D: TypeAlias = Sequence[_To0D[_SCT]] | onp.CanArray1D[_SCT]
_ToStrict2ND: TypeAlias = onp.SequenceND[_To1D[_SCT]] | onp.CanArrayND[_SCT, onp.AtLeast2D]
_ToND: TypeAlias = onp.SequenceND[_To0D[_SCT]] | onp.SequenceND[_To1D[_SCT]] | onp.CanArrayND[_SCT]
_ToDType: TypeAlias = type[_SCT] | np.dtype[_SCT] | onp.HasDType[np.dtype[_SCT]]

_ToJustIntStrict2ND: TypeAlias = onp.SequenceND[onp.ToJustInt1D] | onp.CanArrayND[npc.integer, onp.AtLeast2D]
_ToJustFloatStrict2ND: TypeAlias = onp.SequenceND[onp.ToJustFloat1D] | onp.CanArrayND[npc.floating, onp.AtLeast2D]
_ToJustComplexStrict2ND: TypeAlias = onp.SequenceND[onp.ToJustComplex1D] | onp.CanArrayND[npc.complexfloating, onp.AtLeast2D]

###

#
@overload
@deprecated("`kron` has been deprecated in favour of `numpy.kron` in SciPy 1.15.0 and will be removed in SciPy 1.17.0.")
def kron(a: onp.Array2D[_SCT], b: onp.Array2D[_SCT]) -> onp.Array2D[_SCT]: ...
@overload
@deprecated("`kron` has been deprecated in favour of `numpy.kron` in SciPy 1.15.0 and will be removed in SciPy 1.17.0.")
def kron(a: onp.ArrayND[_SCT], b: onp.ArrayND[_SCT]) -> onp.Array[onp.AtLeast2D, _SCT]: ...

#
@overload
def toeplitz(c: onp.ToJustInt1D, r: onp.ToJustInt1D | None = None) -> _Int2D: ...
@overload
def toeplitz(c: onp.ToJustFloat1D, r: onp.ToJustFloat1D | None = None) -> _Float2D: ...
@overload
def toeplitz(c: onp.ToJustComplex1D, r: onp.ToJustComplex1D | None = None) -> _Complex2D: ...
@overload
def toeplitz(c: _ToStrict1D[_SCT], r: _ToStrict1D[_SCT] | None = None) -> onp.Array2D[_SCT]: ...
@overload
@deprecated(
    "Beginning in SciPy 1.17, multidimensional input will be treated as a batch, not `ravel`ed. "
    "To preserve the existing behavior and silence this warning, `ravel` aruments before passing them to `toeplitz`.",
    category=FutureWarning,
)
def toeplitz(c: _ToJustIntStrict2ND, r: _ToJustIntStrict2ND | None = None) -> _Int2D: ...
@overload
@deprecated(
    "Beginning in SciPy 1.17, multidimensional input will be treated as a batch, not `ravel`ed. "
    "To preserve the existing behavior and silence this warning, `ravel` aruments before passing them to `toeplitz`.",
    category=FutureWarning,
)
def toeplitz(c: _ToJustFloatStrict2ND, r: _ToJustFloatStrict2ND | None = None) -> _Float2D: ...
@overload
@deprecated(
    "Beginning in SciPy 1.17, multidimensional input will be treated as a batch, not `ravel`ed. "
    "To preserve the existing behavior and silence this warning, `ravel` aruments before passing them to `toeplitz`.",
    category=FutureWarning,
)
def toeplitz(c: _ToJustComplexStrict2ND, r: _ToJustComplexStrict2ND | None = None) -> _Complex2D: ...
@overload
@deprecated(
    "Beginning in SciPy 1.17, multidimensional input will be treated as a batch, not `ravel`ed. "
    "To preserve the existing behavior and silence this warning, `ravel` aruments before passing them to `toeplitz`.",
    category=FutureWarning,
)
def toeplitz(c: _ToStrict2ND[_SCT], r: _ToStrict2ND[_SCT] | None = None) -> onp.Array2D[_SCT]: ...
@overload
def toeplitz(c: _To1D[_SCT], r: _To1D[_SCT] | None = None) -> onp.Array2D[_SCT]: ...

#
@overload
def circulant(c: onp.ToJustIntStrict1D) -> _Int2D: ...
@overload
def circulant(c: _ToJustIntStrict2ND) -> _Int3ND: ...
@overload
def circulant(c: onp.ToJustFloatStrict1D) -> _Float2D: ...
@overload
def circulant(c: _ToJustFloatStrict2ND) -> _Float3ND: ...
@overload
def circulant(c: onp.ToJustComplexStrict1D) -> _Complex2D: ...
@overload
def circulant(c: _ToJustComplexStrict2ND) -> _Complex3ND: ...
@overload
def circulant(c: _ToStrict1D[_SCT]) -> onp.Array2D[_SCT]: ...
@overload
def circulant(c: _ToStrict2ND[_SCT]) -> _Array3ND[_SCT]: ...
@overload
def circulant(c: _ToND[_SCT]) -> _Array2ND[_SCT]: ...

#
@overload
def companion(a: onp.ToJustIntStrict1D) -> _Int2D: ...
@overload
def companion(a: _ToJustIntStrict2ND) -> _Int3ND: ...
@overload
def companion(a: onp.ToJustFloatStrict1D) -> _Float2D: ...
@overload
def companion(a: _ToJustFloatStrict2ND) -> _Float3ND: ...
@overload
def companion(a: onp.ToJustComplexStrict1D) -> _Complex2D: ...
@overload
def companion(a: _ToJustComplexStrict2ND) -> _Complex3ND: ...
@overload
def companion(a: _ToStrict1D[_SCT]) -> onp.Array2D[_SCT]: ...
@overload
def companion(a: _ToStrict2ND[_SCT]) -> _Array3ND[_SCT]: ...
@overload
def companion(a: _ToND[_SCT]) -> _Array2ND[_SCT]: ...

#
@overload
def convolution_matrix(a: onp.ToJustIntStrict1D, n: onp.ToInt, mode: ConvMode = "full") -> _Int2D: ...
@overload
def convolution_matrix(a: _ToJustIntStrict2ND, n: onp.ToInt, mode: ConvMode = "full") -> _Int3ND: ...
@overload
def convolution_matrix(a: onp.ToJustFloatStrict1D, n: onp.ToInt, mode: ConvMode = "full") -> _Float2D: ...
@overload
def convolution_matrix(a: _ToJustFloatStrict2ND, n: onp.ToInt, mode: ConvMode = "full") -> _Float3ND: ...
@overload
def convolution_matrix(a: onp.ToJustComplexStrict1D, n: onp.ToInt, mode: ConvMode = "full") -> _Complex2D: ...
@overload
def convolution_matrix(a: _ToJustComplexStrict2ND, n: onp.ToInt, mode: ConvMode = "full") -> _Complex3ND: ...
@overload
def convolution_matrix(a: _ToStrict1D[_SCT], n: onp.ToInt, mode: ConvMode = "full") -> onp.Array2D[_SCT]: ...
@overload
def convolution_matrix(a: _ToStrict2ND[_SCT], n: onp.ToInt, mode: ConvMode = "full") -> _Array3ND[_SCT]: ...
@overload
def convolution_matrix(a: _ToND[_SCT], n: onp.ToInt, mode: ConvMode = "full") -> _Array2ND[_SCT]: ...

#
@overload
def fiedler(a: onp.ToJustIntStrict1D) -> _Int2D: ...
@overload
def fiedler(a: _ToJustIntStrict2ND) -> _Int3ND: ...
@overload
def fiedler(a: onp.ToJustFloatStrict1D) -> _Float2D: ...
@overload
def fiedler(a: _ToJustFloatStrict2ND) -> _Float3ND: ...
@overload
def fiedler(a: onp.ToJustComplexStrict1D) -> _Complex2D: ...
@overload
def fiedler(a: _ToJustComplexStrict2ND) -> _Complex3ND: ...
@overload
def fiedler(a: _ToStrict1D[_SCT]) -> onp.Array2D[_SCT]: ...
@overload
def fiedler(a: _ToStrict2ND[_SCT]) -> _Array3ND[_SCT]: ...
@overload
def fiedler(a: _ToND[_SCT]) -> _Array2ND[_SCT]: ...

#
@overload
def fiedler_companion(a: onp.ToJustIntStrict1D) -> _Int2D: ...
@overload
def fiedler_companion(a: _ToJustIntStrict2ND) -> _Int3ND: ...
@overload
def fiedler_companion(a: onp.ToJustFloatStrict1D) -> _Float2D: ...
@overload
def fiedler_companion(a: _ToJustFloatStrict2ND) -> _Float3ND: ...
@overload
def fiedler_companion(a: onp.ToJustComplexStrict1D) -> _Complex2D: ...
@overload
def fiedler_companion(a: _ToJustComplexStrict2ND) -> _Complex3ND: ...
@overload
def fiedler_companion(a: _ToStrict1D[_SCT]) -> onp.Array2D[_SCT]: ...
@overload
def fiedler_companion(a: _ToStrict2ND[_SCT]) -> _Array3ND[_SCT]: ...
@overload
def fiedler_companion(a: _ToND[_SCT]) -> _Array2ND[_SCT]: ...

# TODO(jorenham): batching
@overload
def leslie(f: onp.ToJustIntStrict1D, s: onp.ToJustIntStrict1D) -> _Int2D: ...
@overload
def leslie(f: onp.ToJustIntND, s: _ToJustIntStrict2ND) -> _Int3ND: ...
@overload
def leslie(f: _ToJustIntStrict2ND, s: onp.ToJustIntND) -> _Int3ND: ...
@overload
def leslie(f: onp.ToJustFloatStrict1D, s: onp.ToJustFloatStrict1D) -> _Float2D: ...
@overload
def leslie(f: onp.ToJustFloatND, s: _ToJustFloatStrict2ND) -> _Float3ND: ...
@overload
def leslie(f: _ToJustFloatStrict2ND, s: onp.ToJustFloatND) -> _Float3ND: ...
@overload
def leslie(f: onp.ToJustComplexStrict1D, s: onp.ToJustComplexStrict1D) -> _Complex2D: ...
@overload
def leslie(f: onp.ToJustComplexND, s: _ToJustComplexStrict2ND) -> _Complex3ND: ...
@overload
def leslie(f: _ToJustComplexStrict2ND, s: onp.ToJustComplexND) -> _Complex3ND: ...
@overload
def leslie(f: _ToStrict1D[_SCT], s: _ToStrict1D[_SCT]) -> onp.Array2D[_SCT]: ...
@overload
def leslie(f: _ToStrict1D[_SCT], s: _ToStrict2ND[_SCT]) -> _Array3ND[_SCT]: ...
@overload
def leslie(f: _ToStrict2ND[_SCT], s: _ToND[_SCT]) -> _Array3ND[_SCT]: ...
@overload
def leslie(f: _ToND[_SCT], s: _ToND[_SCT]) -> _Array2ND[_SCT]: ...

#
@overload
def block_diag() -> _Float2D: ...  # shape=(1, 0)
@overload
def block_diag(arr0: onp.ToJustInt1D, /, *arrs: onp.ToJustInt1D) -> _Int2D: ...
@overload
def block_diag(arr0: onp.ToJustFloat1D, /, *arrs: onp.ToJustFloat1D) -> _Float2D: ...
@overload
def block_diag(arr0: onp.ToJustComplex1D, /, *arrs: onp.ToJustComplex1D) -> _Complex2D: ...
@overload
def block_diag(arr0: _To1D[_SCT], /, *arrs: _To1D[_SCT]) -> onp.Array2D[_SCT]: ...

#
def dft(n: onp.ToInt, scale: L["sqrtn", "n"] | None = None) -> _Complex2D: ...

#
@overload
def hadamard(n: onp.ToInt, dtype: type[opt.JustInt]) -> _Int2D: ...
@overload
def hadamard(n: onp.ToInt, dtype: type[opt.JustFloat]) -> _Float2D: ...
@overload
def hadamard(n: onp.ToInt, dtype: type[opt.JustComplex]) -> _Complex2D: ...
@overload
def hadamard(n: onp.ToInt, dtype: _ToDType[_SCT]) -> onp.Array2D[_SCT]: ...
@overload
def hadamard(n: onp.ToInt, dtype: npt.DTypeLike = ...) -> onp.Array2D: ...

#
@overload
def hankel(c: onp.ToJustInt1D, r: onp.ToJustInt1D | None = None) -> _Int2D: ...
@overload
def hankel(c: onp.ToJustFloat1D, r: onp.ToJustFloat1D | None = None) -> _Float2D: ...
@overload
def hankel(c: onp.ToJustComplex1D, r: onp.ToJustComplex1D | None = None) -> _Complex2D: ...
@overload
def hankel(c: _To1D[_SCT], r: _To1D[_SCT] | None = None) -> onp.Array2D[_SCT]: ...

#
def helmert(n: onp.ToInt, full: bool = False) -> _Float2D: ...

#
def hilbert(n: onp.ToInt) -> _Float2D: ...

#
@overload
def invhilbert(n: onp.ToInt, exact: L[False] = False) -> _Float2D: ...
@overload
def invhilbert(n: onp.ToInt, exact: L[True]) -> onp.Array2D[np.int64 | np.object_]: ...

#
@overload
def pascal(n: onp.ToInt, kind: _Kind = "symmetric", exact: L[True] = True) -> onp.Array2D[np.int64 | np.object_]: ...
@overload
def pascal(n: onp.ToInt, kind: _Kind, exact: L[False]) -> _Float2D: ...
@overload
def pascal(n: onp.ToInt, kind: _Kind = "symmetric", *, exact: L[False]) -> _Float2D: ...

#
@overload
def invpascal(n: onp.ToInt, kind: _Kind = "symmetric", exact: L[True] = True) -> onp.Array2D[np.int64 | np.object_]: ...
@overload
def invpascal(n: onp.ToInt, kind: _Kind, exact: L[False]) -> _Float2D: ...
@overload
def invpascal(n: onp.ToInt, kind: _Kind = "symmetric", *, exact: L[False]) -> _Float2D: ...
