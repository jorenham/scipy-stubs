from typing import Literal, TypeAlias, overload

import numpy as np
import optype.numpy as onp

__all__ = ["qr_delete", "qr_insert", "qr_update"]

_Float2D: TypeAlias = onp.Array2D[np.float32 | np.float64]
_FloatQR: TypeAlias = tuple[_Float2D, _Float2D]

_Complex2D: TypeAlias = onp.Array2D[np.complex64 | np.complex128]
_ComplexQR: TypeAlias = _FloatQR | tuple[_Complex2D, _Complex2D]

_Which: TypeAlias = Literal["row", "col"]

###

@overload
def qr_delete(
    Q: onp.ToFloat2D,
    R: onp.ToFloat2D,
    k: onp.ToJustInt,
    p: onp.ToJustInt = 1,
    which: _Which = "row",
    overwrite_qr: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _FloatQR: ...
@overload
def qr_delete(
    Q: onp.ToComplex2D,
    R: onp.ToComplex2D,
    k: onp.ToJustInt,
    p: onp.ToJustInt = 1,
    which: _Which = "row",
    overwrite_qr: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _ComplexQR: ...

#
@overload
def qr_insert(
    Q: onp.ToFloat2D,
    R: onp.ToFloat2D,
    u: onp.ToFloat1D | onp.ToFloat2D,
    k: onp.ToJustInt,
    which: _Which = "row",
    rcond: onp.ToFloat | None = None,
    overwrite_qru: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _FloatQR: ...
@overload
def qr_insert(
    Q: onp.ToComplex2D,
    R: onp.ToComplex2D,
    u: onp.ToComplex1D | onp.ToComplex2D,
    k: onp.ToJustInt,
    which: _Which = "row",
    rcond: onp.ToFloat | None = None,
    overwrite_qru: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _ComplexQR: ...

#
@overload
def qr_update(
    Q: onp.ToFloat2D,
    R: onp.ToFloat2D,
    u: onp.ToFloat1D | onp.ToFloat2D,
    v: onp.ToFloat1D | onp.ToFloat2D,
    overwrite_qruv: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _FloatQR: ...
@overload
def qr_update(
    Q: onp.ToComplex2D,
    R: onp.ToComplex2D,
    u: onp.ToComplex1D | onp.ToComplex2D,
    v: onp.ToComplex1D | onp.ToComplex2D,
    overwrite_qruv: onp.ToBool = False,
    check_finite: onp.ToBool = True,
) -> _ComplexQR: ...
