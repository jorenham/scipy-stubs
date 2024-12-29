from typing import Any, Literal as L, TypeAlias
from typing_extensions import assert_type

import numpy as np
import optype.numpy as onp
import scipy.special as sp

_Float32ND: TypeAlias = onp.ArrayND[np.float32]
_Float64ND: TypeAlias = onp.ArrayND[np.float64]
_Complex64ND: TypeAlias = onp.ArrayND[np.complex64]
_Complex128ND: TypeAlias = onp.ArrayND[np.complex128]

_b1: np.bool_
_i: np.integer[Any]
_f: np.floating[Any]
_f2: np.float16
_f4: np.float32
_f8: np.float64
_c8: np.complex64
_c16: np.complex128

_b1_nd: onp.ArrayND[np.bool_]
_i1_nd: onp.ArrayND[np.uint8 | np.int8]
_f2_nd: onp.ArrayND[np.float16]
_f4_nd: _Float32ND
_f8_nd: _Float64ND
_c8_nd: _Complex64ND
_c16_nd: _Complex128ND

# NOTE: `[c]longdouble` can't be tested, because it types as `floating[Any]` on `numpy<2.2`

# _UFunc
assert_type(sp.cbrt.__name__, L["cbrt"])
assert_type(sp.cbrt.identity, L[0])

# _UFunc11
assert_type(sp.cbrt.nin, L[1])
assert_type(sp.cbrt.nout, L[1])
assert_type(sp.cbrt.nargs, L[2])
assert_type(sp.cbrt.ntypes, L[2])
assert_type(sp.cbrt.types, list[L["f->f", "d->d"]])
assert_type(sp.exprel.identity, None)

# _UFunc11f
assert_type(sp.cbrt(_b1), np.float64)
assert_type(sp.cbrt(_b1_nd), _Float64ND)
assert_type(sp.cbrt(_i), np.float64)
assert_type(sp.cbrt(_i1_nd), _Float64ND)
assert_type(sp.cbrt(_f2), np.float64)
assert_type(sp.cbrt(_f2_nd), _Float64ND)
assert_type(sp.cbrt(_f4), np.float32)
assert_type(sp.cbrt(_f4_nd), _Float32ND)
assert_type(sp.cbrt(_f8), np.float64)
assert_type(sp.cbrt(_f8_nd), _Float64ND)
sp.cbrt(_c16)  # type:ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
sp.cbrt(_c16_nd)  # type:ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(sp.cbrt(False), np.float64)
assert_type(sp.cbrt([False]), _Float64ND)
assert_type(sp.cbrt(0), np.float64)
assert_type(sp.cbrt([0]), _Float64ND)
assert_type(sp.cbrt(0.0), np.float64)
assert_type(sp.cbrt([0.0]), _Float64ND)
sp.cbrt(0j)  # type:ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
sp.cbrt([0j])  # type:ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(sp.cbrt.at(_b1_nd, _i), None)
assert_type(sp.cbrt.at(_f8_nd, _i), None)
sp.cbrt.at(_c16, _i)  # type:ignore[arg-type]  # pyright: ignore[reportArgumentType]

# _UFunc11g
assert_type(sp.logit.ntypes, L[3])
assert_type(sp.logit(_b1), np.float64)
assert_type(sp.logit(_b1_nd), _Float64ND)
assert_type(sp.logit(_f4), np.float32)
assert_type(sp.logit(_f4_nd), _Float32ND)
assert_type(sp.logit(_f8), np.float64)
assert_type(sp.logit(_f8_nd), _Float64ND)
sp.logit(_c16)  # type:ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
sp.logit(_c16_nd)  # type:ignore[arg-type]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(sp.logit(0), np.float64)
assert_type(sp.logit([0]), _Float64ND)
assert_type(sp.logit(0.0), np.float64)
assert_type(sp.logit([0.0]), _Float64ND)
assert_type(sp.logit.at(_b1_nd, _i), None)
assert_type(sp.logit.at(_f8_nd, _i), None)
sp.logit.at(_c16, _i)  # type:ignore[arg-type]  # pyright: ignore[reportArgumentType]

# _UFunc11c - TODO: wofz
# _UFunc11fc - TODO: erf

# _UFunc12 - TODO
# _UFunc14 - TODO

###

# _UFunc21 - TODO
# _UFunc22 - TODO
# _UFunc24 - TODO

###

# _UFunc31 - TODO
# _UFunc32 - TODO

###

# _UFunc41 - TODO + sph_harm deprecation test
# _UFunc42 - TODO

###

# _UFunc52 - TODO
