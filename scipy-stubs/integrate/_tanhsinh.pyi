from collections.abc import Callable
from typing import Concatenate, Final, Generic, Literal, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onpt
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeNumber_co
from scipy._lib._util import _RichResult
from scipy._typing import AnyBool, AnyInt, AnyReal

_SCT = TypeVar("_SCT")

@type_check_only
class _TanhSinhResult(_RichResult[bool | int | _SCT], Generic[_SCT]):
    success: Final[bool]
    status: Final[Literal[0, -1, -2, -3, -4, 1]]
    integral: _SCT
    error: Final[np.float64]
    maxlevel: Final[int]
    nfev: Final[int]

@type_check_only
class _NSumResult(_RichResult[bool | int | _SCT], Generic[_SCT]):
    success: Final[bool]
    status: Final[Literal[0, -1, -2, -3]]
    sum: _SCT
    error: Final[np.float64]
    maxlevel: Final[int]
    nfev: Final[int]

###

#
@overload
def _tanhsinh(
    f: Callable[Concatenate[onpt.Array[tuple[int], np.float64], ...], _ArrayLikeFloat_co],
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    *,
    args: tuple[object, ...] = (),
    log: AnyBool = False,
    maxfun: AnyInt | None = None,
    maxlevel: AnyInt | None = None,
    minlevel: AnyInt | None = 2,
    atol: AnyReal | None = None,
    rtol: AnyReal | None = None,
    preserve_shape: bool = False,
    callback: Callable[[_TanhSinhResult[np.float64]], None] | None = None,
) -> _TanhSinhResult[np.float64]: ...
@overload
def _tanhsinh(
    f: Callable[Concatenate[onpt.Array[tuple[int], np.float64 | np.complex128], ...], _ArrayLikeNumber_co],
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co,
    *,
    args: tuple[object, ...] = (),
    log: AnyBool = False,
    maxfun: AnyInt | None = None,
    maxlevel: AnyInt | None = None,
    minlevel: AnyInt | None = 2,
    atol: AnyReal | None = None,
    rtol: AnyReal | None = None,
    preserve_shape: bool = False,
    callback: Callable[[_TanhSinhResult[np.float64 | np.complex128]], None] | None = None,
) -> _TanhSinhResult[np.float64 | np.complex128]: ...

#
@overload
def _nsum(
    f: Callable[Concatenate[onpt.Array[tuple[int], np.float64], ...], _ArrayLikeFloat_co],
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    step: _ArrayLikeFloat_co = 1,
    args: tuple[object, ...] = (),
    log: AnyBool = False,
    maxterms: AnyInt = 0x10_00_00,
    atol: AnyReal | None = None,
    rtol: AnyReal | None = None,
) -> _NSumResult[np.float64]: ...
@overload
def _nsum(
    f: Callable[Concatenate[onpt.Array[tuple[int], np.float64 | np.complex128], ...], _ArrayLikeNumber_co],
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co,
    step: _ArrayLikeFloat_co = 1,
    args: tuple[object, ...] = (),
    log: AnyBool = False,
    maxterms: AnyInt = 0x10_00_00,
    atol: AnyReal | None = None,
    rtol: AnyReal | None = None,
) -> _NSumResult[np.float64 | np.complex128]: ...
