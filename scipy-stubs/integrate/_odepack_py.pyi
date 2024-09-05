from collections.abc import Sequence
from typing import Any, Literal, Protocol, overload, type_check_only
from typing_extensions import TypeVar, TypeVarTuple, Unpack

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from ._typing import ODEInfoDict

__all__ = ["ODEintWarning", "odeint"]

_Ts = TypeVarTuple("_Ts", default=Unpack[tuple[()]])
_YT = TypeVar("_YT", bound=onpt.AnyFloatingArray | Sequence[float] | float)

@type_check_only
class _ODEFunc(Protocol[_YT, Unpack[_Ts]]):
    def __call__(self, y: _YT, t: float, /, *args: Unpack[_Ts]) -> _YT: ...

@type_check_only
class _ODEFuncInv(Protocol[_YT, Unpack[_Ts]]):
    def __call__(self, t: float, y: _YT, /, *args: Unpack[_Ts]) -> _YT: ...

class ODEintWarning(Warning): ...

# unspecified args
@overload
def odeint(
    func: _ODEFunc[_YT],
    y0: _YT,
    t: npt.NDArray[np.integer[Any]] | Sequence[int],
    args: tuple[()] = (),
    Dfun: _ODEFunc[_YT] | None = None,
    col_deriv: Literal[False, 0, True, 1] = 0,
    full_output: Literal[False, 0, None] = 0,
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: npt.NDArray[np.integer[Any] | np.floating[Any]] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[False, 0, True, 1] = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[False, 0, True, 1] = 0,
    tfirst: Literal[False, 0, None] = False,
) -> onpt.Array[tuple[int, int], np.floating[Any]]: ...
@overload
def odeint(
    func: _ODEFunc[_YT],
    y0: _YT,
    t: npt.NDArray[np.integer[Any]] | Sequence[int],
    args: tuple[()] = (),
    Dfun: _ODEFunc[_YT] | None = None,
    col_deriv: Literal[False, 0, True, 1] = 0,
    full_output: Literal[False, 0, None] = 0,
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: npt.NDArray[np.integer[Any] | np.floating[Any]] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[False, 0, True, 1] = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[False, 0, True, 1] = 0,
    *,
    tfirst: Literal[True, 1],
) -> onpt.Array[tuple[int, int], np.floating[Any]]: ...
@overload
def odeint(
    func: _ODEFunc[_YT],
    y0: _YT,
    t: npt.NDArray[np.integer[Any]] | Sequence[int],
    args: tuple[()] = (),
    Dfun: _ODEFunc[_YT] | None = None,
    col_deriv: Literal[False, 0, True, 1] = 0,
    *,
    full_output: Literal[True, 1],
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: npt.NDArray[np.integer[Any] | np.floating[Any]] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[False, 0, True, 1] = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[False, 0, True, 1] = 0,
    tfirst: Literal[False, 0, None] = False,
) -> tuple[onpt.Array[tuple[int, int], np.floating[Any]], ODEInfoDict]: ...
@overload
def odeint(
    func: _ODEFunc[_YT],
    y0: _YT,
    t: npt.NDArray[np.integer[Any]] | Sequence[int],
    args: tuple[()] = (),
    Dfun: _ODEFunc[_YT] | None = None,
    col_deriv: Literal[False, 0, True, 1] = 0,
    *,
    full_output: Literal[True, 1],
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: npt.NDArray[np.integer[Any] | np.floating[Any]] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[False, 0, True, 1] = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[False, 0, True, 1] = 0,
    tfirst: Literal[True, 1],
) -> tuple[onpt.Array[tuple[int, int], np.floating[Any]], ODEInfoDict]: ...

# specified args
@overload
def odeint(
    func: _ODEFunc[_YT, Unpack[_Ts]],
    y0: _YT,
    t: npt.NDArray[np.integer[Any]] | Sequence[int],
    args: tuple[Unpack[_Ts]] = ...,
    Dfun: _ODEFunc[_YT, Unpack[_Ts]] | None = None,
    col_deriv: Literal[False, 0, True, 1] = 0,
    full_output: Literal[False, 0, None] = 0,
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: npt.NDArray[np.integer[Any] | np.floating[Any]] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[False, 0, True, 1] = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[False, 0, True, 1] = 0,
    tfirst: Literal[False, 0, None] = False,
) -> onpt.Array[tuple[int, int], np.floating[Any]]: ...
@overload
def odeint(
    func: _ODEFuncInv[_YT, Unpack[_Ts]],
    y0: _YT,
    t: npt.NDArray[np.integer[Any]] | Sequence[int],
    args: tuple[Unpack[_Ts]] = ...,
    Dfun: _ODEFuncInv[_YT, Unpack[_Ts]] | None = None,
    col_deriv: Literal[False, 0, True, 1] = 0,
    full_output: Literal[False, 0, None] = 0,
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: npt.NDArray[np.integer[Any] | np.floating[Any]] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[False, 0, True, 1] = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[False, 0, True, 1] = 0,
    *,
    tfirst: Literal[True, 1],
) -> onpt.Array[tuple[int, int], np.floating[Any]]: ...
@overload
def odeint(
    func: _ODEFunc[_YT, Unpack[_Ts]],
    y0: _YT,
    t: npt.NDArray[np.integer[Any]] | Sequence[int],
    args: tuple[Unpack[_Ts]] = ...,
    Dfun: _ODEFunc[_YT, Unpack[_Ts]] | None = None,
    col_deriv: Literal[False, 0, True, 1] = 0,
    *,
    full_output: Literal[True, 1],
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: npt.NDArray[np.integer[Any] | np.floating[Any]] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[False, 0, True, 1] = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[False, 0, True, 1] = 0,
    tfirst: Literal[False, 0, None] = False,
) -> tuple[onpt.Array[tuple[int, int], np.floating[Any]], ODEInfoDict]: ...
@overload
def odeint(
    func: _ODEFuncInv[_YT, Unpack[_Ts]],
    y0: _YT,
    t: npt.NDArray[np.integer[Any]] | Sequence[int],
    args: tuple[Unpack[_Ts]] = ...,
    Dfun: _ODEFuncInv[_YT, Unpack[_Ts]] | None = None,
    col_deriv: Literal[False, 0, True, 1] = 0,
    *,
    full_output: Literal[True, 1],
    ml: int | None = None,
    mu: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    tcrit: npt.NDArray[np.integer[Any] | np.floating[Any]] | None = None,
    h0: float = 0.0,
    hmax: float = 0.0,
    hmin: float = 0.0,
    ixpr: Literal[False, 0, True, 1] = 0,
    mxstep: int = 0,
    mxhnil: int = 0,
    mxordn: int = 12,
    mxords: int = 5,
    printmessg: Literal[False, 0, True, 1] = 0,
    tfirst: Literal[True, 1],
) -> tuple[onpt.Array[tuple[int, int], np.floating[Any]], ODEInfoDict]: ...
