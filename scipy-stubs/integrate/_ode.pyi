from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Final, Generic, Literal, Protocol, TypeAlias, type_check_only
from typing_extensions import Self, TypeVarTuple, Unpack, override

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from scipy._typing import Untyped, UntypedCallable, UntypedTuple

__all__ = ["complex_ode", "ode"]

_Ts = TypeVarTuple("_Ts", default=Unpack[tuple[()]])

@type_check_only
class _ODEFuncF(Protocol[Unpack[_Ts]]):
    def __call__(
        self,
        t: float,
        y: float | npt.NDArray[np.float64],
        /,
        *args: Unpack[_Ts],
    ) -> float | npt.NDArray[np.floating[Any]]: ...

@type_check_only
class _ODEFuncC(Protocol[Unpack[_Ts]]):
    def __call__(
        self,
        t: float,
        y: complex | npt.NDArray[np.complex128],
        /,
        *args: Unpack[_Ts],
    ) -> complex | npt.NDArray[np.complexfloating[Any, Any]]: ...

_SolOutFunc: TypeAlias = Callable[[float, onpt.Array[tuple[int], np.inexact[Any]]], Literal[0, -1]]

class ode(Generic[Unpack[_Ts]]):
    stiff: int
    f: _ODEFuncF[Unpack[_Ts]]
    jac: _ODEFuncF[Unpack[_Ts]] | None
    f_params: tuple[()] | tuple[Unpack[_Ts]]
    jac_params: tuple[()] | tuple[Unpack[_Ts]]
    t: float
    def __init__(self, /, f: _ODEFuncF[Unpack[_Ts]], jac: _ODEFuncF[Unpack[_Ts]] | None = None) -> None: ...
    @property
    def y(self, /) -> float: ...
    def integrate(self, /, t: float, step: bool = False, relax: bool = False) -> float: ...
    def set_initial_value(self, /, y: onpt.AnyInexactArray | Sequence[Sequence[complex]] | complex, t: float = 0.0) -> Self: ...
    def set_integrator(self, /, name: str, **integrator_params: Untyped) -> Self: ...
    def set_f_params(self, /, *args: Unpack[_Ts]) -> Self: ...
    def set_jac_params(self, /, *args: Unpack[_Ts]) -> Self: ...
    def set_solout(self, /, solout: _SolOutFunc) -> None: ...
    def get_return_code(self, /) -> Literal[-7, -6, -5, -4, -3, -2, -1, 1, 2]: ...
    def successful(self, /) -> bool: ...

class complex_ode(ode[Unpack[_Ts]], Generic[Unpack[_Ts]]):
    cf: _ODEFuncC[Unpack[_Ts]]
    cjac: _ODEFuncC[Unpack[_Ts]] | None
    tmp: onpt.Array[tuple[int], np.float64]
    def __init__(self, /, f: _ODEFuncC[Unpack[_Ts]], jac: _ODEFuncC[Unpack[_Ts]] | None = None) -> None: ...
    @property
    @override
    def y(self, /) -> complex: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def integrate(self, /, t: float, step: bool = False, relax: bool = False) -> complex: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

def find_integrator(name: str) -> type[IntegratorBase] | None: ...

class IntegratorConcurrencyError(RuntimeError):
    def __init__(self, /, name: str) -> None: ...

class IntegratorBase:
    runner: ClassVar[Callable[..., object] | None]  # fortran function or unavailable
    supports_run_relax: ClassVar[Literal[0, 1, None]] = None
    supports_step: ClassVar[Literal[0, 1, None]] = None
    supports_solout: ClassVar[bool]
    scalar: ClassVar[type] = ...
    handle: ClassVar[int]
    success: Literal[0, 1] | None = None
    integrator_classes: list[type[IntegratorBase]]
    istate: int | None = None
    def acquire_new_handle(self, /) -> None: ...
    def check_handle(self, /) -> None: ...
    def reset(self, /, n: int, has_jac: bool) -> None: ...
    def run(
        self,
        /,
        f: UntypedCallable,
        jac: UntypedCallable,
        y0: complex,
        t0: float,
        t1: float,
        f_params: UntypedTuple,
        jac_params: UntypedTuple,
    ) -> tuple[Untyped, float]: ...
    def step(
        self,
        /,
        f: UntypedCallable,
        jac: UntypedCallable,
        y0: complex,
        t0: float,
        t1: float,
        f_params: UntypedTuple,
        jac_params: UntypedTuple,
    ) -> tuple[Untyped, float]: ...
    def run_relax(
        self,
        /,
        f: UntypedCallable,
        jac: UntypedCallable,
        y0: complex,
        t0: float,
        t1: float,
        f_params: UntypedTuple,
        jac_params: UntypedTuple,
    ) -> tuple[Untyped, float]: ...

class vode(IntegratorBase):
    messages: ClassVar[dict[int, str]]
    active_global_handle: int
    meth: int
    with_jacobian: Untyped
    rtol: Untyped
    atol: Untyped
    mu: Untyped
    ml: Untyped
    order: Untyped
    nsteps: Untyped
    max_step: Untyped
    min_step: Untyped
    first_step: Untyped
    initialized: bool
    rwork: Untyped
    iwork: Untyped
    call_args: Untyped
    def __init__(
        self,
        /,
        method: str = "adams",
        with_jacobian: bool = False,
        rtol: float = 1e-06,
        atol: float = 1e-12,
        lband: Untyped | None = None,
        uband: Untyped | None = None,
        order: int = 12,
        nsteps: int = 500,
        max_step: float = 0.0,
        min_step: float = 0.0,
        first_step: float = 0.0,
    ) -> None: ...

class zvode(vode):
    active_global_handle: int
    zwork: Untyped
    rwork: Untyped
    iwork: Untyped
    call_args: Untyped
    initialized: bool

class dopri5(IntegratorBase):
    name: ClassVar = "dopri5"
    messages: ClassVar[dict[int, str]]

    rtol: Final[float]
    atol: Final[float]
    nsteps: Final[int]
    max_step: Final[float]
    first_step: Final[float]
    safety: Final[float]
    ifactor: Final[float]
    dfactor: Final[float]
    beta: Final[float]
    verbosity: Final[int]
    solout: Callable[[float, onpt.Array[tuple[int], np.inexact[Any]]], Literal[0, -1]] | None
    solout_cmplx: bool
    iout: int
    work: onpt.Array[tuple[int], np.float64]
    iwork: onpt.Array[tuple[int], np.int32]
    call_args: Untyped
    def __init__(
        self,
        /,
        rtol: float = 1e-06,
        atol: float = 1e-12,
        nsteps: int = 500,
        max_step: float = 0.0,
        first_step: float = 0.0,
        safety: float = 0.9,
        ifactor: float = 10.0,
        dfactor: float = 0.2,
        beta: float = 0.0,
        method: Untyped | None = None,
        verbosity: int = -1,
    ) -> None: ...
    def set_solout(self, solout: _SolOutFunc | None, complex: bool = False) -> None: ...

class dop853(dopri5):
    name: ClassVar = "dop853"
    def __init__(
        self,
        rtol: float = 1e-06,
        atol: float = 1e-12,
        nsteps: int = 500,
        max_step: float = 0.0,
        first_step: float = 0.0,
        safety: float = 0.9,
        ifactor: float = 6.0,
        dfactor: float = 0.3,
        beta: float = 0.0,
        method: Untyped | None = None,
        verbosity: int = -1,
    ) -> None: ...
    call_args: Untyped

class lsoda(IntegratorBase):
    active_global_handle: ClassVar[int] = 0
    messages: ClassVar[dict[int, str]]

    with_jacobian: Final[bool]
    rtol: Final[float]
    atol: Final[float]
    mu: Final[float | None]
    ml: Final[float | None]
    max_order_ns: Final[int]
    max_order_s: Final[int]
    nsteps: Final[int]
    max_step: Final[float]
    min_step: Final[float]
    first_step: Final[float]
    ixpr: Final[int]
    max_hnil: Final[int]
    initialized: Final[bool]
    rwork: onpt.Array[tuple[int], np.float64]
    iwork: onpt.Array[tuple[int], np.int32]
    call_args: list[float | onpt.Array[tuple[int], np.float64] | onpt.Array[tuple[int], np.int32]]
    def __init__(
        self,
        /,
        with_jacobian: bool = False,
        rtol: float = 1e-06,
        atol: float = 1e-12,
        lband: float | None = None,
        uband: float | None = None,
        nsteps: int = 500,
        max_step: float = 0.0,
        min_step: float = 0.0,
        first_step: float = 0.0,
        ixpr: int = 0,
        max_hnil: int = 0,
        max_order_ns: int = 12,
        max_order_s: int = 5,
        method: Untyped | None = None,
    ) -> None: ...
