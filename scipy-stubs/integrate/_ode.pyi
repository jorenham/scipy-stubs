from collections.abc import Callable
from typing import Any, ClassVar, Final, Generic, Literal, Protocol, TypeAlias, TypedDict, type_check_only
from typing_extensions import Self, TypeVar, TypeVarTuple, Unpack, override

import numpy as np
import optype.numpy as onp

__all__ = ["complex_ode", "ode"]

_SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.inexact[Any], default=np.float64 | np.complex128)
_Ts = TypeVarTuple("_Ts", default=Unpack[tuple[()]])

@type_check_only
class _IntegratorParams(TypedDict, total=False):
    with_jacobian: bool
    rtol: float
    atol: float
    lband: float | None
    uband: float | None
    order: int
    nsteps: int
    max_step: float
    min_step: float
    first_step: float
    ixpr: int
    max_hnil: int
    max_order_ns: int
    max_order_s: int
    method: Literal["adams", "bds"] | None
    safety: float
    ifactor: float
    dfactor: float
    beta: float
    verbosity: int

@type_check_only
class _ODEFuncF(Protocol[Unpack[_Ts]]):
    def __call__(
        self,
        t: float,
        y: float | onp.ArrayND[np.float64],
        /,
        *args: Unpack[_Ts],
    ) -> float | onp.ArrayND[np.floating[Any]]: ...

@type_check_only
class _ODEFuncC(Protocol[Unpack[_Ts]]):
    def __call__(
        self,
        t: float,
        y: complex | onp.ArrayND[np.complex128],
        /,
        *args: Unpack[_Ts],
    ) -> complex | onp.ArrayND[np.complexfloating[Any, Any]]: ...

_SolOutFunc: TypeAlias = Callable[[float, onp.Array1D[np.inexact[Any]]], Literal[0, -1]]

###

class ode(Generic[Unpack[_Ts]]):
    stiff: int
    f: _ODEFuncF[Unpack[_Ts]]
    f_params: tuple[()] | tuple[Unpack[_Ts]]
    jac: _ODEFuncF[Unpack[_Ts]] | None
    jac_params: tuple[()] | tuple[Unpack[_Ts]]
    t: float
    def __init__(self, /, f: _ODEFuncF[Unpack[_Ts]], jac: _ODEFuncF[Unpack[_Ts]] | None = None) -> None: ...
    @property
    def y(self, /) -> float: ...
    def integrate(self, /, t: float, step: bool = False, relax: bool = False) -> float: ...
    def set_initial_value(self, /, y: onp.ToComplex | onp.ToComplexND, t: float = 0.0) -> Self: ...
    def set_integrator(self, /, name: str, **integrator_params: Unpack[_IntegratorParams]) -> Self: ...
    def set_f_params(self, /, *args: Unpack[_Ts]) -> Self: ...
    def set_jac_params(self, /, *args: Unpack[_Ts]) -> Self: ...
    def set_solout(self, /, solout: _SolOutFunc) -> None: ...
    def get_return_code(self, /) -> Literal[-7, -6, -5, -4, -3, -2, -1, 1, 2]: ...
    def successful(self, /) -> bool: ...

class complex_ode(ode[Unpack[_Ts]], Generic[Unpack[_Ts]]):
    cf: _ODEFuncC[Unpack[_Ts]]
    cjac: _ODEFuncC[Unpack[_Ts]] | None
    tmp: onp.Array1D[np.float64]
    def __init__(self, /, f: _ODEFuncC[Unpack[_Ts]], jac: _ODEFuncC[Unpack[_Ts]] | None = None) -> None: ...
    @property
    @override
    def y(self, /) -> complex: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def integrate(self, /, t: float, step: bool = False, relax: bool = False) -> complex: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

def find_integrator(name: str) -> type[IntegratorBase] | None: ...

class IntegratorConcurrencyError(RuntimeError):
    def __init__(self, /, name: str) -> None: ...

class IntegratorBase(Generic[_SCT_co]):
    runner: ClassVar[Callable[..., tuple[Any, ...]] | None]  # fortran function or unavailable
    supports_run_relax: ClassVar[Literal[0, 1] | None] = None
    supports_step: ClassVar[Literal[0, 1] | None] = None
    supports_solout: ClassVar[bool] = ...
    scalar: ClassVar[type] = ...

    handle: int
    success: Literal[0, 1] | bool | None = None
    integrator_classes: list[type[IntegratorBase]]
    istate: int | None = None

    def acquire_new_handle(self, /) -> None: ...
    def check_handle(self, /) -> None: ...
    def reset(self, /, n: int, has_jac: bool) -> None: ...
    def run(
        self,
        /,
        f: Callable[..., _SCT_co],
        jac: Callable[..., onp.ArrayND[_SCT_co]] | None,
        y0: complex,
        t0: float,
        t1: float,
        f_params: tuple[object, ...],
        jac_params: tuple[object, ...],
    ) -> tuple[_SCT_co, float]: ...
    def step(
        self,
        /,
        f: Callable[..., _SCT_co],
        jac: Callable[..., onp.ArrayND[_SCT_co]],
        y0: complex,
        t0: float,
        t1: float,
        f_params: tuple[object, ...],
        jac_params: tuple[object, ...],
    ) -> tuple[_SCT_co, float]: ...
    def run_relax(
        self,
        /,
        f: Callable[..., _SCT_co],
        jac: Callable[..., onp.ArrayND[_SCT_co]],
        y0: complex,
        t0: float,
        t1: float,
        f_params: tuple[object, ...],
        jac_params: tuple[object, ...],
    ) -> tuple[_SCT_co, float]: ...

class vode(IntegratorBase[_SCT_co], Generic[_SCT_co]):
    messages: ClassVar[dict[int, str]] = ...

    active_global_handle: int
    meth: int
    with_jacobian: bool
    rtol: float
    atol: float
    mu: float
    ml: float
    order: int
    nsteps: int
    max_step: float
    min_step: float
    first_step: float
    initialized: bool
    rwork: onp.Array1D[np.float64]
    iwork: onp.Array1D[np.int32]
    call_args: list[float | onp.ArrayND[np.float64] | onp.ArrayND[np.int32]]

    def __init__(
        self,
        /,
        method: Literal["adams", "bdf"] = "adams",
        with_jacobian: bool = False,
        rtol: float = 1e-06,
        atol: float = 1e-12,
        lband: float | None = None,
        uband: float | None = None,
        order: int = 12,
        nsteps: int = 500,
        max_step: float = 0.0,
        min_step: float = 0.0,
        first_step: float = 0.0,
    ) -> None: ...

class zvode(vode[np.complex128]):
    active_global_handle: int
    zwork: onp.Array1D[np.complex128]
    call_args: list[float | onp.ArrayND[np.complex128] | onp.ArrayND[np.float64] | onp.ArrayND[np.int32]]  # type: ignore[assignment] # pyright: ignore[reportIncompatibleVariableOverride]
    initialized: bool

class dopri5(IntegratorBase[np.float64]):
    name: ClassVar[str] = "dopri5"
    messages: ClassVar[dict[int, str]] = ...

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
    solout: Callable[[float, onp.Array1D[np.inexact[Any]]], Literal[0, -1]] | None
    solout_cmplx: bool
    iout: int
    work: onp.Array1D[np.float64]
    iwork: onp.Array1D[np.int32]
    call_args: list[float | Callable[..., Literal[0, -1, 1]] | onp.ArrayND[np.float64] | onp.ArrayND[np.int32]]

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
        method: None = None,  # unused
        verbosity: int = -1,
    ) -> None: ...
    def set_solout(self, /, solout: _SolOutFunc | None, complex: bool = False) -> None: ...
    def _solout(
        self,
        /,
        nr: int,  # unused
        xold: object,  # unused
        x: float,
        y: onp.Array1D[np.floating[Any]],
        nd: int,  # unused
        icomp: int,  # unused
        con: object,  # unused
    ) -> Literal[0, -1, 1]: ...

class dop853(dopri5):
    name: ClassVar[str] = "dop853"
    def __init__(
        self,
        /,
        rtol: float = 1e-06,
        atol: float = 1e-12,
        nsteps: int = 500,
        max_step: float = 0.0,
        first_step: float = 0.0,
        safety: float = 0.9,
        ifactor: float = 6.0,
        dfactor: float = 0.3,
        beta: float = 0.0,
        method: None = None,  # ignored
        verbosity: int = -1,
    ) -> None: ...

class lsoda(IntegratorBase[np.float64]):
    active_global_handle: ClassVar[int] = 0
    messages: ClassVar[dict[int, str]] = ...

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
    rwork: onp.Array1D[np.float64]
    iwork: onp.Array1D[np.int32]
    call_args: list[float | onp.Array1D[np.float64] | onp.Array1D[np.int32]]
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
        method: None = None,  # ignored
    ) -> None: ...
