from scipy._typing import Untyped

class ode:
    stiff: int
    f: Untyped
    jac: Untyped
    f_params: Untyped
    jac_params: Untyped
    def __init__(self, f, jac: Untyped | None = None): ...
    @property
    def y(self) -> Untyped: ...
    t: Untyped
    def set_initial_value(self, y, t: float = 0.0) -> Untyped: ...
    def set_integrator(self, name, **integrator_params) -> Untyped: ...
    def integrate(self, t, step: bool = False, relax: bool = False) -> Untyped: ...
    def successful(self) -> Untyped: ...
    def get_return_code(self) -> Untyped: ...
    def set_f_params(self, *args) -> Untyped: ...
    def set_jac_params(self, *args) -> Untyped: ...
    def set_solout(self, solout): ...

class complex_ode(ode):
    cf: Untyped
    cjac: Untyped
    def __init__(self, f, jac: Untyped | None = None): ...
    @property
    def y(self) -> Untyped: ...
    def set_integrator(self, name, **integrator_params) -> Untyped: ...
    tmp: Untyped
    def set_initial_value(self, y, t: float = 0.0) -> Untyped: ...
    def integrate(self, t, step: bool = False, relax: bool = False) -> Untyped: ...
    def set_solout(self, solout): ...

def find_integrator(name) -> Untyped: ...

class IntegratorConcurrencyError(RuntimeError):
    def __init__(self, name) -> None: ...

class IntegratorBase:
    runner: Untyped
    success: Untyped
    istate: Untyped
    supports_run_relax: Untyped
    supports_step: Untyped
    supports_solout: bool
    integrator_classes: Untyped
    scalar = float
    handle: Untyped
    def acquire_new_handle(self): ...
    def check_handle(self): ...
    def reset(self, n, has_jac): ...
    def run(self, f, jac, y0, t0, t1, f_params, jac_params): ...
    def step(self, f, jac, y0, t0, t1, f_params, jac_params): ...
    def run_relax(self, f, jac, y0, t0, t1, f_params, jac_params): ...

class vode(IntegratorBase):
    runner: Untyped
    messages: Untyped
    supports_run_relax: int
    supports_step: int
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
    success: int
    initialized: bool
    def __init__(
        self,
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
    ): ...
    rwork: Untyped
    iwork: Untyped
    call_args: Untyped
    def reset(self, n, has_jac): ...
    istate: Untyped
    def run(self, f, jac, y0, t0, t1, f_params, jac_params) -> Untyped: ...
    def step(self, *args) -> Untyped: ...
    def run_relax(self, *args) -> Untyped: ...

class zvode(vode):
    runner: Untyped
    supports_run_relax: int
    supports_step: int
    scalar = complex
    active_global_handle: int
    zwork: Untyped
    rwork: Untyped
    iwork: Untyped
    call_args: Untyped
    success: int
    initialized: bool
    def reset(self, n, has_jac): ...

class dopri5(IntegratorBase):
    runner: Untyped
    name: str
    supports_solout: bool
    messages: Untyped
    rtol: Untyped
    atol: Untyped
    nsteps: Untyped
    max_step: Untyped
    first_step: Untyped
    safety: Untyped
    ifactor: Untyped
    dfactor: Untyped
    beta: Untyped
    verbosity: Untyped
    success: int
    def __init__(
        self,
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
    ): ...
    solout: Untyped
    solout_cmplx: Untyped
    iout: int
    def set_solout(self, solout, complex: bool = False): ...
    work: Untyped
    iwork: Untyped
    call_args: Untyped
    def reset(self, n, has_jac): ...
    istate: Untyped
    def run(self, f, jac, y0, t0, t1, f_params, jac_params) -> Untyped: ...

class dop853(dopri5):
    runner: Untyped
    name: str
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
    ): ...
    work: Untyped
    iwork: Untyped
    call_args: Untyped
    success: int
    def reset(self, n, has_jac): ...

class lsoda(IntegratorBase):
    runner: Untyped
    active_global_handle: int
    messages: Untyped
    with_jacobian: Untyped
    rtol: Untyped
    atol: Untyped
    mu: Untyped
    ml: Untyped
    max_order_ns: Untyped
    max_order_s: Untyped
    nsteps: Untyped
    max_step: Untyped
    min_step: Untyped
    first_step: Untyped
    ixpr: Untyped
    max_hnil: Untyped
    success: int
    initialized: bool
    def __init__(
        self,
        with_jacobian: bool = False,
        rtol: float = 1e-06,
        atol: float = 1e-12,
        lband: Untyped | None = None,
        uband: Untyped | None = None,
        nsteps: int = 500,
        max_step: float = 0.0,
        min_step: float = 0.0,
        first_step: float = 0.0,
        ixpr: int = 0,
        max_hnil: int = 0,
        max_order_ns: int = 12,
        max_order_s: int = 5,
        method: Untyped | None = None,
    ): ...
    rwork: Untyped
    iwork: Untyped
    call_args: Untyped
    def reset(self, n, has_jac): ...
    istate: Untyped
    def run(self, f, jac, y0, t0, t1, f_params, jac_params) -> Untyped: ...
    def step(self, *args) -> Untyped: ...
    def run_relax(self, *args) -> Untyped: ...
