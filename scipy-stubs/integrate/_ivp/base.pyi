from scipy._typing import Untyped

def check_arguments(fun, y0, support_complex) -> Untyped: ...

class OdeSolver:
    TOO_SMALL_STEP: str
    t_old: Untyped
    t: Untyped
    t_bound: Untyped
    vectorized: Untyped
    fun: Untyped
    fun_single: Untyped
    fun_vectorized: Untyped
    direction: Untyped
    n: Untyped
    status: str
    nfev: int
    njev: int
    nlu: int
    def __init__(self, fun, t0, y0, t_bound, vectorized, support_complex: bool = False): ...
    @property
    def step_size(self) -> Untyped: ...
    def step(self) -> Untyped: ...
    def dense_output(self) -> Untyped: ...

class DenseOutput:
    t_old: Untyped
    t: Untyped
    t_min: Untyped
    t_max: Untyped
    def __init__(self, t_old, t) -> None: ...
    def __call__(self, t) -> Untyped: ...

class ConstantDenseOutput(DenseOutput):
    value: Untyped
    def __init__(self, t_old, t, value) -> None: ...
