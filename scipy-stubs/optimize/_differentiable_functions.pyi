from typing import Final

from scipy._typing import Untyped

FD_METHODS: Final = ("2-point", "3-point", "cs")

class ScalarFunction:
    xp: Untyped
    x: Untyped
    x_dtype: Untyped
    n: Untyped
    f_updated: bool
    g_updated: bool
    H_updated: bool
    H: Untyped
    x_prev: Untyped
    g_prev: Untyped
    def __init__(
        self,
        fun: Untyped,
        x0: Untyped,
        args: Untyped,
        grad: Untyped,
        hess: Untyped,
        finite_diff_rel_step: Untyped,
        finite_diff_bounds: Untyped,
        epsilon: Untyped | None = None,
    ): ...
    @property
    def nfev(self) -> Untyped: ...
    @property
    def ngev(self) -> Untyped: ...
    @property
    def nhev(self) -> Untyped: ...
    def fun(self, x) -> Untyped: ...
    def grad(self, x) -> Untyped: ...
    def hess(self, x) -> Untyped: ...
    def fun_and_grad(self, x) -> Untyped: ...

class VectorFunction:
    xp: Untyped
    x: Untyped
    x_dtype: Untyped
    n: Untyped
    nfev: int
    njev: int
    nhev: int
    f_updated: bool
    J_updated: bool
    H_updated: bool
    x_diff: Untyped
    f: Untyped
    v: Untyped
    m: Untyped
    J: Untyped
    sparse_jacobian: bool
    H: Untyped
    x_prev: Untyped
    J_prev: Untyped
    def __init__(
        self,
        fun: Untyped,
        x0: Untyped,
        jac: Untyped,
        hess: Untyped,
        finite_diff_rel_step: Untyped,
        finite_diff_jac_sparsity: Untyped,
        finite_diff_bounds: Untyped,
        sparse_jacobian: Untyped,
    ) -> None: ...
    def fun(self, x) -> Untyped: ...
    def jac(self, x) -> Untyped: ...
    def hess(self, x, v) -> Untyped: ...

class LinearVectorFunction:
    J: Untyped
    sparse_jacobian: bool
    xp: Untyped
    x: Untyped
    x_dtype: Untyped
    f: Untyped
    f_updated: bool
    v: Untyped
    H: Untyped
    def __init__(self, A, x0, sparse_jacobian) -> None: ...
    def fun(self, x) -> Untyped: ...
    def jac(self, x) -> Untyped: ...
    def hess(self, x, v) -> Untyped: ...

class IdentityVectorFunction(LinearVectorFunction):
    def __init__(self, x0, sparse_jacobian) -> None: ...
