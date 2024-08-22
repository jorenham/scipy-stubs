from ._hessian_update_strategy import HessianUpdateStrategy as HessianUpdateStrategy
from ._numdiff import approx_derivative as approx_derivative, group_columns as group_columns
from scipy._lib._array_api import array_namespace as array_namespace, xp_atleast_nd as xp_atleast_nd
from scipy._typing import Untyped
from scipy.sparse.linalg import LinearOperator as LinearOperator

FD_METHODS: Untyped

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
    def __init__(self, fun, x0, args, grad, hess, finite_diff_rel_step, finite_diff_bounds, epsilon: Untyped | None = None): ...
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
        self, fun, x0, jac, hess, finite_diff_rel_step, finite_diff_jac_sparsity, finite_diff_bounds, sparse_jacobian
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
