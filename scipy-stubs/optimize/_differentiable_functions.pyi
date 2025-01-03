from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, Concatenate, Final, Generic, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import optype.numpy as onp
from scipy.sparse import csr_matrix, sparray, spmatrix
from scipy.sparse.linalg import LinearOperator
from ._hessian_update_strategy import HessianUpdateStrategy

_XT = TypeVar("_XT", bound=np.floating[Any], default=np.floating[Any])
_XT_contra = TypeVar("_XT_contra", bound=np.floating[Any], default=np.floating[Any], contravariant=True)

_ToFloat64Vec: TypeAlias = Sequence[float | np.float64 | np.integer[Any] | np.bool_] | onp.CanArrayND[np.float64]
_ToJac: TypeAlias = onp.ToFloat2D | spmatrix | sparray
_ToHess: TypeAlias = _ToJac | LinearOperator

_Vec: TypeAlias = onp.Array1D[_XT]
_Jac: TypeAlias = onp.Array2D[_XT] | csr_matrix
_Hess: TypeAlias = _Jac[_XT] | LinearOperator

_ScalarFun: TypeAlias = Callable[Concatenate[onp.Array1D[_XT], ...], onp.ToFloat]
_VectorFun: TypeAlias = Callable[Concatenate[onp.Array1D[_XT], ...], onp.ToFloat1D]
_JacFun: TypeAlias = Callable[Concatenate[onp.Array1D[_XT], ...], _ToJac]
_HessFun: TypeAlias = Callable[Concatenate[onp.Array1D[_XT], ...], _ToHess]

_FDMethod: TypeAlias = Literal["2-point", "3-point", "cs"]
_FDBounds: TypeAlias = onp.ToFloat1D | onp.ToFloat2D  # len-2 array-like of scalar- or vector-likes

_ToGradFun: TypeAlias = _VectorFun[_XT_contra] | _FDMethod
_ToJacFun: TypeAlias = _JacFun[_XT_contra] | _FDMethod
_ToHessFun: TypeAlias = _HessFun[_XT_contra] | _FDMethod | HessianUpdateStrategy

###

FD_METHODS: Final = "2-point", "3-point", "cs"

class ScalarFunction(Generic[_XT_contra]):
    xp: Final[ModuleType]

    _args: Final[tuple[object, ...]]

    n: Final[int]
    x_dtype: _XT_contra  # readonly
    x: _Vec[_XT_contra]
    x_prev: _Vec[_XT_contra] | None
    _lowest_x: _Vec[_XT_contra] | None

    _orig_fun: _ScalarFun[_XT_contra]  # readonly
    _wrapped_fun: Callable[[onp.Array1D[_XT_contra]], onp.ToFloat]  # readonly
    _nfev: Final[list[int]]  # size 1
    _lowest_f: onp.ToFloat
    f_updated: bool

    _orig_grad: _ToGradFun[_XT_contra]  # readonly
    _wrapped_grad: Callable[[onp.Array1D[_XT_contra]], _Vec]  # readonly
    _ngev: Final[list[int]]  # size 1
    g_prev: _Vec | None
    g_updated: bool

    _orig_hess: _ToHessFun[_XT_contra]  # readonly
    _wrapped_hess: Callable[[onp.Array1D[_XT_contra]], _Hess]  # readonly
    _nhev: Final[list[int]]  # size 1
    H: _Hess
    H_updated: bool

    #
    @property
    def nfev(self, /) -> int: ...
    @property
    def ngev(self, /) -> int: ...
    @property
    def nhev(self, /) -> int: ...

    #
    @overload
    def __init__(
        self: ScalarFunction[np.float64],
        /,
        fun: _ScalarFun[np.float64],
        x0: _ToFloat64Vec,
        args: tuple[object, ...],
        grad: _ToGradFun[np.float64],
        hess: _ToHessFun[np.float64],
        finite_diff_rel_step: onp.ToFloat | onp.ToFloat1D | None,
        finite_diff_bounds: _FDBounds,
        epsilon: onp.ToFloat | onp.ToFloat1D | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        fun: _ScalarFun[_XT_contra],
        x0: _Vec[_XT_contra] | onp.ToFloat1D,
        args: tuple[object, ...],
        grad: _ToGradFun[_XT_contra],
        hess: _ToHessFun[_XT_contra],
        finite_diff_rel_step: onp.ToFloat | onp.ToFloat1D | None,
        finite_diff_bounds: _FDBounds,
        epsilon: onp.ToFloat | onp.ToFloat1D | None = None,
    ) -> None: ...

    #
    def _update_x(self, /, x: onp.ToFloat1D) -> None: ...
    def _update_fun(self, /) -> None: ...
    def _update_grad(self, /) -> None: ...
    def _update_hess(self, /) -> None: ...

    #
    def fun(self, /, x: onp.ToFloat1D) -> float | np.floating[Any]: ...
    def grad(self, /, x: onp.ToFloat1D) -> _Vec: ...
    def hess(self, /, x: onp.ToFloat1D) -> _Hess: ...
    def fun_and_grad(self, /, x: onp.ToFloat1D) -> tuple[float | np.floating[Any], _Vec]: ...

class VectorFunction(Generic[_XT_contra]):
    xp: Final[ModuleType]

    n: Final[int]
    x_dtype: _XT_contra  # readonly
    x: _Vec[_XT_contra]
    x_diff: _Vec[_XT_contra]
    x_prev: _Vec[_XT_contra] | None

    m: Final[int]
    f: _Vec
    v: _Vec
    f_updated: bool
    nfev: int

    sparse_jacobian: Final[bool]
    J: _Jac
    J_prev: _Jac | None
    J_updated: bool
    njev: int

    H: _Hess
    H_updated: bool
    nhev: int

    @overload
    def __init__(
        self: VectorFunction[np.float64],
        /,
        fun: _VectorFun[np.float64],
        x0: _ToFloat64Vec,
        jac: _ToJacFun[np.float64],
        hess: _ToHessFun[np.float64],
        finite_diff_rel_step: onp.ToFloat | onp.ToFloat1D | None,
        finite_diff_jac_sparsity: _ToJac | None,
        finite_diff_bounds: _FDBounds,
        sparse_jacobian: onp.ToBool | None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        fun: _VectorFun[_XT_contra],
        x0: _Vec[_XT_contra] | onp.ToFloat1D,
        jac: _ToJacFun[_XT_contra],
        hess: _ToHessFun[_XT_contra],
        finite_diff_rel_step: onp.ToFloat | onp.ToFloat1D | None,
        finite_diff_jac_sparsity: _ToJac | None,
        finite_diff_bounds: _FDBounds,
        sparse_jacobian: onp.ToBool | None,
    ) -> None: ...

    #
    def _update_v(self, /, v: onp.ToFloat1D) -> None: ...
    def _update_x(self, /, x: onp.ToFloat1D) -> None: ...
    def _update_fun(self, /) -> None: ...
    def _update_jac(self, /) -> None: ...
    def _update_hess(self, /) -> None: ...

    #
    def fun(self, /, x: onp.ToFloat1D) -> _Vec: ...
    def jac(self, /, x: onp.ToFloat1D) -> _Jac: ...
    def hess(self, /, x: onp.ToFloat1D, v: onp.ToFloat1D) -> _Hess: ...

class LinearVectorFunction(Generic[_XT_contra]):
    xp: Final[ModuleType]

    n: Final[int]
    x_dtype: _XT_contra  # readonly
    x: _Vec[_XT_contra]

    m: Final[int]
    f: _Vec
    f_updated: bool

    sparse_jacobian: Final[bool]
    J: Final[_Jac]

    H: Final[csr_matrix]
    v: _Vec[np.float64]

    @overload
    def __init__(
        self: LinearVectorFunction[np.float64],
        /,
        A: onp.ToFloat2D | spmatrix | sparray,
        x0: _ToFloat64Vec,
        sparse_jacobian: onp.ToBool | None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        /,
        A: onp.ToFloat2D | spmatrix | sparray,
        x0: _Vec[_XT_contra] | onp.ToFloat1D,
        sparse_jacobian: onp.ToBool | None,
    ) -> None: ...

    #
    def _update_x(self, /, x: onp.ToFloat1D) -> None: ...

    #
    def fun(self, /, x: onp.ToFloat1D) -> _Vec: ...
    def jac(self, /, x: onp.ToFloat1D) -> _Jac: ...
    def hess(self, /, x: onp.ToFloat1D, v: _Vec[np.float64]) -> csr_matrix: ...

class IdentityVectorFunction(LinearVectorFunction[_XT_contra]):
    @overload
    def __init__(self: IdentityVectorFunction[np.float64], /, x0: _ToFloat64Vec, sparse_jacobian: onp.ToBool | None) -> None: ...
    @overload
    def __init__(self, /, x0: onp.CanArrayND[_XT_contra] | onp.ToFloat1D, sparse_jacobian: onp.ToBool | None) -> None: ...
