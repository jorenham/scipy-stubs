import abc
from collections.abc import Callable
from typing import Any, Final, Literal, Protocol, TypeAlias, TypedDict, type_check_only
from typing_extensions import Unpack, override

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from scipy.sparse import sparray, spmatrix
from scipy.sparse.linalg import LinearOperator

__all__ = [
    "BroydenFirst",
    "InverseJacobian",
    "KrylovJacobian",
    "NoConvergence",
    "anderson",
    "broyden1",
    "broyden2",
    "diagbroyden",
    "excitingmixing",
    "linearmixing",
    "newton_krylov",
]

_Float: TypeAlias = np.floating[Any]
_Complex: TypeAlias = np.complexfloating[Any, Any]
_Inexact: TypeAlias = _Float | _Complex

_FloatND: TypeAlias = onp.ArrayND[_Float]
_Inexact1D: TypeAlias = onp.Array1D[_Inexact]
_Inexact2D: TypeAlias = onp.Array2D[_Inexact]
_InexactND: TypeAlias = onp.ArrayND[_Inexact]

_SparseArray: TypeAlias = sparray | spmatrix

_JacobianMethod: TypeAlias = Literal[
    "krylov",
    "broyden1",
    "broyden2",
    "anderson",
    "diagbroyden",
    "linearmixing",
    "excitingmixing",
]
_KrylovMethod: TypeAlias = Literal["lgmres", "gmres", "bicgstab", "cgs", "minres", "tfqmr"]
_ReductionMethod: TypeAlias = Literal["restart", "simple", "svd"]
_LineSearch: TypeAlias = Literal["armijo", "wolfe"]

_Callback: TypeAlias = (
    Callable[[onp.ArrayND[np.float64], np.float64], None] | Callable[[onp.ArrayND[np.complex128], np.float64], None]
)
_ResidFunc: TypeAlias = Callable[[onp.ArrayND[np.float64]], onp.ToFloat] | Callable[[onp.ArrayND[np.complex128]], onp.ToFloat]

_JacobianLike: TypeAlias = (
    Jacobian
    | type[Jacobian]
    | onp.ArrayND
    | _SparseArray
    | _SupportsJacobian
    | Callable[[onp.ArrayND[np.float64]], _FloatND | _SparseArray]
    | Callable[[onp.ArrayND[np.complex128]], _InexactND | _SparseArray]
    | str
)

@type_check_only
class _SupportsJacobian(Protocol):
    @property
    def shape(self, /) -> tuple[int, ...]: ...
    @property
    def dtype(self, /) -> np.dtype[np.generic]: ...
    def solve(self, /, v: onp.ToComplexND, tol: onp.ToFloat = 0) -> _Inexact2D: ...

@type_check_only
class _JacobianKwargs(TypedDict, total=False):
    solve: Callable[[_InexactND], _Inexact2D] | Callable[[_InexactND, onp.ToFloat], _Inexact2D]
    rsolve: Callable[[_InexactND], _Inexact2D] | Callable[[_InexactND, onp.ToFloat], _Inexact2D]
    matvec: Callable[[_InexactND], _Inexact1D] | Callable[[_InexactND, onp.ToFloat], _Inexact1D]
    rmatvec: Callable[[_InexactND], _Inexact1D] | Callable[[_InexactND, onp.ToFloat], _Inexact1D]
    matmat: Callable[[_InexactND], _Inexact2D]
    update: Callable[[_InexactND, _InexactND], None]
    todense: Callable[[], _Inexact2D]
    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]

###

class NoConvergence(Exception): ...

class TerminationCondition:
    x_tol: Final[float]
    x_rtol: Final[float]
    f_tol: Final[float]
    f_rtol: Final[float]
    iter: Final[int | None]
    norm: Final[Callable[[_InexactND], float | np.float64]]

    f0_norm: float | None
    iteration: int

    def __init__(
        self,
        /,
        f_tol: float | None = None,
        f_rtol: float | None = None,
        x_tol: float | None = None,
        x_rtol: float | None = None,
        iter: int | None = None,
        norm: Callable[[_InexactND], float | np.float64] = ...,
    ) -> None: ...
    def check(self, /, f: _InexactND, x: _InexactND, dx: _InexactND) -> int: ...

# TODO(jorenham): Make generic on (inexact) dtype
class Jacobian:
    shape: Final[tuple[int, int]]
    dtype: Final[np.dtype[np.generic]]
    func: Final[_ResidFunc]

    def __init__(self, /, **kw: Unpack[_JacobianKwargs]) -> None: ...
    #
    @abc.abstractmethod
    def solve(self, /, v: _InexactND, tol: float = 0) -> _Inexact2D: ...
    # `x` and `F` are 1-d
    def setup(self, /, x: _InexactND, F: _InexactND, func: _ResidFunc) -> None: ...
    def update(self, /, x: _InexactND, F: _InexactND) -> None: ...  # does nothing
    def aspreconditioner(self, /) -> InverseJacobian: ...

# TODO(jorenham): Make generic on shape an dtype
class InverseJacobian:  # undocumented
    jacobian: Final[Jacobian]
    matvec: Final[Callable[[_InexactND], _Inexact1D] | Callable[[_InexactND, onp.ToFloat], _Inexact1D]]
    rmatvec: Final[Callable[[_InexactND], _Inexact1D] | Callable[[_InexactND, onp.ToFloat], _Inexact1D]]

    @property
    def shape(self, /) -> tuple[int, ...]: ...
    @property
    def dtype(self, /) -> np.dtype[np.generic]: ...
    #
    def __init__(self, /, jacobian: Jacobian) -> None: ...

def asjacobian(J: _JacobianLike) -> Jacobian: ...  # undocumented

class GenericBroyden(Jacobian, metaclass=abc.ABCMeta):
    alpha: Final[float | None]
    last_x: _Inexact1D
    last_f: float

    @override
    def setup(self, /, x0: _InexactND, f0: _InexactND, func: _ResidFunc) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def update(self, /, x0: _InexactND, f0: _InexactND) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

class LowRankMatrix:
    alpha: Final[float]
    n: Final[int]
    cs: Final[list[_InexactND]]
    ds: Final[list[_InexactND]]
    dtype: Final[np.dtype[np.inexact[Any]]]
    collapsed: _InexactND | None

    def __init__(self, /, alpha: float, n: int, dtype: np.dtype[np.inexact[Any]]) -> None: ...
    def __array__(self, /, dtype: npt.DTypeLike | None = None, copy: bool | None = None) -> _Inexact2D: ...
    def solve(self, /, v: _InexactND, tol: float = 0) -> _Inexact2D: ...
    def rsolve(self, /, v: _InexactND, tol: float = 0) -> _Inexact2D: ...
    def matvec(self, /, v: _InexactND) -> _Inexact1D: ...
    def rmatvec(self, /, v: _InexactND) -> _Inexact1D: ...
    def append(self, /, c: _InexactND, d: _InexactND) -> None: ...
    def collapse(self, /) -> None: ...
    def restart_reduce(self, /, rank: float) -> None: ...
    def simple_reduce(self, /, rank: float) -> None: ...
    def svd_reduce(self, /, max_rank: float, to_retain: int | None = None) -> None: ...

class BroydenFirst(GenericBroyden):
    max_rank: Final[float]
    Gm: LowRankMatrix | None

    def __init__(
        self,
        /,
        alpha: float | None = None,
        reduction_method: _ReductionMethod = "restart",
        max_rank: float | None = None,
    ) -> None: ...
    @override
    def solve(self, /, f: _InexactND, tol: float = 0) -> _Inexact2D: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def rsolve(self, /, f: _InexactND, tol: float = 0) -> _Inexact2D: ...
    def matvec(self, /, f: _InexactND) -> _Inexact1D: ...
    def rmatvec(self, /, f: _InexactND) -> _Inexact1D: ...
    def todense(self, /) -> _Inexact2D: ...

class BroydenSecond(BroydenFirst): ...

class Anderson(GenericBroyden):
    w0: Final[float]
    M: Final[float]
    dx: list[_InexactND]
    df: list[_InexactND]
    gamma: _InexactND | None

    def __init__(self, /, alpha: float | None = None, w0: float = 0.01, M: float = 5) -> None: ...
    @override
    def solve(self, /, f: _InexactND, tol: float = 0) -> _Inexact2D: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def matvec(self, /, f: _InexactND) -> _Inexact1D: ...

class DiagBroyden(GenericBroyden):
    d: _Inexact1D

    def __init__(self, /, alpha: float | None = None) -> None: ...
    @override
    def solve(self, /, f: _InexactND, tol: float = 0) -> _Inexact2D: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def rsolve(self, /, f: _InexactND, tol: float = 0) -> _Inexact2D: ...
    def matvec(self, /, f: _InexactND) -> _Inexact1D: ...
    def rmatvec(self, /, f: _InexactND) -> _Inexact1D: ...
    def todense(self, /) -> _Inexact2D: ...

class LinearMixing(GenericBroyden):
    def __init__(self, /, alpha: float | None = None) -> None: ...
    @override
    def solve(self, /, f: _InexactND, tol: float = 0) -> _Inexact2D: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def rsolve(self, /, f: _InexactND, tol: float = 0) -> _Inexact2D: ...
    def matvec(self, /, f: _InexactND) -> _Inexact1D: ...
    def rmatvec(self, /, f: _InexactND) -> _Inexact1D: ...
    def todense(self, /) -> _Inexact2D: ...

class ExcitingMixing(GenericBroyden):
    alphamax: Final[float]
    beta: _Inexact1D | None

    def __init__(self, /, alpha: float | None = None, alphamax: float = 1.0) -> None: ...
    @override
    def solve(self, /, f: _InexactND, tol: float = 0) -> _Inexact2D: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def rsolve(self, /, f: _InexactND, tol: float = 0) -> _Inexact2D: ...
    def matvec(self, /, f: _InexactND) -> _Inexact1D: ...
    def rmatvec(self, /, f: _InexactND) -> _Inexact1D: ...
    def todense(self, /) -> _Inexact2D: ...

class KrylovJacobian(Jacobian):
    rdiff: Final[float]
    method: Final[_KrylovMethod]
    method_kw: Final[dict[str, object]]
    preconditioner: LinearOperator | InverseJacobian | None
    x0: _InexactND
    f0: _InexactND
    op: LinearOperator

    def __init__(
        self,
        /,
        rdiff: float | None = None,
        method: _KrylovMethod = "lgmres",
        inner_maxiter: int = 20,
        inner_M: LinearOperator | InverseJacobian | None = None,
        outer_k: int = 10,
        **kw: object,
    ) -> None: ...
    def matvec(self, /, v: _InexactND) -> _Inexact2D: ...
    @override
    def solve(self, /, rhs: _InexactND, tol: float = 0) -> _Inexact2D: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def update(self, /, x: _InexactND, f: _InexactND) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def setup(self, /, x: _InexactND, f: _InexactND, func: _ResidFunc) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

def maxnorm(x: onp.ToComplexND) -> float | np.float64: ...  # undocumented

# TOOD: overload on full_output
def nonlin_solve(
    F: _ResidFunc,
    x0: onp.ToComplexND,
    jacobian: _JacobianMethod | _JacobianLike = "krylov",
    iter: onp.ToInt | None = None,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch = "armijo",
    callback: _Callback | None = None,
    full_output: op.CanBool = False,
    raise_exception: op.CanBool = True,
) -> _InexactND: ...

#
def broyden1(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    alpha: onp.ToFloat | None = None,
    reduction_method: _ReductionMethod = "restart",
    max_rank: onp.ToInt | None = None,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
def broyden2(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    alpha: onp.ToFloat | None = None,
    reduction_method: _ReductionMethod = "restart",
    max_rank: onp.ToInt | None = None,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
def anderson(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    alpha: onp.ToFloat | None = None,
    w0: onp.ToFloat = 0.01,
    M: onp.ToInt = 5,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
def linearmixing(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    alpha: onp.ToFloat | None = None,
    verbose: op.CanBool = False,
    maxiter: int | None = None,
    f_tol: onp.ToInt | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
def diagbroyden(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    alpha: onp.ToFloat | None = None,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
def excitingmixing(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    alpha: onp.ToFloat | None = None,
    alphamax: onp.ToFloat = 1.0,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
def newton_krylov(
    F: _ResidFunc,
    xin: onp.ToComplexND,
    iter: onp.ToInt | None = None,
    rdiff: onp.ToFloat | None = None,
    method: _KrylovMethod = "lgmres",
    inner_maxiter: onp.ToInt = 20,
    inner_M: LinearOperator | InverseJacobian | None = None,
    outer_k: onp.ToInt = 10,
    verbose: op.CanBool = False,
    maxiter: onp.ToInt | None = None,
    f_tol: onp.ToFloat | None = None,
    f_rtol: onp.ToFloat | None = None,
    x_tol: onp.ToFloat | None = None,
    x_rtol: onp.ToFloat | None = None,
    tol_norm: onp.ToFloat | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _InexactND: ...
