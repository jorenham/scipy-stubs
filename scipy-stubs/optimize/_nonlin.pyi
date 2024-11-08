import abc
from collections.abc import Callable
from typing import Any, Final, Literal, Protocol, TypeAlias, TypedDict, type_check_only
from typing_extensions import Unpack, override

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onpt
from numpy._typing import _ArrayLikeNumber_co
from scipy._typing import AnyInt, AnyReal
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

_Floating: TypeAlias = np.floating[Any]
_ComplexFloating: TypeAlias = np.complexfloating[Any, Any]
_Inexact: TypeAlias = _Floating | _ComplexFloating

_Array_f: TypeAlias = npt.NDArray[_Floating]
_Array_fc: TypeAlias = npt.NDArray[_Inexact]
_Array_fc_1d: TypeAlias = onpt.Array[tuple[int], _Inexact]
_Array_fc_2d: TypeAlias = onpt.Array[tuple[int, int], _Inexact]

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
    Callable[[npt.NDArray[np.float64], np.float64], None] | Callable[[npt.NDArray[np.complex128], np.float64], None]
)
_ResidFunc: TypeAlias = Callable[[npt.NDArray[np.float64]], AnyReal] | Callable[[npt.NDArray[np.complex128]], AnyReal]

_JacobianLike: TypeAlias = (
    Jacobian
    | type[Jacobian]
    | npt.NDArray[np.generic]
    | _SparseArray
    | _SupportsJacobian
    | Callable[[npt.NDArray[np.float64]], _Array_f | _SparseArray]
    | Callable[[npt.NDArray[np.complex128]], _Array_fc | _SparseArray]
    | str
)

@type_check_only
class _SupportsJacobian(Protocol):
    @property
    def shape(self, /) -> tuple[int, ...]: ...
    @property
    def dtype(self, /) -> np.dtype[np.generic]: ...
    def solve(self, /, v: _ArrayLikeNumber_co, tol: AnyReal = 0) -> _Array_fc_2d: ...

@type_check_only
class _JacobianKwargs(TypedDict, total=False):
    solve: Callable[[_Array_fc], _Array_fc_2d] | Callable[[_Array_fc, AnyReal], _Array_fc_2d]
    rsolve: Callable[[_Array_fc], _Array_fc_2d] | Callable[[_Array_fc, AnyReal], _Array_fc_2d]
    matvec: Callable[[_Array_fc], _Array_fc_1d] | Callable[[_Array_fc, AnyReal], _Array_fc_1d]
    rmatvec: Callable[[_Array_fc], _Array_fc_1d] | Callable[[_Array_fc, AnyReal], _Array_fc_1d]
    matmat: Callable[[_Array_fc], _Array_fc_2d]
    update: Callable[[_Array_fc, _Array_fc], None]
    todense: Callable[[], _Array_fc_2d]
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
    norm: Final[Callable[[_Array_fc], float | np.float64]]

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
        norm: Callable[[_Array_fc], float | np.float64] = ...,
    ) -> None: ...
    def check(self, /, f: _Array_fc, x: _Array_fc, dx: _Array_fc) -> int: ...

# TODO(jorenham): Make generic on (inexact) dtype
class Jacobian:
    shape: Final[tuple[int, int]]
    dtype: Final[np.dtype[np.generic]]
    func: Final[_ResidFunc]

    def __init__(self, /, **kw: Unpack[_JacobianKwargs]) -> None: ...
    #
    @abc.abstractmethod
    def solve(self, /, v: _Array_fc, tol: float = 0) -> _Array_fc_2d: ...
    # `x` and `F` are 1-d
    def setup(self, /, x: _Array_fc, F: _Array_fc, func: _ResidFunc) -> None: ...
    def update(self, /, x: _Array_fc, F: _Array_fc) -> None: ...  # does nothing
    def aspreconditioner(self, /) -> InverseJacobian: ...

# TODO(jorenham): Make generic on shape an dtype
class InverseJacobian:  # undocumented
    jacobian: Final[Jacobian]
    matvec: Final[Callable[[_Array_fc], _Array_fc_1d] | Callable[[_Array_fc, AnyReal], _Array_fc_1d]]
    rmatvec: Final[Callable[[_Array_fc], _Array_fc_1d] | Callable[[_Array_fc, AnyReal], _Array_fc_1d]]

    @property
    def shape(self, /) -> tuple[int, ...]: ...
    @property
    def dtype(self, /) -> np.dtype[np.generic]: ...
    #
    def __init__(self, /, jacobian: Jacobian) -> None: ...

def asjacobian(J: _JacobianLike) -> Jacobian: ...  # undocumented

class GenericBroyden(Jacobian, metaclass=abc.ABCMeta):
    alpha: Final[float | None]
    last_x: _Array_fc_1d
    last_f: float

    @override
    def setup(self, /, x0: _Array_fc, f0: _Array_fc, func: _ResidFunc) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def update(self, /, x0: _Array_fc, f0: _Array_fc) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

class LowRankMatrix:
    alpha: Final[float]
    n: Final[int]
    cs: Final[list[_Array_fc]]
    ds: Final[list[_Array_fc]]
    dtype: Final[np.dtype[np.inexact[Any]]]
    collapsed: _Array_fc | None

    def __init__(self, /, alpha: float, n: int, dtype: np.dtype[np.inexact[Any]]) -> None: ...
    def __array__(self, /, dtype: npt.DTypeLike | None = None, copy: bool | None = None) -> _Array_fc_2d: ...
    def solve(self, /, v: _Array_fc, tol: float = 0) -> _Array_fc_2d: ...
    def rsolve(self, /, v: _Array_fc, tol: float = 0) -> _Array_fc_2d: ...
    def matvec(self, /, v: _Array_fc) -> _Array_fc_1d: ...
    def rmatvec(self, /, v: _Array_fc) -> _Array_fc_1d: ...
    def append(self, /, c: _Array_fc, d: _Array_fc) -> None: ...
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
    def solve(self, /, f: _Array_fc, tol: float = 0) -> _Array_fc_2d: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def rsolve(self, /, f: _Array_fc, tol: float = 0) -> _Array_fc_2d: ...
    def matvec(self, /, f: _Array_fc) -> _Array_fc_1d: ...
    def rmatvec(self, /, f: _Array_fc) -> _Array_fc_1d: ...
    def todense(self, /) -> _Array_fc_2d: ...

class BroydenSecond(BroydenFirst): ...

class Anderson(GenericBroyden):
    w0: Final[float]
    M: Final[float]
    dx: list[_Array_fc]
    df: list[_Array_fc]
    gamma: _Array_fc | None

    def __init__(self, /, alpha: float | None = None, w0: float = 0.01, M: float = 5) -> None: ...
    @override
    def solve(self, /, f: _Array_fc, tol: float = 0) -> _Array_fc_2d: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def matvec(self, /, f: _Array_fc) -> _Array_fc_1d: ...

class DiagBroyden(GenericBroyden):
    d: _Array_fc_1d

    def __init__(self, /, alpha: float | None = None) -> None: ...
    @override
    def solve(self, /, f: _Array_fc, tol: float = 0) -> _Array_fc_2d: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def rsolve(self, /, f: _Array_fc, tol: float = 0) -> _Array_fc_2d: ...
    def matvec(self, /, f: _Array_fc) -> _Array_fc_1d: ...
    def rmatvec(self, /, f: _Array_fc) -> _Array_fc_1d: ...
    def todense(self, /) -> _Array_fc_2d: ...

class LinearMixing(GenericBroyden):
    def __init__(self, /, alpha: float | None = None) -> None: ...
    @override
    def solve(self, /, f: _Array_fc, tol: float = 0) -> _Array_fc_2d: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def rsolve(self, /, f: _Array_fc, tol: float = 0) -> _Array_fc_2d: ...
    def matvec(self, /, f: _Array_fc) -> _Array_fc_1d: ...
    def rmatvec(self, /, f: _Array_fc) -> _Array_fc_1d: ...
    def todense(self, /) -> _Array_fc_2d: ...

class ExcitingMixing(GenericBroyden):
    alphamax: Final[float]
    beta: _Array_fc_1d | None

    def __init__(self, /, alpha: float | None = None, alphamax: float = 1.0) -> None: ...
    @override
    def solve(self, /, f: _Array_fc, tol: float = 0) -> _Array_fc_2d: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def rsolve(self, /, f: _Array_fc, tol: float = 0) -> _Array_fc_2d: ...
    def matvec(self, /, f: _Array_fc) -> _Array_fc_1d: ...
    def rmatvec(self, /, f: _Array_fc) -> _Array_fc_1d: ...
    def todense(self, /) -> _Array_fc_2d: ...

class KrylovJacobian(Jacobian):
    rdiff: Final[float]
    method: Final[_KrylovMethod]
    method_kw: Final[dict[str, object]]
    preconditioner: LinearOperator | InverseJacobian | None
    x0: _Array_fc
    f0: _Array_fc
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
    def matvec(self, /, v: _Array_fc) -> _Array_fc_2d: ...
    @override
    def solve(self, /, rhs: _Array_fc, tol: float = 0) -> _Array_fc_2d: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def update(self, /, x: _Array_fc, f: _Array_fc) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def setup(self, /, x: _Array_fc, f: _Array_fc, func: _ResidFunc) -> None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

def maxnorm(x: _ArrayLikeNumber_co) -> float | np.float64: ...  # undocumented

# TOOD: overload on full_output
def nonlin_solve(
    F: _ResidFunc,
    x0: _ArrayLikeNumber_co,
    jacobian: _JacobianMethod | _JacobianLike = "krylov",
    iter: AnyInt | None = None,
    verbose: op.CanBool = False,
    maxiter: AnyInt | None = None,
    f_tol: AnyReal | None = None,
    f_rtol: AnyReal | None = None,
    x_tol: AnyReal | None = None,
    x_rtol: AnyReal | None = None,
    tol_norm: AnyReal | None = None,
    line_search: _LineSearch = "armijo",
    callback: _Callback | None = None,
    full_output: op.CanBool = False,
    raise_exception: op.CanBool = True,
) -> _Array_fc: ...

#
def broyden1(
    F: _ResidFunc,
    xin: _ArrayLikeNumber_co,
    iter: AnyInt | None = None,
    alpha: AnyReal | None = None,
    reduction_method: _ReductionMethod = "restart",
    max_rank: AnyInt | None = None,
    verbose: op.CanBool = False,
    maxiter: AnyInt | None = None,
    f_tol: AnyReal | None = None,
    f_rtol: AnyReal | None = None,
    x_tol: AnyReal | None = None,
    x_rtol: AnyReal | None = None,
    tol_norm: AnyReal | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _Array_fc: ...
def broyden2(
    F: _ResidFunc,
    xin: _ArrayLikeNumber_co,
    iter: AnyInt | None = None,
    alpha: AnyReal | None = None,
    reduction_method: _ReductionMethod = "restart",
    max_rank: AnyInt | None = None,
    verbose: op.CanBool = False,
    maxiter: AnyInt | None = None,
    f_tol: AnyReal | None = None,
    f_rtol: AnyReal | None = None,
    x_tol: AnyReal | None = None,
    x_rtol: AnyReal | None = None,
    tol_norm: AnyReal | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _Array_fc: ...
def anderson(
    F: _ResidFunc,
    xin: _ArrayLikeNumber_co,
    iter: AnyInt | None = None,
    alpha: AnyReal | None = None,
    w0: AnyReal = 0.01,
    M: AnyInt = 5,
    verbose: op.CanBool = False,
    maxiter: AnyInt | None = None,
    f_tol: AnyReal | None = None,
    f_rtol: AnyReal | None = None,
    x_tol: AnyReal | None = None,
    x_rtol: AnyReal | None = None,
    tol_norm: AnyReal | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _Array_fc: ...
def linearmixing(
    F: _ResidFunc,
    xin: _ArrayLikeNumber_co,
    iter: AnyInt | None = None,
    alpha: AnyReal | None = None,
    verbose: op.CanBool = False,
    maxiter: int | None = None,
    f_tol: AnyInt | None = None,
    f_rtol: AnyReal | None = None,
    x_tol: AnyReal | None = None,
    x_rtol: AnyReal | None = None,
    tol_norm: AnyReal | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _Array_fc: ...
def diagbroyden(
    F: _ResidFunc,
    xin: _ArrayLikeNumber_co,
    iter: AnyInt | None = None,
    alpha: AnyReal | None = None,
    verbose: op.CanBool = False,
    maxiter: AnyInt | None = None,
    f_tol: AnyReal | None = None,
    f_rtol: AnyReal | None = None,
    x_tol: AnyReal | None = None,
    x_rtol: AnyReal | None = None,
    tol_norm: AnyReal | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _Array_fc: ...
def excitingmixing(
    F: _ResidFunc,
    xin: _ArrayLikeNumber_co,
    iter: AnyInt | None = None,
    alpha: AnyReal | None = None,
    alphamax: AnyReal = 1.0,
    verbose: op.CanBool = False,
    maxiter: AnyInt | None = None,
    f_tol: AnyReal | None = None,
    f_rtol: AnyReal | None = None,
    x_tol: AnyReal | None = None,
    x_rtol: AnyReal | None = None,
    tol_norm: AnyReal | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _Array_fc: ...
def newton_krylov(
    F: _ResidFunc,
    xin: _ArrayLikeNumber_co,
    iter: AnyInt | None = None,
    rdiff: AnyReal | None = None,
    method: _KrylovMethod = "lgmres",
    inner_maxiter: AnyInt = 20,
    inner_M: LinearOperator | InverseJacobian | None = None,
    outer_k: AnyInt = 10,
    verbose: op.CanBool = False,
    maxiter: AnyInt | None = None,
    f_tol: AnyReal | None = None,
    f_rtol: AnyReal | None = None,
    x_tol: AnyReal | None = None,
    x_rtol: AnyReal | None = None,
    tol_norm: AnyReal | None = None,
    line_search: _LineSearch | None = "armijo",
    callback: _Callback | None = None,
) -> _Array_fc: ...
