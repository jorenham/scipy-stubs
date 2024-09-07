from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
import scipy._typing as spt

_PointsWeights: TypeAlias = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
_PointsWeightsMu: TypeAlias = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], np.float64]

__all__ = [
    "c_roots",
    "cg_roots",
    "chebyc",
    "chebys",
    "chebyt",
    "chebyu",
    "gegenbauer",
    "genlaguerre",
    "h_roots",
    "he_roots",
    "hermite",
    "hermitenorm",
    "j_roots",
    "jacobi",
    "js_roots",
    "l_roots",
    "la_roots",
    "laguerre",
    "legendre",
    "p_roots",
    "ps_roots",
    "roots_chebyc",
    "roots_chebys",
    "roots_chebyt",
    "roots_chebyu",
    "roots_gegenbauer",
    "roots_genlaguerre",
    "roots_hermite",
    "roots_hermitenorm",
    "roots_jacobi",
    "roots_laguerre",
    "roots_legendre",
    "roots_sh_chebyt",
    "roots_sh_chebyu",
    "roots_sh_jacobi",
    "roots_sh_legendre",
    "s_roots",
    "sh_chebyt",
    "sh_chebyu",
    "sh_jacobi",
    "sh_legendre",
    "t_roots",
    "ts_roots",
    "u_roots",
    "us_roots",
]

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

# mypy: disable-error-code="explicit-override"
class orthopoly1d(np.poly1d):
    limits: tuple[float, float]
    weights: npt.NDArray[np.float64]
    weight_func: Callable[[float], float]
    normcoef: float
    def __init__(
        self,
        /,
        roots: npt.ArrayLike,
        weights: npt.ArrayLike | None = None,
        hn: float = 1.0,
        kn: float = 1.0,
        wfunc: Callable[[float], float] | None = None,
        limits: tuple[float, float] | None = None,
        monic: bool = False,
        eval_func: np.ufunc | None = None,
    ) -> None: ...
    @overload  # type: ignore[override]
    def __call__(self, v: np.poly1d) -> np.poly1d: ...
    @overload
    def __call__(self, v: spt.AnyReal) -> np.floating[Any]: ...
    @overload
    def __call__(self, v: spt.AnyComplex) -> np.inexact[Any]: ...
    @overload
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        v: onpt.CanArray[_ShapeT, np.dtype[np.floating[Any] | np.integer[Any] | np.bool_]] | Sequence[npt.ArrayLike],
    ) -> onpt.Array[_ShapeT, np.floating[Any]]: ...

@overload
def roots_jacobi(n: spt.AnyInt, alpha: spt.AnyReal, beta: spt.AnyReal, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_jacobi(n: spt.AnyInt, alpha: spt.AnyReal, beta: spt.AnyReal, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_sh_jacobi(n: spt.AnyInt, p1: spt.AnyReal, q1: spt.AnyReal, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_sh_jacobi(n: spt.AnyInt, p1: spt.AnyReal, q1: spt.AnyReal, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_genlaguerre(n: spt.AnyInt, alpha: spt.AnyReal, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_genlaguerre(n: spt.AnyInt, alpha: spt.AnyReal, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_laguerre(n: spt.AnyInt, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_laguerre(n: spt.AnyInt, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_hermite(n: spt.AnyInt, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_hermite(n: spt.AnyInt, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_hermitenorm(n: spt.AnyInt, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_hermitenorm(n: spt.AnyInt, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_gegenbauer(n: spt.AnyInt, alpha: spt.AnyReal, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_gegenbauer(n: spt.AnyInt, alpha: spt.AnyReal, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_chebyt(n: spt.AnyInt, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_chebyt(n: spt.AnyInt, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_chebyu(n: spt.AnyInt, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_chebyu(n: spt.AnyInt, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_chebyc(n: spt.AnyInt, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_chebyc(n: spt.AnyInt, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_chebys(n: spt.AnyInt, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_chebys(n: spt.AnyInt, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_sh_chebyt(n: spt.AnyInt, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_sh_chebyt(n: spt.AnyInt, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_sh_chebyu(n: spt.AnyInt, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_sh_chebyu(n: spt.AnyInt, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_legendre(n: spt.AnyInt, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_legendre(n: spt.AnyInt, mu: Literal[True]) -> _PointsWeightsMu: ...
@overload
def roots_sh_legendre(n: spt.AnyInt, mu: Literal[False] = ...) -> _PointsWeights: ...
@overload
def roots_sh_legendre(n: spt.AnyInt, mu: Literal[True]) -> _PointsWeightsMu: ...
def legendre(n: spt.AnyInt, monic: bool = ...) -> orthopoly1d: ...
def chebyt(n: spt.AnyInt, monic: bool = ...) -> orthopoly1d: ...
def chebyu(n: spt.AnyInt, monic: bool = ...) -> orthopoly1d: ...
def chebyc(n: spt.AnyInt, monic: bool = ...) -> orthopoly1d: ...
def chebys(n: spt.AnyInt, monic: bool = ...) -> orthopoly1d: ...
def jacobi(n: spt.AnyInt, alpha: spt.AnyReal, beta: spt.AnyReal, monic: bool = ...) -> orthopoly1d: ...
def laguerre(n: spt.AnyInt, monic: bool = ...) -> orthopoly1d: ...
def genlaguerre(n: spt.AnyInt, alpha: spt.AnyReal, monic: bool = ...) -> orthopoly1d: ...
def hermite(n: spt.AnyInt, monic: bool = ...) -> orthopoly1d: ...
def hermitenorm(n: spt.AnyInt, monic: bool = ...) -> orthopoly1d: ...
def gegenbauer(n: spt.AnyInt, alpha: spt.AnyReal, monic: bool = ...) -> orthopoly1d: ...
def sh_legendre(n: spt.AnyInt, monic: bool = ...) -> orthopoly1d: ...
def sh_chebyt(n: spt.AnyInt, monic: bool = ...) -> orthopoly1d: ...
def sh_chebyu(n: spt.AnyInt, monic: bool = ...) -> orthopoly1d: ...
def sh_jacobi(n: spt.AnyInt, p: spt.AnyReal, q: spt.AnyReal, monic: bool = ...) -> orthopoly1d: ...

# These functions are not public, but still need stubs because they
# get checked in the tests.
def _roots_hermite_asy(n: spt.AnyInt) -> _PointsWeights: ...

p_roots = roots_legendre
t_roots = roots_chebyt
u_roots = roots_chebyu
c_roots = roots_chebyc
s_roots = roots_chebys
j_roots = roots_jacobi
l_roots = roots_laguerre
la_roots = roots_genlaguerre
h_roots = roots_hermite
he_roots = roots_hermitenorm
cg_roots = roots_gegenbauer
ps_roots = roots_sh_legendre
ts_roots = roots_sh_chebyt
us_roots = roots_sh_chebyu
js_roots = roots_sh_jacobi
