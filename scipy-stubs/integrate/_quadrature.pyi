from typing import NamedTuple
from typing_extensions import deprecated

import numpy.typing as npt
from scipy._typing import Untyped, UntypedCallable, UntypedTuple
from scipy.stats.qmc import QMCEngine

__all__ = [
    "AccuracyWarning",
    "cumulative_simpson",
    "cumulative_trapezoid",
    "fixed_quad",
    "newton_cotes",
    "qmc_quad",
    "quadrature",
    "romb",
    "romberg",
    "simpson",
    "trapezoid",
]

class AccuracyWarning(Warning): ...

class QMCQuadResult(NamedTuple):
    integral: float
    standard_error: float

@deprecated("deprecated as of SciPy 1.12.0 and will be removed in SciPy 1.15.0")
def quadrature(*args: Untyped, **kwargs: Untyped) -> UntypedTuple: ...
@deprecated("deprecated as of SciPy 1.12.0 and will be removed in SciPy 1.15.0")
def romberg(*args: Untyped, **kwargs: Untyped) -> Untyped: ...
def trapezoid(y: npt.ArrayLike, x: npt.ArrayLike | None = None, dx: float = 1.0, axis: int = -1) -> Untyped: ...
def fixed_quad(func: UntypedCallable, a: float, b: float, args: UntypedTuple = (), n: int = 5) -> Untyped: ...
def cumulative_trapezoid(
    y: npt.ArrayLike,
    x: npt.ArrayLike | None = None,
    dx: float = 1.0,
    axis: int = -1,
    initial: Untyped | None = None,
) -> Untyped: ...
def simpson(y: npt.ArrayLike, *, x: npt.ArrayLike | None = None, dx: float = 1.0, axis: int = -1) -> Untyped: ...
def cumulative_simpson(
    y: npt.ArrayLike,
    *,
    x: npt.ArrayLike | None = None,
    dx: float = 1.0,
    axis: int = -1,
    initial: Untyped | None = None,
) -> Untyped: ...
def romb(y: npt.ArrayLike, dx: float = 1.0, axis: int = -1, show: bool = False) -> Untyped: ...
def newton_cotes(rn: int, equal: int = 0) -> Untyped: ...
def qmc_quad(
    func: UntypedCallable,
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    n_estimates: int = 8,
    n_points: int = 1024,
    qrng: QMCEngine | None = None,
    log: bool = False,
) -> Untyped: ...
def tupleset(t: UntypedTuple, i: int, value: Untyped) -> UntypedTuple: ...
