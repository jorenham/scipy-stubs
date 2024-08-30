import numpy.typing as npt
from ._optimize import OptimizeResult

__all__ = ["isotonic_regression"]

def isotonic_regression(y: npt.ArrayLike, *, weights: npt.ArrayLike | None = None, increasing: bool = True) -> OptimizeResult: ...
