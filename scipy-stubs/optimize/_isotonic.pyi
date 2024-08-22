import numpy.typing as npt

from ._optimize import OptimizeResult as OptimizeResult
from ._pava_pybind import pava as pava

def isotonic_regression(y: npt.ArrayLike, *, weights: npt.ArrayLike | None = None, increasing: bool = True) -> OptimizeResult: ...
