import numpy as np
import numpy.typing as npt
from scipy.interpolate import PPoly
from ._rotation import Rotation

__all__ = ["RotationSpline"]

class RotationSpline:
    MAX_ITER: int
    TOL: float
    times: npt.NDArray[np.float64]
    rotations: Rotation
    interpolator: PPoly
    def __init__(self, times: npt.ArrayLike, rotations: Rotation) -> None: ...
    def __call__(self, times: npt.ArrayLike, order: int = ...) -> Rotation | npt.NDArray[np.float64]: ...
