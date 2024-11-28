import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from scipy.interpolate import PPoly
from ._rotation import Rotation

class RotationSpline:
    MAX_ITER: int
    TOL: float
    times: onp.ArrayND[np.float64]
    rotations: Rotation
    interpolator: PPoly
    def __init__(self, /, times: npt.ArrayLike, rotations: Rotation) -> None: ...
    def __call__(self, /, times: npt.ArrayLike, order: int = ...) -> Rotation | onp.ArrayND[np.float64]: ...
