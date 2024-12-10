import numpy as np
import optype.numpy as onp
from scipy.interpolate import PPoly
from ._rotation import Rotation

class RotationSpline:
    MAX_ITER: int
    TOL: float

    times: onp.Array1D[np.float64]
    rotations: Rotation
    interpolator: PPoly

    def __init__(self, /, times: onp.ToFloat1D, rotations: Rotation) -> None: ...
    def __call__(self, /, times: onp.ToFloat1D, order: int = ...) -> Rotation | onp.ArrayND[np.float64]: ...
