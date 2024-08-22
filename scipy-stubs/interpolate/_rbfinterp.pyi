from scipy._typing import Untyped
from scipy.linalg.lapack import dgesv as dgesv
from scipy.spatial import KDTree as KDTree
from scipy.special import comb as comb

class RBFInterpolator:
    y: Untyped
    d: Untyped
    d_shape: Untyped
    d_dtype: Untyped
    neighbors: Untyped
    smoothing: Untyped
    kernel: Untyped
    epsilon: Untyped
    powers: Untyped
    def __init__(
        self,
        y,
        d,
        neighbors: Untyped | None = None,
        smoothing: float = 0.0,
        kernel: str = "thin_plate_spline",
        epsilon: Untyped | None = None,
        degree: Untyped | None = None,
    ): ...
    def __call__(self, x) -> Untyped: ...
