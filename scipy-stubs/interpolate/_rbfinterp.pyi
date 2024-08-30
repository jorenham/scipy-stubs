from scipy._typing import Untyped

__all__ = ["RBFInterpolator"]

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
        y: Untyped,
        d: Untyped,
        neighbors: Untyped | None = None,
        smoothing: float = 0.0,
        kernel: str = "thin_plate_spline",
        epsilon: Untyped | None = None,
        degree: Untyped | None = None,
    ) -> None: ...
    def __call__(self, x: Untyped) -> Untyped: ...
