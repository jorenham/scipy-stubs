from scipy._typing import Untyped

def savgol_coeffs(
    window_length,
    polyorder,
    deriv: int = 0,
    delta: float = 1.0,
    pos: Untyped | None = None,
    use: str = "conv",
) -> Untyped: ...
def savgol_filter(
    x,
    window_length,
    polyorder,
    deriv: int = 0,
    delta: float = 1.0,
    axis: int = -1,
    mode: str = "interp",
    cval: float = 0.0,
) -> Untyped: ...
