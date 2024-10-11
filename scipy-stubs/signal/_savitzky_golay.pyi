from scipy._typing import Untyped, UntypedArray

def savgol_coeffs(
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    pos: int | None = None,
    use: str = "conv",
) -> UntypedArray: ...
def savgol_filter(
    x: Untyped,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    axis: int = -1,
    mode: str = "interp",
    cval: float = 0.0,
) -> UntypedArray: ...
