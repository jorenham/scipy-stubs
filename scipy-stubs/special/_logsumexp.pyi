import numpy as np
import numpy.typing as npt

__all__ = ["log_softmax", "logsumexp", "softmax"]

def logsumexp(
    a: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = ...,
    b: npt.ArrayLike | None = ...,
    keepdims: bool = ...,
    return_sign: bool = ...,
) -> npt.NDArray[np.float64]: ...
def softmax(x: npt.ArrayLike, axis: int | tuple[int, ...] | None = ...) -> npt.NDArray[np.float64]: ...
def log_softmax(x: npt.ArrayLike, axis: int | tuple[int, ...] | None = ...) -> npt.NDArray[np.float64]: ...
