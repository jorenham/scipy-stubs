# These stubs won't be used, because the implementation has type annotations, which unfortunately are incorrect.
# This again shows that "no type annotations than wrong ones".
# Anyway, these stubs are just for show, and perhaps could be used as an example of how to correctly annotate `_entropy.py`.
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy._typing as spt

__all__ = ["differential_entropy", "entropy"]

def entropy(
    pk: npt.ArrayLike,
    qk: npt.ArrayLike | None = None,
    base: float | None = None,
    axis: int = 0,
    *,
    nan_policy: spt.NanPolicy = "propagate",
    keepdims: bool = False,
) -> np.floating[Any] | npt.NDArray[np.floating[Any]]: ...
def differential_entropy(
    values: npt.ArrayLike,
    *,
    window_length: int | None = None,
    base: float | None = None,
    axis: int = 0,
    method: Literal["vasicek", "van es", "ebrahimi", "correa", "auto"] = "auto",
    nan_policy: spt.NanPolicy = "propagate",
    keepdims: bool = False,
) -> np.floating[Any] | npt.NDArray[np.floating[Any]]: ...
