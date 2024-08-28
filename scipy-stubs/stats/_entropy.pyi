# These stubs won't be used, because the implementation has type annotations, which unfortunately are incorrect.
# This again shows that "no type annotations than wrong ones".
# Anyway, these stubs are just for show, and perhaps could be used as an example of how to correctly annotate `_entropy.py`.
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

__all__ = ["differential_entropy", "entropy"]

def entropy(
    pk: npt.ArrayLike,
    qk: npt.ArrayLike | None = None,
    base: float | None = None,
    axis: int = 0,
) -> np.floating[Any] | npt.NDArray[np.floating[Any]]: ...
def differential_entropy(
    values: npt.ArrayLike,
    *,
    window_length: int | None = None,
    base: float | None = None,
    axis: int = 0,
    method: Literal["vasicek", "van es", "ebrahimi", "correa", "auto"] = "auto",
) -> np.floating[Any] | npt.NDArray[np.floating[Any]]: ...
