# These stubs won't be used, because the implementation has type annotations, which unfortunately are incorrect.
# This again shows that "no type annotations than wrong ones".
# Anyway, these stubs are just for show, and perhaps could be used as an example of how to correctly annotate `_entropy.py`.
from typing import Any, Literal

import numpy as np
import optype as op
import optype.numpy as onp
import scipy._typing as spt

__all__ = ["differential_entropy", "entropy"]

def entropy(
    pk: onp.ToFloatND,
    qk: onp.ToFloatND | None = None,
    base: onp.ToFloat | None = None,
    axis: int = 0,
    *,
    nan_policy: spt.NanPolicy = "propagate",
    keepdims: onp.ToBool = False,
) -> float | np.floating[Any] | onp.ArrayND[np.floating[Any]]: ...
def differential_entropy(
    values: onp.ToFloatND,
    *,
    window_length: onp.ToInt | None = None,
    base: onp.ToFloat | None = None,
    axis: op.CanIndex = 0,
    method: Literal["vasicek", "van es", "ebrahimi", "correa", "auto"] = "auto",
    nan_policy: spt.NanPolicy = "propagate",
    keepdims: onp.ToBool = False,
) -> float | np.floating[Any] | onp.ArrayND[np.floating[Any]]: ...
