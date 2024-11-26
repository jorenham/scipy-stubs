from typing import Any
from typing_extensions import override

import numpy as np
import optype.numpy as onp
from scipy._typing import Untyped
from .interpnd import CloughTocher2DInterpolator, LinearNDInterpolator, NDInterpolatorBase

__all__ = ["CloughTocher2DInterpolator", "LinearNDInterpolator", "NearestNDInterpolator", "griddata"]

class NearestNDInterpolator(NDInterpolatorBase):
    tree: Untyped
    values: Untyped
    @override
    def __init__(
        self,
        /,
        x: onp.ToFloatND,
        y: onp.ToFloatND,
        rescale: bool = False,
        tree_options: Untyped | None = None,
    ) -> None: ...
    @override
    def __call__(self, /, *args: onp.ToFloatND, **query_options: Untyped) -> onp.ArrayND[np.floating[Any]]: ...

def griddata(
    points: onp.ToFloatND,
    values: onp.ToFloatND,
    xi: Untyped,
    method: str = "linear",
    fill_value: float = ...,
    rescale: bool = False,
) -> Untyped: ...
