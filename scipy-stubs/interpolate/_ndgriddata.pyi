from typing import Any
from typing_extensions import override

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
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
        x: onpt.AnyIntegerArray | onpt.AnyFloatingArray,
        y: onpt.AnyFloatingArray,
        rescale: bool = False,
        tree_options: Untyped | None = None,
    ) -> None: ...
    @override
    def __call__(self, /, *args: onpt.AnyFloatingArray, **query_options: Untyped) -> npt.NDArray[np.floating[Any]]: ...

def griddata(
    points: onpt.AnyIntegerArray | onpt.AnyFloatingArray,
    values: onpt.AnyFloatingArray,
    xi: Untyped,
    method: str = "linear",
    fill_value: float = ...,
    rescale: bool = False,
) -> Untyped: ...
